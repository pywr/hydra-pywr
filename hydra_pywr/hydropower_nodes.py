from pywr.nodes import Link, Storage
from pywr.parameters.control_curves import ControlCurveInterpolatedParameter
from pywr.parameters._thresholds import ParameterThresholdParameter
from pywr.parameters import InterpolatedVolumeParameter, ConstantParameter, Parameter
from pywr.parameters._hydropower import HydropowerTargetParameter
from pywr.recorders import HydropowerRecorder
from pywr.schema import NodeSchema, fields
from pywr.domains.river import Catchment
import numpy as np
import marshmallow
from .parameters import MonthlyArrayIndexedParameter


class LinearStorageReleaseControl(Link):
    """A specialised node that provides a default max_flow based on a release rule.


    """
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)
        storage_node = fields.NodeField()
        release_values = marshmallow.fields.List(marshmallow.fields.List(marshmallow.fields.Number))

    def __init__(self, model, name, storage_node, release_values, **kwargs):

        control_curves = []
        values = []
        for (cc, v) in release_values:
            control_curves.append(cc)
            values.append(v)
        control_curves = control_curves[1:-1]

        max_flow_param = ControlCurveInterpolatedParameter(model, storage_node, control_curves, values)
        super().__init__(model, name, max_flow=max_flow_param, **kwargs)


class Turbine(Link):

    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        min_flow = fields.ParameterReferenceField(allow_none=True)
        max_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)

        storage_node = fields.NodeField()
        turbine_elevation = marshmallow.fields.Number()
        generation_capacity = marshmallow.fields.Number()
        min_operating_elevation = marshmallow.fields.Number()
        efficiency = marshmallow.fields.Number()
        density = marshmallow.fields.Number()
        flow_unit_conversion = marshmallow.fields.Number(default=1.0)
        energy_unit_conversion = marshmallow.fields.Number(default=1e-6)

    def __init__(self, model, name, **kwargs):

        # Create the keyword arguments for the HP recorder
        hp_recorder_kwarg_names = ('efficiency', 'density', 'flow_unit_conversion', 'energy_unit_conversion')
        hp_kwargs = {}
        for kwd in hp_recorder_kwarg_names:
            try:
                hp_kwargs[kwd] = kwargs.pop(kwd)
            except KeyError:
                pass

        self.storage_node = storage_node = kwargs.pop('storage_node', None)
        # See if there is a level parameter on the storage node
        if storage_node is not None:
            level_parameter = storage_node.level
            if not isinstance(level_parameter, Parameter):
                level_parameter = ConstantParameter(model, value=level_parameter)
        else:
            level_parameter = None

        turbine_elevation = kwargs.pop('turbine_elevation', 0)
        generation_capacity = kwargs.pop('generation_capacity', 0)
        min_operating_elevation = kwargs.pop('min_operating_elevation', 0)
        min_head = min_operating_elevation - turbine_elevation

        super().__init__(model, name, **kwargs)

        if isinstance(generation_capacity, (float, int)):
            generation_capacity = ConstantParameter(model, generation_capacity)

        hp_target_flow = HydropowerTargetParameter(model, generation_capacity,
                                                   water_elevation_parameter=level_parameter,
                                                   min_head=min_head,
                                                   turbine_elevation=turbine_elevation,
                                                   **{k: v for k, v in hp_kwargs.items()})

        self.max_flow = hp_target_flow

        hp_recorder = HydropowerRecorder(model, self, water_elevation_parameter=level_parameter,
                                         **hp_kwargs)
        self.hydropower_recorder = hp_recorder


class Reservoir(Storage):
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        max_volume = fields.ParameterReferenceField(required=False)
        min_volume = fields.ParameterReferenceField(required=False)
        cost = fields.ParameterReferenceField(required=False)
        initial_volume = fields.ParameterValuesField(required=False)
        initial_volume_pc = marshmallow.fields.Number(required=False)
        bathymetry = marshmallow.fields.List(marshmallow.fields.List(marshmallow.fields.Number),
                                             required=False)

    def __init__(self, model, name, **kwargs):

        bathymetry = kwargs.pop('bathymetry', None)

        super().__init__(model, name, **kwargs)
        if bathymetry is not None:
            self._set_bathymetry(bathymetry)

    def _set_bathymetry(self, values):

        values = np.array(values)
        volumes = values[:, 0]
        levels = values[:, 1]
        areas = values[:, 2]

        self.level = InterpolatedVolumeParameter(self.model, self, volumes, levels)
        self.area = InterpolatedVolumeParameter(self.model, self, volumes, areas)


class MonthlyCatchment(Catchment):
    class Schema(NodeSchema):
        flow = marshmallow.fields.List(marshmallow.fields.Number)

    def __init__(self, model, name, **kwargs):
        flow_values = kwargs.pop('flow')
        flow_param = MonthlyArrayIndexedParameter(model, flow_values)
        super().__init__(model, name, flow=flow_param, **kwargs)
