from pywr.nodes import Link, Storage, Output, Input, AggregatedNode
from pywr.parameters.control_curves import ControlCurveInterpolatedParameter
from pywr.parameters._thresholds import ParameterThresholdParameter
from pywr.parameters import InterpolatedVolumeParameter, ConstantParameter, Parameter, MonthlyProfileParameter
from pywr.parameters._hydropower import HydropowerTargetParameter
from pywr.recorders import HydropowerRecorder
from pywr.schema import NodeSchema, fields
from pywr.domains.river import Catchment
import numpy as np
import pandas
import marshmallow
from .parameters import MonthlyArrayIndexedParameter


# This is a parameter instance.
class DataFrameField(marshmallow.fields.Field):
    """ Marshmallow field representing a Parameter. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj):
        return value.to_json()

    def _deserialize(self, value, attr, data):
        return pandas.DataFrame.from_dict(value)



class LinearStorageReleaseControl(Link):
    """A specialised node that provides a default max_flow based on a release rule.


    """
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)
        storage_node = fields.NodeField()
        release_values = DataFrameField()

    def __init__(self, model, name, storage_node, release_values, **kwargs):

        control_curves = release_values['volume'].iloc[1:-1].astype(np.float64)
        values = release_values['value'].astype(np.float64)

        max_flow_param = ControlCurveInterpolatedParameter(model, storage_node, control_curves, values)
        super().__init__(model, name, max_flow=max_flow_param, **kwargs)


class Turbine(Link):

    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)

        storage_node = fields.NodeField()
        turbine_elevation = marshmallow.fields.Number()
        generation_capacity = marshmallow.fields.Number()
        min_operating_elevation = marshmallow.fields.Number()
        efficiency = marshmallow.fields.Number()
        density = marshmallow.fields.Number()

        # Defaults here are for inputting for Mm3/day (1e6) and MW (1.15741e-11)
        flow_unit_conversion = marshmallow.fields.Number(default=1e6)
        energy_unit_conversion = marshmallow.fields.Number(default=1.15741e-11)

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
                                                   min_head=min_head, min_flow=ConstantParameter(model, 0),
                                                   turbine_elevation=turbine_elevation,
                                                   **{k: v for k, v in hp_kwargs.items()})

        self.max_flow = hp_target_flow

        hp_recorder = HydropowerRecorder(model, self, water_elevation_parameter=level_parameter,
                                         turbine_elevation=turbine_elevation, **hp_kwargs)
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
        bathymetry = DataFrameField()

    def __init__(self, model, name, **kwargs):

        bathymetry = kwargs.pop('bathymetry', None)

        super().__init__(model, name, **kwargs)
        if bathymetry is not None:
            self._set_bathymetry(bathymetry)

    def _set_bathymetry(self, values):
        volumes = values['volume'].astype(np.float64)
        levels = values['level'].astype(np.float64)
        areas = values['area'].astype(np.float64)

        self.level = InterpolatedVolumeParameter(self.model, self, volumes, levels)
        self.area = InterpolatedVolumeParameter(self.model, self, volumes, areas)


class MonthlyCatchment(Catchment):
    class Schema(NodeSchema):
        flow = DataFrameField()

    def __init__(self, model, name, **kwargs):
        flow_values = kwargs.pop('flow')
        flow_param = MonthlyArrayIndexedParameter(model, flow_values.iloc[:, 0].values.astype(np.float64))
        super().__init__(model, name, flow=flow_param, **kwargs)


class MonthlyOutput(Output):
    class Schema(NodeSchema):
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)
        max_flow = DataFrameField()

    def __init__(self, model, name, **kwargs):
        flow_values = kwargs.pop('max_flow')
        flow_param = MonthlyProfileParameter(model, flow_values.iloc[:, 0].values.astype(np.float64))
        super().__init__(model, name, **kwargs)
        self.max_flow = flow_param


class ProportionalInput(Input):
    class Schema(NodeSchema):
        node = fields.NodeField()
        proportion = marshmallow.fields.Number()

    def __init__(self, model, name, node, proportion, **kwargs):
        super().__init__(model, name, **kwargs)

        self.node = node
        # Create the flow factors for the other node and self
        factors = [1-proportion, proportion]
        # Create the aggregated node to apply the factors.
        self.aggregated_node = AggregatedNode(model, f'{name}.aggregated', [node, self], factors=factors)


class MonthlyOutputWithReturn(MonthlyOutput):
    class Schema(NodeSchema):
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)
        max_flow = DataFrameField()
        proportion = marshmallow.fields.Number()

    def __init__(self, model, name, proportion, **kwargs):
        super().__init__(model, name, **kwargs)
        self.input = ProportionalInput(model, '{}.input'.format(name), self, proportion)

    def iter_slots(self, slot_name=None, is_connector=True):
        if is_connector:
            yield self.input
        else:
            yield self
