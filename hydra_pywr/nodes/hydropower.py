from pywr.nodes import Link, Storage, Output, Input, AggregatedNode
from pywr.parameters.control_curves import ControlCurveInterpolatedParameter
from pywr.parameters._thresholds import ParameterThresholdParameter
from pywr.parameters import InterpolatedVolumeParameter, ConstantParameter, Parameter, MonthlyProfileParameter, \
    AggregatedParameter, ScenarioWrapperParameter
from pywr.parameters._hydropower import HydropowerTargetParameter
from pywr.recorders import HydropowerRecorder, NumpyArrayLevelRecorder, NumpyArrayNodeRecorder
from pywr.schema import NodeSchema, fields
from pywr.domains.river import Catchment
import numpy as np
import pandas
import marshmallow
from ..parameters import MonthlyArrayIndexedParameter
from . import DataFrameField


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
        scenario = fields.ScenarioReferenceField(allow_none=True)

    def __init__(self, model, name, storage_node, release_values, scenario=None, **kwargs):

        if scenario is None:
            # Only one control curve should be defined. Get it explicitly
            control_curves = release_values['volume'].iloc[1:-1].astype(np.float64)
            values = release_values['value'].astype(np.float64)
            max_flow_param = ControlCurveInterpolatedParameter(model, storage_node, control_curves, values)

        else:
            # There should be multiple control curves defined.
            if release_values.shape[1] % 2 != 0:
                raise ValueError("An even number of columns (i.e. pairs) is required for the release rules "
                                 "when using a scenario.")

            ncurves = release_values.shape[1] // 2
            if ncurves != scenario.size:
                raise ValueError(f"The number of curves ({ncurves}) should equal the size of the "
                                 f"scenario ({scenario.size}).")

            curves = []
            for i in range(ncurves):
                volume = release_values.iloc[1:-1, i*2]
                values = release_values.iloc[:, i*2+1]
                control_curve = ControlCurveInterpolatedParameter(model, storage_node, volume, values)
                curves.append(control_curve)

            max_flow_param = ScenarioWrapperParameter(model, scenario, curves)
        self.scenario = scenario
        super().__init__(model, name, max_flow=max_flow_param, **kwargs)


class Turbine(Link):

    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        # TODO add turbine max_flow in here
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)

        storage_node = fields.NodeField()
        turbine_elevation = marshmallow.fields.Number()
        generation_capacity = marshmallow.fields.Number()
        min_operating_elevation = marshmallow.fields.Number()
        efficiency = marshmallow.fields.Number()
        density = marshmallow.fields.Number()

        # Default values for missing inputs.
        # Assume inputting for Mm3/day (1e6) and MW (1.15741e-11)
        flow_unit_conversion = marshmallow.fields.Number(missing=1e6)
        energy_unit_conversion = marshmallow.fields.Number(missing=1.15741e-11)

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
        weather = DataFrameField()

    def __init__(self, model, name, **kwargs):

        bathymetry = kwargs.pop('bathymetry', None)
        weather = kwargs.pop('weather', None)
        weather_cost = kwargs.pop('weather_cost', -999)

        super().__init__(model, name, **kwargs)
        if bathymetry is not None:
            self._set_bathymetry(bathymetry)

        self.rainfall_node = None
        self.evaporation_node = None
        if weather is not None:
            self._make_weather_nodes(model, weather, weather_cost)

    def _set_bathymetry(self, values):
        volumes = values['volume'].astype(np.float64)
        levels = values['level'].astype(np.float64)
        areas = values['area'].astype(np.float64)

        self.level = InterpolatedVolumeParameter(self.model, self, volumes, levels)
        self.area = InterpolatedVolumeParameter(self.model, self, volumes, areas)

        # Record the level
        # TODO Pywr is missing an equivalent area recorder.
        # TODO fix bug in Pywr with missing `to_dataframe` method (see https://github.com/pywr/pywr/issues/670)
        # self.level_recorder = NumpyArrayLevelRecorder(self.model, self, name=f'__{self.name}__:level')

    def _make_weather_nodes(self, model, weather, cost):

        if not isinstance(self.area, Parameter):
            raise ValueError('Weather nodes can only be created if an area Parameter is given.')

        rainfall = weather['rainfall'].astype(np.float64)
        rainfall_param = MonthlyProfileParameter(model, rainfall)

        evaporation = weather['evaporation'].astype(np.float64)
        evaporation_param = MonthlyProfileParameter(model, evaporation)

        # Assume rainfall/evap is mm/day
        # Need to convert:
        #   Mm2 -> m2
        #   mm/day -> m/day
        #   m3/day -> Mm3/day
        # TODO allow this to be configured
        const = ConstantParameter(model, 1e6 * 1e-3 * 1e-6)

        # Create the flow parameters multiplying area by rate of rainfall/evap
        rainfall_flow_param = AggregatedParameter(model, [rainfall_param, const, self.area],
                                                  agg_func='product')
        evaporation_flow_param = AggregatedParameter(model, [evaporation_param, const, self.area],
                                                     agg_func='product')

        # Create the nodes to provide the flows
        rainfall_node = Input(model, '{}.rainfall'.format(self.name), parent=self)
        rainfall_node.max_flow = rainfall_flow_param
        rainfall_node.cost = cost

        evporation_node = Output(model, '{}.evaporation'.format(self.name), parent=self)
        evporation_node.max_flow = evaporation_flow_param
        evporation_node.cost = cost

        rainfall_node.connect(self)
        self.connect(evporation_node)
        self.rainfall_node = rainfall_node
        self.evaporation_node = evporation_node

        # Finally record these flows
        self.rainfall_recorder = NumpyArrayNodeRecorder(model, rainfall_node,
                                                        name=f'__{rainfall_node.name}__:rainfall')
        self.evaporation_recorder = NumpyArrayNodeRecorder(model, evporation_node,
                                                           name=f'__{evporation_node.name}__:evaporation')


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
        scenario = fields.ScenarioReferenceField(allow_none=True)

    def __init__(self, model, name, scenario=None, **kwargs):
        flow_values = kwargs.pop('max_flow')
        try:
            if flow_values is not None and int(flow_values.index[-1]) != len(flow_values.index):
                raise Exception("Invalid dataframe")
        except:
            import pudb; pudb.set_trace()

        if scenario is None:
            flow_param = MonthlyProfileParameter(model, flow_values.iloc[:, 0].values.astype(np.float64))
        else:
            # There should be multiple control curves defined.
            nprofiles = flow_values.shape[1]
            if nprofiles != scenario.size:
                raise ValueError(f"The number of profiles ({nprofiles}) should equal the size of the "
                                 f"scenario ({scenario.size}).")

            profiles = []
            for i in range(nprofiles):
                profile = MonthlyProfileParameter(model, flow_values.iloc[:, i].values.astype(np.float64))
                profiles.append(profile)

            flow_param = ScenarioWrapperParameter(model, scenario, profiles)
        self.scenario = scenario

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
        if proportion < 1e-6:
            self.max_flow = 0.0
        else:
            factors = [1, proportion]
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


class WasteWaterTreatmentWorks(ProportionalInput):
    class Schema(NodeSchema):
        node = fields.NodeField()
        proportion = marshmallow.fields.Number()
        reuse_proportion = marshmallow.fields.Number()

    def __init__(self, model, name, *args, **kwargs):
        reuse_proportion = kwargs.pop('reuse_proportion')
        super().__init__(model, name, *args, **kwargs)

        effluent_node = Link(model, f'{name}.effluent', parent=self)
        reuse_node = Link(model, f'{name}.reuse', parent=self)
        self.connect(effluent_node)
        self.connect(reuse_node)

        reuse_factors = [1-reuse_proportion, reuse_proportion]
        self.reuse_aggregated_node = AggregatedNode(model, f'{name}.reuse_aggregated', factors=reuse_factors)

    def iter_slots(self, slot_name=None, is_connector=True):
        if is_connector:
            pass

