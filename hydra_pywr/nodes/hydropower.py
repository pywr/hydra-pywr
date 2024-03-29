from pywr.nodes import Link, Storage, Output, Input, AggregatedNode
from pywr.parameters.control_curves import ControlCurveInterpolatedParameter
from pywr.parameters._thresholds import ParameterThresholdParameter
from pywr.parameters import InterpolatedVolumeParameter, ConstantParameter, Parameter, MonthlyProfileParameter, \
    AggregatedParameter, ScenarioWrapperParameter
from pywr.parameters._hydropower import HydropowerTargetParameter
from pywr.recorders import HydropowerRecorder, NumpyArrayLevelRecorder, NumpyArrayNodeRecorder
from pywr.schema import NodeSchema, fields
from pywr.domains.river import Catchment
from pywr.parameters import load_parameter
import numpy as np
import pandas as pd
import marshmallow
from ..parameters import MonthlyArrayIndexedParameter
from . import DataFrameField

import logging
log = logging.getLogger(__name__)


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

        #Weather and rainfall / evaporation are mutually exclusive. If both
        #are specified, then weather will be used (to simplify logic because
        #the the other way around requires checking that both rainfall & evaporation
        #are set

        weather = DataFrameField()
        weather_cost = fields.ParameterReferenceField(required=False)

        rainfall = fields.ParameterReferenceField(required=False)
        rainfall_cost = fields.ParameterReferenceField(required=False)
        evaporation = fields.ParameterReferenceField(required=False)
        evaporation_cost = fields.ParameterReferenceField(required=False)

    def __init__(self, model, name, **kwargs):

        #bathymetry can be specified as a single dataframe or as 3 individual
        #dataframes
        bathymetry = kwargs.pop('bathymetry', None)
        #bathymetry -- these are used if the combined 'bathymetry' dataframe is not set
        #NOTE: These will only appear as not NOne if they are not specified as
        #parameters. i.e. if they are specified as parameters, their value needs to be
        #extracted using the load_parameter() function
        volume = kwargs.pop('volume', None)
        level = kwargs.pop('level', None)
        area = kwargs.pop('area', None)


        weather = kwargs.pop('weather', None)
        evaporation = kwargs.pop('evaporation', None)
        rainfall = kwargs.pop('rainfall', None)
        weather_cost = kwargs.pop('weather_cost', -999)
        evaporation_cost = kwargs.pop('evaporation_cost', -999)
        rainfall_cost = kwargs.pop('rainfall_cost', -999)
        # Assume rainfall/evap is mm/day
        # Need to convert:
        #   Mm2 -> m2
        #   mm/day -> m/day
        #   m3/day -> Mm3/day
        # TODO allow this to be configured
        const = kwargs.pop('const', 1e6 * 1e-3 * 1e-6)

        super().__init__(model, name, **kwargs)

        self._set_bathymetry(model, bathymetry, volume, level, area)

        self.const = ConstantParameter(model, const)

        self.rainfall_node = None
        self.rainfall_recorder = None
        self.evaporation_node = None
        self.evaporation_recorder = None
        if weather is not None:
            self._make_weather_nodes(model, weather, weather_cost)
        else:
            self._make_evaporation_node(model, evaporation, evaporation_cost)
            self._make_rainfall_node(model, rainfall, rainfall_cost)

    def _set_bathymetry(self, model, bathymetry, volume=None, level=None, area=None):
        if bathymetry is not None:
            volumes = bathymetry['volume'].astype(np.float64)
            levels = bathymetry['level'].astype(np.float64)
            areas = bathymetry['area'].astype(np.float64)
        elif volume is not None and level is not None and area is not None:
            #cater for when they are not specified as parameters, eg they are
            #scalars or standard dataframes
            volumes = volume
            levels = level
            areas = area
        else:
            try:
                volumes = load_parameter(model, f'__{self.name}__:volume')
            except KeyError:
                log.warning(f"Please specify a bathymetry or volume on node {self.name}")
                volumes = None
            try:
                areas = load_parameter(model, f'__{self.name}__:area')
            except KeyError:
                log.warning(f"Please specify a bathymetry or area on node {self.name}")
                areas = None
            try:
                levels = load_parameter(model, f'__{self.name}__:level')
            except KeyError:
                log.warning(f"Please specify a bathymetry or level on node {self.name}")
                levels = None

        if volumes is not None and levels is not None:
            self.level = InterpolatedVolumeParameter(self.model, self, volumes, levels)

        if volumes is not None and areas is not None:
            self.area = InterpolatedVolumeParameter(self.model, self, volumes, areas)

        # Record the level
        # TODO Pywr is missing an equivalent area recorder.
        # TODO fix bug in Pywr with missing `to_dataframe` method (see https://github.com/pywr/pywr/issues/670)
        # self.level_recorder = NumpyArrayLevelRecorder(self.model, self, name=f'__{self.name}__:level')

    def _make_weather_nodes(self, model, weather, cost):

        if not isinstance(self.area, Parameter):
            raise ValueError('Weather nodes can only be created if an area Parameter is given.')

        rainfall = weather['rainfall'].astype(np.float64)
        evaporation = weather['evaporation'].astype(np.float64)

        self._make_evaporation_node(model, evaporation, cost)
        self._make_rainfall_node(model, rainfall, cost)

    def _make_evaporation_node(self, model, evaporation, cost):

        if not isinstance(self.area, Parameter):
            log.warning('Evaporation nodes can only be created if an area Parameter is given.')
            return

        if evaporation is None:
            try:
                evaporation_param = load_parameter(model, f'__{self.name}__:evaporation')
            except KeyError:
                log.warning(f"Please speficy an evaporation or a weather on node {self.name}")
                return
        elif isinstance(evaporation, pd.DataFrame) or isinstance(evaporation, pd.Series):
            evaporation = evaporation.astype(np.float64)
            evaporation_param = MonthlyProfileParameter(model, evaporation)
        else:
            evaporation_param = evaporation

        evaporation_flow_param = AggregatedParameter(model, [evaporation_param, self.const, self.area],
                                                     agg_func='product')

        evporation_node = Output(model, '{}.evaporation'.format(self.name), parent=self)
        evporation_node.max_flow = evaporation_flow_param
        evporation_node.cost = cost

        self.connect(evporation_node)
        self.evaporation_node = evporation_node

        self.evaporation_recorder = NumpyArrayNodeRecorder(model, evporation_node,
                                                           name=f'__{evporation_node.name}__:evaporation')

    def _make_rainfall_node(self, model, rainfall, cost):

        if not isinstance(self.area, Parameter):
            log.warning('Weather nodes can only be created if an area Parameter is given.')
            return

        if rainfall is None:
            try:
                rainfall_param = load_parameter(model, f'__{self.name}__:rainfall')
            except KeyError:
                log.warning(f"Please speficy a rainfall or a weather on node {self.name}")
                return
        elif isinstance(rainfall, pd.DataFrame) or isinstance(rainfall, pd.Series):
            #assume it's a dataframe
            rainfall = rainfall.astype(np.float64)
            rainfall_param = MonthlyProfileParameter(model, rainfall)
        else:
            rainfall_param = rainfall

        # Create the flow parameters multiplying area by rate of rainfall/evap
        rainfall_flow_param = AggregatedParameter(model, [rainfall_param, self.const, self.area],
                                                  agg_func='product')

        # Create the nodes to provide the flows
        rainfall_node = Input(model, '{}.rainfall'.format(self.name), parent=self)
        rainfall_node.max_flow = rainfall_flow_param
        rainfall_node.cost = cost

        rainfall_node.connect(self)
        self.rainfall_node = rainfall_node

        # Finally record these flows
        self.rainfall_recorder = NumpyArrayNodeRecorder(model, rainfall_node,
                                                        name=f'__{rainfall_node.name}__:rainfall')
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
        if flow_values is not None and int(flow_values.index[-1]) != len(flow_values.index):
            raise Exception("Invalid dataframe")

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

