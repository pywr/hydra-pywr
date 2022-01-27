import pandas as pd
import numpy as np

from pywr.nodes import (
    NodeMeta,
    Link,
    Input,
    AggregatedNode,
    Storage
)

from pywr.parameters import (
    load_parameter,
    ScenarioWrapperParameter,
    InterpolatedVolumeParameter,
    ConstantParameter,
    HydropowerTargetParameter,
	AggregatedParameter,
	MonthlyProfileParameter
)

from pywr.recorders import (
    HydropowerRecorder
)

from pywr.parameters.control_curves import (
    ControlCurveInterpolatedParameter
)

import logging
log = logging.getLogger(__name__)

""" Define nodes exported to `from hydra_pywr.nodes import *` """
__all__ = (
    "ProportionalInput",
    "LinearStorageReleaseControl",
    "Reservoir",
    "Turbine"
)

class ProportionalInput(Input, metaclass=NodeMeta):
    min_proportion = 1e-6

    def __init__(self, model, name, node, proportion, **kwargs):
        super().__init__(model, name, **kwargs)

        self.node = model._get_node_from_ref(model, node)

        # Create the flow factors for the other node and self
        if proportion < self.__class__.min_proportion:
            self.max_flow = 0.0
        else:
            factors = [1, proportion]
            # Create the aggregated node to apply the factors.
            # factors no longer accepted by ctor
            #self.aggregated_node = AggregatedNode(model, f'{name}.aggregated', [node, self], factors=factors)
            self.aggregated_node = AggregatedNode(model, f'{name}.aggregated', [self.node, self])
            self.aggregated_node.factors = factors


class LinearStorageReleaseControl(Link, metaclass=NodeMeta):
    """ A specialised node that provides a default max_flow based on a release rule. """

    def __init__(self, model, name, storage_node, release_values, scenario=None, **kwargs):

        release_values = pd.DataFrame.from_dict(release_values)
        storage_node = model._get_node_from_ref(model, storage_node)

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


class Reservoir(Storage, metaclass=NodeMeta):
    def __init__(self, model, name, **kwargs):
        bathymetry = kwargs.pop('bathymetry', None)
        volume = kwargs.pop('volume', None)
        level = kwargs.pop('level', None)
        area = kwargs.pop('area', None)
        const = kwargs.pop('const', 1e6 * 1e-3 * 1e-6)

        # Pywr Storage does not expect a 'weather' kwargs, so move this to instance
        self.weather = kwargs.pop("weather", None)

        super().__init__(model, name, **kwargs)

        self._set_bathymetry(model, bathymetry, volume, level, area)
        self.const = ConstantParameter(model, const)

        self.rainfall_node = None
        self.rainfall_recorder = None
        self.evaporation_node = None
        self.evaporation_recorder = None

    def _set_bathymetry(self, model, bathymetry, volume=None, level=None, area=None):
        if bathymetry is not None:
            if isinstance(bathymetry, str):
                bathymetry = load_parameter(model, bathymetry)
                volumes = bathymetry.dataframe['volume'].astype(np.float64)
                levels = bathymetry.dataframe['level'].astype(np.float64)
                areas = bathymetry.dataframe['area'].astype(np.float64)
            else:
                bathymetry = pd.DataFrame.from_dict(bathymetry)
                volumes = bathymetry['volume'].astype(np.float64)
                levels = bathymetry['level'].astype(np.float64)
                areas = bathymetry['area'].astype(np.float64)

            #breakpoint()
        elif volume is not None and level is not None and area is not None:
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


    def _make_weather_nodes(self, model, weather, cost):

        if not isinstance(self.area, Parameter):
            raise ValueError('Weather nodes can only be created if an area Parameter is given.')

        weather = pd.DataFrame.from_dict(weather)

        rainfall = weather['rainfall'].astype(np.float64)
        evaporation = weather['evaporation'].astype(np.float64)

        self._make_evaporation_node(model, evaporation, cost)
        self._make_rainfall_node(model, rainfall, cost)


    def _make_evaporation_node(self, model, evaporation, cost):

        if not isinstance(self.area, Parameter):
            log.warning('Evaporation nodes be created only if an area Parameter is given.')
            return

        if evaporation is None:
            try:
                evaporation_param = load_parameter(model, f'__{self.name}__:evaporation')
            except KeyError:
                log.warning(f"Please specify evaporation or weather on node {self.name}")
                return
        elif isinstance(evaporation, pd.DataFrame) or isinstance(evaporation, pd.Series):
            evaporation = evaporation.astype(np.float64)
            evaporation_param = MonthlyProfileParameter(model, evaporation)
        else:
            evaporation_param = evaporation

        evaporation_flow_param = AggregatedParameter(model, [evaporation_param, self.const, self.area],
                                                     agg_func='product')

        evaporation_node = Output(model, '{}.evaporation'.format(self.name), parent=self)
        evaporation_node.max_flow = evaporation_flow_param
        evaporation_node.cost = cost

        self.connect(evaporation_node)
        self.evaporation_node = evaporation_node

        self.evaporation_recorder = NumpyArrayNodeRecorder(model, evaporation_node,
                                                           name=f'__{evaporation_node.name}__:evaporation')

    def _make_rainfall_node(self, model, rainfall, cost):

        if not isinstance(self.area, Parameter):
            log.warning('Weather nodes can be created only if an area Parameter is given.')
            return

        if rainfall is None:
            try:
                rainfall_param = load_parameter(model, f'__{self.name}__:rainfall')
            except KeyError:
                log.warning(f"Please specify rainfall or weather on node {self.name}")
                return
        elif isinstance(rainfall, pd.DataFrame) or isinstance(rainfall, pd.Series):
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

        self.rainfall_recorder = NumpyArrayNodeRecorder(model, rainfall_node,
                                                        name=f'__{rainfall_node.name}__:rainfall')



class Turbine(Link, metaclass=NodeMeta):
    def __init__(self, model, name, **kwargs):
        hp_recorder_kwarg_names = ('efficiency', 'density', 'flow_unit_conversion', 'energy_unit_conversion')
        hp_kwargs = {}
        for kwd in hp_recorder_kwarg_names:
            try:
                hp_kwargs[kwd] = kwargs.pop(kwd)
            except KeyError:
                pass

        level_parameter = None
        storage_node = kwargs.pop("storage_node", None)

        if storage_node is not None:
            storage_node = model._get_node_from_ref(model, storage_node)
            if hasattr(storage_node, "level") and storage_node.level is not None:
                level_parameter = ConstantParameter(model, value=storage_node.level)

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
