import pandas as pd
import numpy as np

from pywr.nodes import (
    NodeMeta,
    Link,
    Input,
    AggregatedNode,
    Storage,
    Output
)

from pywr.parameters import (
    Parameter,
    load_parameter,
    ScenarioWrapperParameter,
    InterpolatedVolumeParameter,
    ConstantParameter,
    HydropowerTargetParameter,
	AggregatedParameter,
	MonthlyProfileParameter
)

from pywr.recorders import (
    HydropowerRecorder,
    NumpyArrayNodeRecorder
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
    "Turbine",
    "MonthlyOutput"
)

class ProportionalInput(Input, metaclass=NodeMeta):
    min_proportion = 1e-6

    def __init__(self, model, name, node, proportion, **kwargs):
        super().__init__(model, name, **kwargs)

        self.node = model.pre_load_node(node)

        # Create the flow factors for the other node and self
        if proportion < self.__class__.min_proportion:
            self.max_flow = 0.0
        else:
            factors = [1, proportion]
            # Create the aggregated node to apply the factors.
            self.aggregated_node = AggregatedNode(model, f'{name}.aggregated', [self.node, self])
            self.aggregated_node.factors = factors


class LinearStorageReleaseControl(Link, metaclass=NodeMeta):
    """ A specialised node that provides a default max_flow based on a release rule. """

    def __init__(self, model, name, storage_node, release_values, scenario=None, **kwargs):

        if isinstance(release_values, str):
            release_values = load_parameter(model, release_values).dataframe.astype(np.float64)
        else:
            release_values = pd.DataFrame.from_dict(release_values)

        storage_node = model.pre_load_node(storage_node)

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

        self.max_flow = max_flow_param
        self.scenario = scenario
        super().__init__(model, name, max_flow=max_flow_param, **kwargs)

    @classmethod
    def pre_load(cls, model, data):
        name = data.pop("name")
        cost = data.pop("cost", 0.0)
        min_flow = data.pop("min_flow", None)

        node = cls(name=name, model=model, **data)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0

        node.cost = cost
        node.min_flow = min_flow

        """
            The Pywr Loadable base class contains a reference to
            `self.__parameters_to_load.items()` which will fail unless
            a pre-mangled name which matches the expected value from
            inside the Loadable class is added here.

            See pywr/nodes.py:80 Loadable.finalise_load()
        """
        setattr(node, "_Loadable__parameters_to_load", {})
        return node

class Reservoir(Storage, metaclass=NodeMeta):

    def __init__(self, model, name, **kwargs):
        bathymetry = kwargs.pop('bathymetry', None)
        volume = kwargs.pop('volume', None)
        level = kwargs.pop('level', None)
        area = kwargs.pop('area', None)
        self.weather_cost = kwargs.pop('weather_cost', -999)
        self.evaporation_cost = kwargs.pop('evaporation_cost', -999)
        self.rainfall_cost = kwargs.pop('rainfall_cost', -999)
        const = kwargs.pop('const', 1e6 * 1e-3 * 1e-6)

        # Pywr Storage does not expect a 'weather' kwargs, so move this to instance
        self.weather = kwargs.pop("weather", None)

        super().__init__(model, name, **kwargs)

        self.const = ConstantParameter(model, const)

        self.rainfall_node = None
        self.rainfall_recorder = None
        self.evaporation_node = None
        self.evaporation_recorder = None


    @classmethod
    def pre_load(cls, model, data):

        bathymetry = data.pop("bathymetry", None)
        name = data.pop("name")
        node = cls(name=name, model=model, **data)

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

        if volumes is not None and levels is not None:
            node.level = InterpolatedVolumeParameter(model, node, volumes, levels)

        if volumes is not None and areas is not None:
            node.area = InterpolatedVolumeParameter(model, node, volumes, areas)
        if node.weather is not None:
            node._make_weather_nodes(model, node.weather, node.weather_cost)
        setattr(node, "_Loadable__parameters_to_load", {})
        return node


    def _make_weather_nodes(self, model, weather, cost):

        if not isinstance(self.area, Parameter):
            raise ValueError('Weather nodes can only be created if an area Parameter is given.')

        if isinstance(weather, str):
            weather = load_parameter(model, weather).dataframe.astype(np.float64)
        else:
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
            storage_node = model.pre_load_node(storage_node)
            self.storage_node = storage_node
            if hasattr(storage_node, "level") and storage_node.level is not None:
                if not isinstance(storage_node.level, Parameter):
                    level_parameter = ConstantParameter(model, value=storage_node.level)
                else:
                    level_parameter = storage_node.level

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
                                                   **hp_kwargs)

        self.max_flow = hp_target_flow

        hp_recorder = HydropowerRecorder(model, self,
                                         name=f"__{name}__:hydropowerrecorder",
                                         water_elevation_parameter=level_parameter,
                                         turbine_elevation=turbine_elevation, **hp_kwargs)
        self.hydropower_recorder = hp_recorder

    @classmethod
    def pre_load(cls, model, data):
        name = data.pop("name")
        cost = data.pop("cost", 0.0)
        min_flow = data.pop("min_flow", None)

        node = cls(name=name, model=model, **data)

        cost = load_parameter(model, cost)
        min_flow = load_parameter(model, min_flow)
        if cost is None:
            cost = 0.0
        if min_flow is None:
            min_flow = 0.0

        node.cost = cost
        node.min_flow = min_flow
        setattr(node, "_Loadable__parameters_to_load", {})

        return node


class MonthlyOutput(Output, metaclass=NodeMeta):
    def __init__(self, model, name, scenario=None, **kwargs):
        super().__init__(model, name, **kwargs)
