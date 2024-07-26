import pandas as pd
import numpy as np

from pywr.nodes import (
    NodeMeta,
    Link,
    Input,
    AggregatedNode,
    Output
)

from pywr.domains.river import Reservoir as PywrReservoir

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
            release_values = load_parameter(model, release_values)
        else:
            release_values = pd.DataFrame.from_dict(release_values)

        storage_node = model.pre_load_node(storage_node)


        if isinstance(release_values, ControlCurveInterpolatedParameter):
            max_flow_param = release_values

        elif scenario is None:
            control_curves = release_values['volume'].astype(float).values[1:-1]
            values =  release_values['value'].astype(float).values

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

class Reservoir(PywrReservoir, metaclass=NodeMeta):

    def __init__(self, model, **kwargs):

        self.bathymetry = kwargs.pop('bathymetry', None)
        #same as unit_conversion
        const = kwargs.pop('const', 1e6 * 1e-3 * 1e-6)

        # Pywr Storage does not expect a 'weather' kwargs, so move this to instance
        self.weather = kwargs.pop("weather", None)

        super().__init__(model, **kwargs)

        self.unit_conversion = ConstantParameter(model, const)

    def finalise_load(self):

        super(Reservoir, self).finalise_load()

        if self.bathymetry is not None:
            volumes = None
            levels = None
            if isinstance(self.bathymetry, str):
                bathymetry = load_parameter(self.model, self.bathymetry)
                volumes = bathymetry.dataframe['volume'].astype(np.float64)
                levels = bathymetry.dataframe['level'].astype(np.float64)
                areas = bathymetry.dataframe['area'].astype(np.float64)
            else:
                bathymetry = pd.DataFrame.from_dict(self.bathymetry)
                volumes = bathymetry['volume'].astype(np.float64)
                levels = bathymetry['level'].astype(np.float64)
                areas = bathymetry['area'].astype(np.float64)

            if volumes is not None and levels is not None:
                self.level = InterpolatedVolumeParameter(self.model, self, volumes, levels)

            if volumes is not None and areas is not None:
                self.area = InterpolatedVolumeParameter(self.model, self, volumes, areas)

        if self.weather is not None:
            self._make_weather_nodes(self.weather, self.evaporation_cost)

    def _make_weather_nodes(self, weather, cost):

        if not isinstance(self.area, Parameter):
            raise ValueError('Weather nodes can only be created if an area Parameter is given.')

        if isinstance(weather, str):
            weather = load_parameter(self.model, weather).dataframe.astype(np.float64)
        else:
            weather = pd.DataFrame.from_dict(weather)

        weather.index = weather.index.astype(int)
        weather = weather.sort_index()
        rainfall = MonthlyProfileParameter(self.model, weather['rainfall'].astype(np.float64))
        evaporation = MonthlyProfileParameter(self.model, weather['evaporation'].astype(np.float64))

        self._make_evaporation_node(evaporation, cost)
        self._make_rainfall_node(rainfall)

class MonthlyOutput(Output, metaclass=NodeMeta):
    def __init__(self, model, name, scenario=None, **kwargs):
        super().__init__(model, name, **kwargs)
