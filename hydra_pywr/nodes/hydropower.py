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
    HydropowerTargetParameter
)

from pywr.recorders import (
    HydropowerRecorder
)

from pywr.parameters.control_curves import (
    ControlCurveInterpolatedParameter
)


class ProportionalInput(Input, metaclass=NodeMeta):
    min_proportion = 1e-6

    def __init__(self, model, name, node, proportion, **kwargs):
        super().__init__(model, name, **kwargs)

        self.node = node
        # Create the flow factors for the other node and self
        if proportion < self.__class__.min_proportion:
            self.max_flow = 0.0
        else:
            factors = [1, proportion]
            # Create the aggregated node to apply the factors.
            #self.aggregated_node = AggregatedNode(model, f'{name}.aggregated', [node, self], factors=factors)
            self.aggregated_node = AggregatedNode(model, f'{name}.aggregated', [node, self])
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


class Turbine(Link, metaclass=NodeMeta):
    def __init__(self, model, name, **kwargs):
        hp_recorder_kwarg_names = ('efficiency', 'density', 'flow_unit_conversion', 'energy_unit_conversion')
        hp_kwargs = {}
        for kwd in hp_recorder_kwarg_names:
            try:
                hp_kwargs[kwd] = kwargs.pop(kwd)
            except KeyError:
                pass

        """  Some error here: storage_node is str...
        self.storage_node = storage_node = kwargs.pop('storage_node', None)
        # See if there is a level parameter on the storage node
        if storage_node is not None:
            breakpoint()
            level_parameter = storage_node.level
            if not isinstance(level_parameter, Parameter):
                level_parameter = ConstantParameter(model, value=level_parameter)
        else:
            level_parameter = None
        """
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
