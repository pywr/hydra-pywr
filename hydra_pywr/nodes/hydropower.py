import pandas as pd
import numpy as np

from pywr.nodes import Link, Input, AggregatedNode, NodeMeta
from pywr.parameters import ScenarioWrapperParameter
from pywr.parameters.control_curves import ControlCurveInterpolatedParameter


class ProportionalInput(Input, metaclass=NodeMeta):
    """
    class Schema(NodeSchema):
        node = fields.NodeField()
        proportion = marshmallow.fields.Number()
    """
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
    """
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        min_flow = fields.ParameterReferenceField(allow_none=True)
        cost = fields.ParameterReferenceField(allow_none=True)
        storage_node = fields.NodeField()
        release_values = DataFrameField()
        scenario = fields.ScenarioReferenceField(allow_none=True)
    """

    def __init__(self, model, name, storage_node, release_values, scenario=None, **kwargs):

        #{'volume': {'1': 1.0, '2': 0.850482315, '3': 0.310088424, '4': 0.11, '5': 0.0}, 'value': {'1': 300, '2': 400, '3': 200, '4': 0, '5': 0}}
        print(f"type(release_values): {type(release_values)}")
        print(release_values)
        #pd.DataFrame.from_dict(rv)["volume"].iloc[1:-1].astype(np.float64)

        release_values = pd.DataFrame.from_dict(release_values)

        """ storage_node: retrieve node from model with return model._get_node_from_ref(model, value) """

        storage_node = model._get_node_from_ref(model, storage_node)

        if scenario is None:
            # Only one control curve should be defined. Get it explicitly
            #cc_values = [*release_values["volume"].values()][1:-1]
            #control_curves = [np.float64(v) for v in cc_values]
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
