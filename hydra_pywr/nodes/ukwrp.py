from pywr.nodes import Link, Storage, Output, Input, AggregatedNode
from pywr.schema import NodeSchema, fields
from pywr.parameters import AggregatedParameter, MaxParameter, MinParameter, DeficitParameter
from pywr.recorders import NumpyArrayParameterRecorder
from . import DataFrameField
from ..parameters import YearlyDataFrameParameter


class WaterResourceZonePR19(Link):
    """ A general node to represent a WRZ from the PR19 WRP tables.

    """
    class Schema(NodeSchema):
        # The main attributes are not validated (i.e. `Raw`)
        # They could be many different things.
        wafu_own = DataFrameField()
        distribution_input = DataFrameField()
        target_headroom_climate_change = DataFrameField()
        target_headroom_other = DataFrameField()

    def __init__(self, model, name, *args, **kwargs):
        wafu_own = YearlyDataFrameParameter(model, kwargs.pop('wafu_own'))
        distribution_input = YearlyDataFrameParameter(model, kwargs.pop('distribution_input'))
        target_headroom_climate_change = YearlyDataFrameParameter(model, kwargs.pop('target_headroom_climate_change'))
        target_headroom_other = YearlyDataFrameParameter(model, kwargs.pop('target_headroom_other'))

        super().__init__(model, name, *args, **kwargs)

        self.supply_node = Input(model, name=f'{name}-supply', parent=self)
        self.demand_node = Output(model, name=f'{name}-demand', parent=self)

        self.supply_node.max_flow = MaxParameter(model, wafu_own)

        demand = AggregatedParameter(model, [distribution_input,
                                             target_headroom_climate_change,
                                             target_headroom_other,
                                             MinParameter(model, wafu_own)],
                                     agg_func='sum',)

        self.demand_node.max_flow = demand
        self.demand_node.cost = -100

        self.supply_node.connect(self)
        self.connect(self.demand_node)

        # Track deficits
        deficit_param = DeficitParameter(model, self.demand_node, name=f'{name}-deficit_param')
        NumpyArrayParameterRecorder(model, deficit_param, name=f'__{name}__:deficit', temporal_agg_func='sum')







