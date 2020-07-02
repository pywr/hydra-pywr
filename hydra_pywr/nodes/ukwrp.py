from pywr.nodes import Link, Storage, Output, Input, AggregatedNode
from pywr.schema import NodeSchema, fields
from pywr.parameters import AggregatedParameter, MaxParameter, MinParameter, DeficitParameter, ConstantParameter
from pywr.recorders import NumpyArrayParameterRecorder
from . import DataFrameField
from ..parameters import YearlyDataFrameParameter


class WaterResourceZonePR19(Link):
   """ A general node to represent a WRZ from the PR19 WRP tables.

   """
   class Schema(NodeSchema):
       # The main attributes are not validated (i.e. `Raw`)
       # They could be many different things.
       demand = fields.ParameterField()
       supply = fields.ParameterField()
       headroom = fields.ParameterField()

   def __init__(self, model, name, *args, **kwargs):
       demand = kwargs.pop('demand', ConstantParameter(model, 0))
       supply = kwargs.pop('supply', ConstantParameter(model, 0))
       headroom = kwargs.pop('headroom', ConstantParameter(model, 0))
       super().__init__(model, name, *args, **kwargs)

       self.supply_node = Input(model, name=f'{name}-supply', parent=self)
       self.demand_node = Output(model, name=f'{name}-demand', parent=self)

       self.supply_node.max_flow = MaxParameter(model, supply)

       demand_headroom = AggregatedParameter(model, parameters=[demand, headroom], agg_func='sum')

       self.demand_node.max_flow = MaxParameter(model, demand_headroom)
       self.demand_node.cost = -100

       self.supply_node.connect(self)
       self.connect(self.demand_node)

       # Track deficits
       deficit_param = DeficitParameter(model, self.demand_node, name=f'{name}-deficit_param')
       NumpyArrayParameterRecorder(model, deficit_param, name=f'__{name}-demand__:deficit', temporal_agg_func='sum')
