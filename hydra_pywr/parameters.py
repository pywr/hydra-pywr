from pywr.parameters import Parameter
import numpy as np


class MonthlyArrayIndexedParameter(Parameter):
    def __init__(self, model, values, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.values = np.array(values)

    def value(self, ts, scenario_index):

        start = self.model.timestepper.start
        start_year = start.year
        start_month = start.month

        current_year = ts.year
        current_month = ts.month

        index = (current_year - start_year)*12 + current_month - start_month
        return self.values[index]
