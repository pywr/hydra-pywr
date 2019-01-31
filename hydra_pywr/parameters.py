from pywr.parameters import Parameter, DataFrameParameter
import numpy as np
import pandas


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
MonthlyArrayIndexedParameter.register()


class YearlyDataFrameParameter(Parameter):
    def __init__(self, model, dataframe, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.dataframe = dataframe

    def value(self, ts, si):
        return self.dataframe.loc[str(ts.year)].iloc[0]

    @classmethod
    def load(cls, model, data):
        df = pandas.DataFrame.from_dict(data.pop('data'), orient='index')
        df.index = pandas.to_datetime(df.index)
        df.index.freq = pandas.infer_freq(df.index)
        return cls(model, df, **data)
YearlyDataFrameParameter.register()


class EmbeddedDataframeParameter(DataFrameParameter):
    @classmethod
    def load(cls, model, data):
        scenario = data.pop('scenario', None)
        if scenario is not None:
            scenario = model.scenarios[scenario]
        df = pandas.DataFrame.from_dict(data.pop('data'), orient='index')
        df.index = pandas.to_datetime(df.index)
        df.index.freq = pandas.infer_freq(df.index)
        return cls(model, df, scenario=scenario, **data)
EmbeddedDataframeParameter.register()
