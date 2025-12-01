from pywr import recorders
from pywr import parameters
from pywr.parameters import *
from pywr.recorders import *
import pandas
import pandas as pd
import numpy
import numpy as np
import scipy
import math
from pywr.nodes import *
from pywr.parameters.control_curves import *
from pywr.parameters._thresholds import *
from pywr.parameters._hydropower import *
from pywr.domains.river import *

class IrrigationWaterRequirementParameter(Parameter):
    """Simple irrigation water requirement model. """
    def __init__(self, model, rainfall_parameter, et_parameter, crop_water_factor_parameter, area, reference_et, yield_per_area, conveyance_efficiency, 
                 application_efficiency, factor=1e6, revenue_per_yield=1,  et_factor=0.001, area_factor=10000, **kwargs):

        super().__init__(model, **kwargs)

        self._area = None
        self.area = area
        self.factor = factor
        self.et_factor = et_factor
        self.area_factor = area_factor
        self._et_parameter = None
        self._rainfall_parameter = None
        self.reference_et = reference_et
        self.et_parameter = et_parameter
        self.yield_per_area = yield_per_area
        self._crop_water_factor_parameter = None
        self.revenue_per_yield = revenue_per_yield
        self.rainfall_parameter = rainfall_parameter
        self._conveyance_efficiency = None
        self.conveyance_efficiency = conveyance_efficiency
        self._application_efficiency = None
        self.application_efficiency = application_efficiency
        self.crop_water_factor_parameter = crop_water_factor_parameter

    et_parameter = parameter_property("_et_parameter")
    rainfall_parameter = parameter_property("_rainfall_parameter")
    crop_water_factor_parameter = parameter_property("_crop_water_factor_parameter")
    conveyance_efficiency = parameter_property("_conveyance_efficiency")
    application_efficiency = parameter_property("_application_efficiency")
    area = parameter_property("_area")

    def value(self, timestep, scenario_index):

        et = self.et_parameter.get_value(scenario_index) * self.et_factor
        effective_rainfall = self.rainfall_parameter.get_value(scenario_index) * self.et_factor
        crop_water_factor = self.crop_water_factor_parameter.get_value(scenario_index)
        conv_efficiency = self.conveyance_efficiency.get_value(scenario_index)
        app_efficiency = self.application_efficiency.get_value(scenario_index)
        area_ = self.area.get_value(scenario_index)
      
        # Calculate crop water requirement
        if effective_rainfall > crop_water_factor * et:
            # No crop water requirement if there is enough rainfall
            crop_water_requirement = 0.0
        else:
            # Irrigation required to meet shortfall in rainfall
            
            crop_water_requirement = (crop_water_factor * et - effective_rainfall) * (area_ * self.area_factor)

        # Calculate overall efficiency
        efficiency = app_efficiency * conv_efficiency

        # TODO error checking on division by zero
        irrigation_water_requirement = crop_water_requirement / efficiency
        
        return irrigation_water_requirement/self.factor #To have Mm3/day

    def crop_yield(self, curtailment_ratio):
        return self.area * self.yield_per_area * curtailment_ratio

    def crop_revenue(self, curtailment_ratio):
        return self.revenue_per_yield * self.crop_yield(curtailment_ratio)

    @classmethod
    def load(cls, model, data):

        rainfall_parameter = load_parameter(model, data.pop('rainfall_parameter'))
        et_parameter = load_parameter(model, data.pop('et_parameter'))
        cwf_parameter = load_parameter(model, data.pop('crop_water_factor_parameter'))

        attribute_list = ["conveyance_efficiency", "application_efficiency", "area", "reference_et", "yield_per_area"]
        attributes = {}

        for attribute in attribute_list:
            if attribute in data:
                if isinstance(data[attribute], (int, float)):
                    attributes[attribute] = data.pop(attribute)
                else:
                    attributes[attribute] = load_parameter(model, data.pop(attribute))

        return cls(model, rainfall_parameter, et_parameter, cwf_parameter, attributes["area"], attributes["reference_et"], 
                   attributes["yield_per_area"], attributes["conveyance_efficiency"], attributes["application_efficiency"], **data)


IrrigationWaterRequirementParameter.register()


class TransientDecisionParameter(Parameter):
    """ Return one of two values depending on the current time-step

    This `Parameter` can be used to model a discrete decision event
     that happens at a given date. Prior to this date the `before`
     value is returned, and post this date the `after` value is returned.

    Parameters
    ----------
    decision_date : string or pandas.Timestamp
        The trigger date for the decision.
    before_parameter : Parameter
        The value to use before the decision date.
    after_parameter : Parameter
        The value to use after the decision date.
    earliest_date : string or pandas.Timestamp or None
        Earliest date that the variable can be set to. Defaults to `model.timestepper.start`
    latest_date : string or pandas.Timestamp or None
        Latest date that the variable can be set to. Defaults to `model.timestepper.end`
    decision_freq : pandas frequency string (default 'AS')
        The resolution of feasible dates. For example 'AS' would create feasible dates every
        year between `earliest_date` and `latest_date`. The `pandas` functions are used
        internally for delta date calculations.

    """

    def __init__(self, model, decision_date, before_parameter, after_parameter,
                 earliest_date=None, latest_date=None, decision_freq='AS', **kwargs):
        super(TransientDecisionParameter, self).__init__(model, **kwargs)
        self._decision_date = None
        self.decision_date = decision_date

        if not isinstance(before_parameter, Parameter):
            raise ValueError('The `before` value should be a Parameter instance.')
        before_parameter.parents.add(self)
        self.before_parameter = before_parameter

        if not isinstance(after_parameter, Parameter):
            raise ValueError('The `after` value should be a Parameter instance.')
        after_parameter.parents.add(self)
        self.after_parameter = after_parameter

        # These parameters are mostly used if this class is used as variable.
        self._earliest_date = None
        self.earliest_date = earliest_date

        self._latest_date = None
        self.latest_date = latest_date

        self.decision_freq = decision_freq
        self._feasible_dates = None
        self.integer_size = 1  # This parameter has a single integer variable

    def decision_date():
        def fget(self):
            return self._decision_date
        
        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._decision_date = value
            else:
                self._decision_date = pd.to_datetime(value)

        return locals()

    decision_date = property(**decision_date())

    def earliest_date():
        def fget(self):
            if self._earliest_date is not None:
                return self._earliest_date
            else:
                return self.model.timestepper.start

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._earliest_date = value
            else:
                self._earliest_date = pd.to_datetime(value)

        return locals()

    earliest_date = property(**earliest_date())

    def latest_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pd.to_datetime(value)

        return locals()

    latest_date = property(**latest_date())

    def setup(self):
        super(TransientDecisionParameter, self).setup()

        # Now setup the feasible dates for when this object is used as a variable.
        self._feasible_dates = pd.date_range(self.earliest_date, self.latest_date,
                                                 freq=self.decision_freq)
        
    def value(self, ts, scenario_index):

        if ts is None:
            v = self.before_parameter.get_value(scenario_index)
        elif ts.datetime >= self.decision_date:
            v = self.after_parameter.get_value(scenario_index)
        else:
            v = self.before_parameter.get_value(scenario_index)
        return v

    def get_integer_lower_bounds(self):
        return np.array([0, ], dtype=np.int)

    def get_integer_upper_bounds(self):
        return np.array([len(self._feasible_dates) - 1, ], dtype=np.int)

    def set_integer_variables(self, values):
        # Update the decision date with the corresponding feasible date
        self.decision_date = self._feasible_dates[values[0]]

    def get_integer_variables(self):
        return np.array([self._feasible_dates.get_loc(self.decision_date), ], dtype=np.int)

    def dump(self):

        data = {
            'earliest_date': self.earliest_date.isoformat(),
            'latest_date': self.latest_date.isoformat(),
            'decision_date': self.decision_date.isoformat(),
            'decision_frequency': self.decision_freq
        }

        return data

    @classmethod
    def load(cls, model, data):

        before_parameter = load_parameter(model, data.pop('before_parameter'))
        after_parameter = load_parameter(model, data.pop('after_parameter'))

        return cls(model, before_parameter=before_parameter, after_parameter=after_parameter, **data)
        
        
TransientDecisionParameter.register()

class IrrigationWaterRequirementParameter(Parameter):
    """Simple irrigation water requirement model. """
    def __init__(self, model, rainfall_parameter, et_parameter, crop_water_factor_parameter, area, reference_et, yield_per_area, conveyance_efficiency, 
                 application_efficiency, factor=1e6, revenue_per_yield=1,  et_factor=0.001, area_factor=10000, **kwargs):

        super().__init__(model, **kwargs)

        self._area = None
        self.area = area
        self.factor = factor
        self.et_factor = et_factor
        self.area_factor = area_factor
        self._et_parameter = None
        self._rainfall_parameter = None
        self.reference_et = reference_et
        self.et_parameter = et_parameter
        self.yield_per_area = yield_per_area
        self._crop_water_factor_parameter = None
        self.revenue_per_yield = revenue_per_yield
        self.rainfall_parameter = rainfall_parameter
        self._conveyance_efficiency = None
        self.conveyance_efficiency = conveyance_efficiency
        self._application_efficiency = None
        self.application_efficiency = application_efficiency
        self.crop_water_factor_parameter = crop_water_factor_parameter

    et_parameter = parameter_property("_et_parameter")
    rainfall_parameter = parameter_property("_rainfall_parameter")
    crop_water_factor_parameter = parameter_property("_crop_water_factor_parameter")
    conveyance_efficiency = parameter_property("_conveyance_efficiency")
    application_efficiency = parameter_property("_application_efficiency")
    area = parameter_property("_area")

    def value(self, timestep, scenario_index):

        et = self.et_parameter.get_value(scenario_index) * self.et_factor
        effective_rainfall = self.rainfall_parameter.get_value(scenario_index) * self.et_factor
        crop_water_factor = self.crop_water_factor_parameter.get_value(scenario_index)
        conv_efficiency = self.conveyance_efficiency.get_value(scenario_index)
        app_efficiency = self.application_efficiency.get_value(scenario_index)
        area_ = self.area.get_value(scenario_index)
      
        # Calculate crop water requirement
        if effective_rainfall > crop_water_factor * et:
            # No crop water requirement if there is enough rainfall
            crop_water_requirement = 0.0
        else:
            # Irrigation required to meet shortfall in rainfall
            
            crop_water_requirement = (crop_water_factor * et - effective_rainfall) * (area_ * self.area_factor)

        # Calculate overall efficiency
        efficiency = app_efficiency * conv_efficiency

        # TODO error checking on division by zero
        irrigation_water_requirement = crop_water_requirement / efficiency
        
        return irrigation_water_requirement/self.factor #To have Mm3/day

    def crop_yield(self, curtailment_ratio):
        return self.area * self.yield_per_area * curtailment_ratio

    def crop_revenue(self, curtailment_ratio):
        return self.revenue_per_yield * self.crop_yield(curtailment_ratio)

    @classmethod
    def load(cls, model, data):

        rainfall_parameter = load_parameter(model, data.pop('rainfall_parameter'))
        et_parameter = load_parameter(model, data.pop('et_parameter'))
        cwf_parameter = load_parameter(model, data.pop('crop_water_factor_parameter'))

        attribute_list = ["conveyance_efficiency", "application_efficiency", "area", "reference_et", "yield_per_area"]
        attributes = {}

        for attribute in attribute_list:
            if attribute in data:
                if isinstance(data[attribute], (int, float)):
                    attributes[attribute] = data.pop(attribute)
                else:
                    attributes[attribute] = load_parameter(model, data.pop(attribute))

        return cls(model, rainfall_parameter, et_parameter, cwf_parameter, attributes["area"], attributes["reference_et"], 
                   attributes["yield_per_area"], attributes["conveyance_efficiency"], attributes["application_efficiency"], **data)


IrrigationWaterRequirementParameter.register()


class TransientDecisionParameter(Parameter):
    """ Return one of two values depending on the current time-step

    This `Parameter` can be used to model a discrete decision event
     that happens at a given date. Prior to this date the `before`
     value is returned, and post this date the `after` value is returned.

    Parameters
    ----------
    decision_date : string or pandas.Timestamp
        The trigger date for the decision.
    before_parameter : Parameter
        The value to use before the decision date.
    after_parameter : Parameter
        The value to use after the decision date.
    earliest_date : string or pandas.Timestamp or None
        Earliest date that the variable can be set to. Defaults to `model.timestepper.start`
    latest_date : string or pandas.Timestamp or None
        Latest date that the variable can be set to. Defaults to `model.timestepper.end`
    decision_freq : pandas frequency string (default 'AS')
        The resolution of feasible dates. For example 'AS' would create feasible dates every
        year between `earliest_date` and `latest_date`. The `pandas` functions are used
        internally for delta date calculations.

    """

    def __init__(self, model, decision_date, before_parameter, after_parameter,
                 earliest_date=None, latest_date=None, decision_freq='AS', **kwargs):
        super(TransientDecisionParameter, self).__init__(model, **kwargs)
        self._decision_date = None
        self.decision_date = decision_date

        if not isinstance(before_parameter, Parameter):
            raise ValueError('The `before` value should be a Parameter instance.')
        before_parameter.parents.add(self)
        self.before_parameter = before_parameter

        if not isinstance(after_parameter, Parameter):
            raise ValueError('The `after` value should be a Parameter instance.')
        after_parameter.parents.add(self)
        self.after_parameter = after_parameter

        # These parameters are mostly used if this class is used as variable.
        self._earliest_date = None
        self.earliest_date = earliest_date

        self._latest_date = None
        self.latest_date = latest_date

        self.decision_freq = decision_freq
        self._feasible_dates = None
        self.integer_size = 1  # This parameter has a single integer variable

    def decision_date():
        def fget(self):
            return self._decision_date
        
        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._decision_date = value
            else:
                self._decision_date = pd.to_datetime(value)

        return locals()

    decision_date = property(**decision_date())

    def earliest_date():
        def fget(self):
            if self._earliest_date is not None:
                return self._earliest_date
            else:
                return self.model.timestepper.start

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._earliest_date = value
            else:
                self._earliest_date = pd.to_datetime(value)

        return locals()

    earliest_date = property(**earliest_date())

    def latest_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pd.to_datetime(value)

        return locals()

    latest_date = property(**latest_date())

    def setup(self):
        super(TransientDecisionParameter, self).setup()

        # Now setup the feasible dates for when this object is used as a variable.
        self._feasible_dates = pd.date_range(self.earliest_date, self.latest_date,
                                                 freq=self.decision_freq)
        
    def value(self, ts, scenario_index):

        if ts is None:
            v = self.before_parameter.get_value(scenario_index)
        elif ts.datetime >= self.decision_date:
            v = self.after_parameter.get_value(scenario_index)
        else:
            v = self.before_parameter.get_value(scenario_index)
        return v

    def get_integer_lower_bounds(self):
        return np.array([0, ], dtype=np.int)

    def get_integer_upper_bounds(self):
        return np.array([len(self._feasible_dates) - 1, ], dtype=np.int)

    def set_integer_variables(self, values):
        # Update the decision date with the corresponding feasible date
        self.decision_date = self._feasible_dates[values[0]]

    def get_integer_variables(self):
        return np.array([self._feasible_dates.get_loc(self.decision_date), ], dtype=np.int)

    def dump(self):

        data = {
            'earliest_date': self.earliest_date.isoformat(),
            'latest_date': self.latest_date.isoformat(),
            'decision_date': self.decision_date.isoformat(),
            'decision_frequency': self.decision_freq
        }

        return data

    @classmethod
    def load(cls, model, data):

        before_parameter = load_parameter(model, data.pop('before_parameter'))
        after_parameter = load_parameter(model, data.pop('after_parameter'))

        return cls(model, before_parameter=before_parameter, after_parameter=after_parameter, **data)
        
        
TransientDecisionParameter.register()

class IrrigationWaterRequirementParameter(Parameter):
    """Simple irrigation water requirement model. """
    def __init__(self, model, rainfall_parameter, et_parameter, crop_water_factor_parameter, area, reference_et, yield_per_area, conveyance_efficiency, 
                 application_efficiency, factor=1e6, revenue_per_yield=1,  et_factor=0.001, area_factor=10000, **kwargs):

        super().__init__(model, **kwargs)

        self._area = None
        self.area = area
        self.factor = factor
        self.et_factor = et_factor
        self.area_factor = area_factor
        self._et_parameter = None
        self._rainfall_parameter = None
        self.reference_et = reference_et
        self.et_parameter = et_parameter
        self.yield_per_area = yield_per_area
        self._crop_water_factor_parameter = None
        self.revenue_per_yield = revenue_per_yield
        self.rainfall_parameter = rainfall_parameter
        self._conveyance_efficiency = None
        self.conveyance_efficiency = conveyance_efficiency
        self._application_efficiency = None
        self.application_efficiency = application_efficiency
        self.crop_water_factor_parameter = crop_water_factor_parameter

    et_parameter = parameter_property("_et_parameter")
    rainfall_parameter = parameter_property("_rainfall_parameter")
    crop_water_factor_parameter = parameter_property("_crop_water_factor_parameter")
    conveyance_efficiency = parameter_property("_conveyance_efficiency")
    application_efficiency = parameter_property("_application_efficiency")
    area = parameter_property("_area")

    def value(self, timestep, scenario_index):

        et = self.et_parameter.get_value(scenario_index) * self.et_factor
        effective_rainfall = self.rainfall_parameter.get_value(scenario_index) * self.et_factor
        crop_water_factor = self.crop_water_factor_parameter.get_value(scenario_index)
        conv_efficiency = self.conveyance_efficiency.get_value(scenario_index)
        app_efficiency = self.application_efficiency.get_value(scenario_index)
        area_ = self.area.get_value(scenario_index)
      
        # Calculate crop water requirement
        if effective_rainfall > crop_water_factor * et:
            # No crop water requirement if there is enough rainfall
            crop_water_requirement = 0.0
        else:
            # Irrigation required to meet shortfall in rainfall
            
            crop_water_requirement = (crop_water_factor * et - effective_rainfall) * (area_ * self.area_factor)

        # Calculate overall efficiency
        efficiency = app_efficiency * conv_efficiency

        # TODO error checking on division by zero
        irrigation_water_requirement = crop_water_requirement / efficiency
        
        return irrigation_water_requirement/self.factor #To have Mm3/day

    def crop_yield(self, curtailment_ratio):
        return self.area * self.yield_per_area * curtailment_ratio

    def crop_revenue(self, curtailment_ratio):
        return self.revenue_per_yield * self.crop_yield(curtailment_ratio)

    @classmethod
    def load(cls, model, data):

        rainfall_parameter = load_parameter(model, data.pop('rainfall_parameter'))
        et_parameter = load_parameter(model, data.pop('et_parameter'))
        cwf_parameter = load_parameter(model, data.pop('crop_water_factor_parameter'))

        attribute_list = ["conveyance_efficiency", "application_efficiency", "area", "reference_et", "yield_per_area"]
        attributes = {}

        for attribute in attribute_list:
            if attribute in data:
                if isinstance(data[attribute], (int, float)):
                    attributes[attribute] = data.pop(attribute)
                else:
                    attributes[attribute] = load_parameter(model, data.pop(attribute))

        return cls(model, rainfall_parameter, et_parameter, cwf_parameter, attributes["area"], attributes["reference_et"], 
                   attributes["yield_per_area"], attributes["conveyance_efficiency"], attributes["application_efficiency"], **data)


IrrigationWaterRequirementParameter.register()


class TransientDecisionParameter(Parameter):
    """ Return one of two values depending on the current time-step

    This `Parameter` can be used to model a discrete decision event
     that happens at a given date. Prior to this date the `before`
     value is returned, and post this date the `after` value is returned.

    Parameters
    ----------
    decision_date : string or pandas.Timestamp
        The trigger date for the decision.
    before_parameter : Parameter
        The value to use before the decision date.
    after_parameter : Parameter
        The value to use after the decision date.
    earliest_date : string or pandas.Timestamp or None
        Earliest date that the variable can be set to. Defaults to `model.timestepper.start`
    latest_date : string or pandas.Timestamp or None
        Latest date that the variable can be set to. Defaults to `model.timestepper.end`
    decision_freq : pandas frequency string (default 'AS')
        The resolution of feasible dates. For example 'AS' would create feasible dates every
        year between `earliest_date` and `latest_date`. The `pandas` functions are used
        internally for delta date calculations.

    """

    def __init__(self, model, decision_date, before_parameter, after_parameter,
                 earliest_date=None, latest_date=None, decision_freq='AS', **kwargs):
        super(TransientDecisionParameter, self).__init__(model, **kwargs)
        self._decision_date = None
        self.decision_date = decision_date

        if not isinstance(before_parameter, Parameter):
            raise ValueError('The `before` value should be a Parameter instance.')
        before_parameter.parents.add(self)
        self.before_parameter = before_parameter

        if not isinstance(after_parameter, Parameter):
            raise ValueError('The `after` value should be a Parameter instance.')
        after_parameter.parents.add(self)
        self.after_parameter = after_parameter

        # These parameters are mostly used if this class is used as variable.
        self._earliest_date = None
        self.earliest_date = earliest_date

        self._latest_date = None
        self.latest_date = latest_date

        self.decision_freq = decision_freq
        self._feasible_dates = None
        self.integer_size = 1  # This parameter has a single integer variable

    def decision_date():
        def fget(self):
            return self._decision_date
        
        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._decision_date = value
            else:
                self._decision_date = pd.to_datetime(value)

        return locals()

    decision_date = property(**decision_date())

    def earliest_date():
        def fget(self):
            if self._earliest_date is not None:
                return self._earliest_date
            else:
                return self.model.timestepper.start

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._earliest_date = value
            else:
                self._earliest_date = pd.to_datetime(value)

        return locals()

    earliest_date = property(**earliest_date())

    def latest_date():
        def fget(self):
            if self._latest_date is not None:
                return self._latest_date
            else:
                return self.model.timestepper.end

        def fset(self, value):
            if isinstance(value, pd.Timestamp):
                self._latest_date = value
            else:
                self._latest_date = pd.to_datetime(value)

        return locals()

    latest_date = property(**latest_date())

    def setup(self):
        super(TransientDecisionParameter, self).setup()

        # Now setup the feasible dates for when this object is used as a variable.
        self._feasible_dates = pd.date_range(self.earliest_date, self.latest_date,
                                                 freq=self.decision_freq)
        
    def value(self, ts, scenario_index):

        if ts is None:
            v = self.before_parameter.get_value(scenario_index)
        elif ts.datetime >= self.decision_date:
            v = self.after_parameter.get_value(scenario_index)
        else:
            v = self.before_parameter.get_value(scenario_index)
        return v

    def get_integer_lower_bounds(self):
        return np.array([0, ], dtype=np.int)

    def get_integer_upper_bounds(self):
        return np.array([len(self._feasible_dates) - 1, ], dtype=np.int)

    def set_integer_variables(self, values):
        # Update the decision date with the corresponding feasible date
        self.decision_date = self._feasible_dates[values[0]]

    def get_integer_variables(self):
        return np.array([self._feasible_dates.get_loc(self.decision_date), ], dtype=np.int)

    def dump(self):

        data = {
            'earliest_date': self.earliest_date.isoformat(),
            'latest_date': self.latest_date.isoformat(),
            'decision_date': self.decision_date.isoformat(),
            'decision_frequency': self.decision_freq
        }

        return data

    @classmethod
    def load(cls, model, data):

        before_parameter = load_parameter(model, data.pop('before_parameter'))
        after_parameter = load_parameter(model, data.pop('after_parameter'))

        return cls(model, before_parameter=before_parameter, after_parameter=after_parameter, **data)
        
        
TransientDecisionParameter.register()

class ReservoirMonthlyReliabilityRecorder(NumpyArrayAbstractStorageRecorder):

    """
    1 - (Total months below minimum storage level / total months in simulation)
    """

    def __init__(self, model, node, threshold, **kwargs):
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_volume = node.get_max_volume(scenario_index)

            if node.volume[scenario_index.global_id] < max_volume * self.threshold:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        DataFrame = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('M').max()

        return 1 - ((DataFrame.sum().round(0) / DataFrame.shape[0]))
    
    def to_dataframe(self):

        raise NotImplementedError()


ReservoirMonthlyReliabilityRecorder.register()


class ReservoirAnnualReliabilityRecorder(NumpyArrayAbstractStorageRecorder):

    """
    1 - (Total years below minimum storage level / total years in simulation)
    """

    def __init__(self, model, node, threshold, **kwargs):
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_volume = node.get_max_volume(scenario_index)

            if node.volume[scenario_index.global_id] < max_volume * self.threshold:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        DataFrame = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('Y').max()

        return 1 - ((DataFrame.sum().round(0) / DataFrame.shape[0]))
    
    def to_dataframe(self):
        
        raise NotImplementedError()


ReservoirAnnualReliabilityRecorder.register()


class SupplyReliabilityRecorder(NodeRecorder):

    """
    add description
    """

    def __init__(self, model, node, **kwargs):
        super().__init__(model, node, **kwargs)

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)

            if max_flow == 0:
                deficit = 0

            else:
                deficit  = (max_flow - node.flow[scenario_index.global_id]) / max_flow

            if deficit > 0.01:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        DataFrame = pd.DataFrame(np.array(self._data), index=index, columns=sc_index).resample('M').max().loc[:str(last_year), :]

        return 1 - ((DataFrame.sum().round(0) / DataFrame.shape[0]))
    
    def to_dataframe(self):
        
        raise NotImplementedError()


SupplyReliabilityRecorder.register()


class AnnualDeficitRecorder(NodeRecorder):

    """
    Annual deficit recorder (%)
    """

    def __init__(self, model, node, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        
        super().__init__(model, node, **kwargs)
        self.temporal_aggregator = temporal_agg_func

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        rlts = 1 - supply.divide(demand)

        if self.temporal_aggregator == 'mean':
            to_save = rlts.mean()

        if self.temporal_aggregator == 'max':
            to_save = rlts.max()

        if self.temporal_aggregator == 'min':
            to_save = rlts.min()

        return to_save
    
    def to_dataframe(self):
        
        raise NotImplementedError()


AnnualDeficitRecorder.register()


class ReservoirResilienceRecorder(NumpyArrayAbstractStorageRecorder):

    """
    add description
    """

    def __init__(self, model, node, threshold, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        
        super().__init__(model, node, **kwargs)
        self.temporal_aggregator = temporal_agg_func
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._data = np.zeros((nts, ncomb))

    def reset(self):
        self._data[:, :] = 0.0
                
    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:
            max_volume = node.get_max_volume(scenario_index)

            if node.volume[scenario_index.global_id] < max_volume * self.threshold:
                self._data[ts.index,scenario_index.global_id] = 1

            else:
                self._data[ts.index,scenario_index.global_id] = 0

        return 0

    def values(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        tem_dams = pd.DataFrame(np.array(self._data), index=index, columns=sc_index)

        tem_dams_diff = tem_dams.diff().ne(0).cumsum()
        
        tem_dams_occurrence = tem_dams.multiply(tem_dams_diff)

        resilience = {}
        
        levels = [x for x, _ in enumerate(tem_dams_occurrence.columns.names)]
        
        for idx, dataframe in tem_dams_occurrence.groupby(level=levels, axis=1):
            
            tem = dataframe.T.reset_index(drop=True).T
            
            tem.columns = ['col']
            
            tem_res = tem[tem['col'] != 0].groupby(['col'])['col'].count()

            if self.temporal_aggregator == 'mean':
                resilience[idx] = tem_res.mean()
                
            if self.temporal_aggregator == 'max':
                resilience[idx] = tem_res.max()
            
        
        rlts = pd.DataFrame.from_dict(resilience, orient='index', columns=[""])

        rlts = rlts.T

        rlts.columns = pd.MultiIndex.from_tuples(rlts.columns, names=sc_index.names)

        return rlts.T
    
    def to_dataframe(self):
        
        raise NotImplementedError()


ReservoirResilienceRecorder.register()


class RelativeCropYieldRecorder(Recorder):
    """Relative crop yield recorder.

    This recorder computes the relative crop yield based on a curtailment ratio between a node's
    actual flow and it's `max_flow` expected flow. It is assumed the `max_flow` parameter is an
    `AggregatedParameter` containing only `IrrigationWaterRequirementParameter` parameters.

    """
    def __init__(self, model, nodes, **kwargs):
        temporal_agg_func = kwargs.pop('temporal_agg_func', 'mean')
        super().__init__(model, **kwargs)

        for node in nodes:
            max_flow_param = node.max_flow
            self.children.add(max_flow_param)

        self.nodes = nodes
        self._temporal_aggregator = Aggregator(temporal_agg_func)
        self.data = None

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)
        self.data = np.zeros((nts, ncomb))

    def reset(self):
        self.data[:, :] = 0.0

    def after(self):

        norm_crop_revenue = None
        full_norm_crop_revenue = None
        ts = self.model.timestepper.current
        self.data[ts.index, :] = 0
        norm_yield = 0
        full_norm_yield = 0

        for node in self.nodes:
            crop_aggregated_parameter = node.max_flow
            actual = node.flow
            requirement = np.array(crop_aggregated_parameter.get_all_values())
            # Divide non-zero elements
            curtailment_ratio = np.divide(actual, requirement, out=np.zeros_like(actual), where=requirement != 0)
            no_curtailment = np.ones_like(curtailment_ratio)

            if norm_crop_revenue is None:
                norm_crop_revenue = crop_aggregated_parameter.parameters[0].crop_revenue(curtailment_ratio)
                full_norm_crop_revenue = crop_aggregated_parameter.parameters[0].crop_revenue(no_curtailment)

            for parameter in crop_aggregated_parameter.parameters:
                crop_revenue = parameter.crop_revenue(curtailment_ratio)
                full_crop_revenue = parameter.crop_revenue(no_curtailment)
                crop_yield = parameter.crop_yield(curtailment_ratio)
                full_crop_yield = parameter.crop_yield(no_curtailment)
                # Increment effective yield, scaled by the first crop's revenue
                norm_yield += crop_yield * np.divide(crop_revenue, norm_crop_revenue,
                                                    out=np.zeros_like(crop_revenue),
                                                    where=norm_crop_revenue != 0)

                full_norm_yield += full_crop_yield * np.divide(full_crop_revenue, full_norm_crop_revenue,
                                                              out=np.ones_like(full_crop_revenue),
                                                              where=full_norm_crop_revenue != 0)
                
                if requirement<0.00001:
                    self.data[ts.index, :] = 99999
                else:
                    self.data[ts.index, :] = norm_yield / full_norm_yield

    def values(self):
        """Compute a value for each scenario using `temporal_agg_func`.
        """
        return self._temporal_aggregator.aggregate_2d(self.data, axis=0, ignore_nan=self.ignore_nan)

    def to_dataframe(self):
        """ Return a `pandas.DataFrame` of the recorder data

        This DataFrame contains a MultiIndex for the columns with the recorder name
        as the first level and scenario combination names as the second level. This
        allows for easy combination with multiple recorder's DataFrames
        """
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        return pd.DataFrame(data=np.array(self.data), index=index, columns=sc_index)

    @classmethod
    def load(cls, model, data):
        nodes = [model._get_node_from_ref(model, n) for n in data.pop('nodes')]
        return cls(model, nodes, **data)

RelativeCropYieldRecorder.register()


class AverageAnnualCropYieldScenarioRecorder(NodeRecorder):

    """
    This recorder computes the average annual crop yield for each scenario based on the curtailment ratio between a node's
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._area = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        max_flow_param = self.node.max_flow

        areas = []
        for scenario_index in self.model.scenarios.combinations:
            areas.append(max_flow_param.area.get_value(scenario_index))
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        areas = pd.Series(np.array(areas), index=sc_index)
        # units for yields are in kg/ha
        # units for areas are in ha
        # units for crop_yield are in kg
        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(max_flow_param.yield_per_area, axis=0)


        return crop_yield.mean(axis=0)
    

AverageAnnualCropYieldScenarioRecorder.register()


class TotalAnnualCropYieldScenarioRecorder(NodeRecorder):

    """
    This recorder computes the Total annual crop yield for each scenario assuming there is anough water to irrigate the crop
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        max_flow_param = self.node.max_flow

        areas = []

        for scenario_index in self.model.scenarios.combinations:
            areas.append(max_flow_param.area.get_value(scenario_index))
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        #supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        curtailment_ratio = demand.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        areas = pd.Series(np.array(areas), index=sc_index)

        crop_yield = curtailment_ratio.multiply(areas, axis=1).multiply(max_flow_param.yield_per_area, axis=0)

        return crop_yield.mean(axis=0)


TotalAnnualCropYieldScenarioRecorder.register()


class IrrigationSupplyReliabilityScenarioRecorder(NodeRecorder):

    """
    This recorder calculates the supply reliability of an irrigation node considering only the months with higher demand 
    based on the Kc parameter. 
    
    A month with high demand > 0.8 Kc
    A year is considered that fails if the supply is less than 80% of the demand in any of the months with higher demand
    The supply reliability is calculated as (1 - ((number of years of failure / number of years in the simulation)))
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))
        self._kc = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0
        self._kc[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node
        max_flow_param = self.node.max_flow

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)
            self._kc[ts.index, scenario_index.global_id] = max_flow_param.crop_water_factor_parameter.get_value(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('M').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('M').sum().loc[:str(last_year), :]

        # Here we calculate the months where the demand is higher than 0.8 Kc
        mths_kc = np.where(self._kc < np.max(self._kc)*0.8, 0, 1)
        mths_kc = pd.DataFrame(mths_kc, index=index, columns=sc_index).resample('M').mean().loc[:str(last_year), :]

        moths_failures = np.where(supply < demand*self.threshold, 1, 0)
        moths_failures = pd.DataFrame(moths_failures, index=demand.index, columns=demand.columns)

        # Here we calculate the years where there is a failure only considering the months with high demand "mths_kc"
        failures = moths_failures.multiply(mths_kc).dropna().resample('Y').max()


        return 1 - (failures.sum().round(0) / failures.shape[0])


IrrigationSupplyReliabilityScenarioRecorder.register()


class CropCurtailmentRatioScenarioRecorder(NodeRecorder):

    """
    This recorder save the Annual Curtailment Ratios
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def to_dataframe(self):

        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]

        curtailment_ratio = supply.divide(demand)
        curtailment_ratio.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0

        
        return curtailment_ratio


    def values(self):
        

        return NotImplementedError()


CropCurtailmentRatioScenarioRecorder.register()


class AnnualIrrigationSupplyReliabilityScenarioRecorder(NodeRecorder):

    """
    This recorder calculates the annual supply reliability based on a threashold. 

    A year is considered that fails if the supply is less than a threashold
    The supply reliability is calculated as (1 - ((number of years of failure / number of years in the simulation)))
    """

    def __init__(self, model, node, threshold=None, **kwargs):
        
        super().__init__(model, node, **kwargs)
        self.threshold = threshold

    def setup(self):
        ncomb = len(self.model.scenarios.combinations)
        nts = len(self.model.timestepper)

        self._supply = np.zeros((nts, ncomb))
        self._demand = np.zeros((nts, ncomb))

    def reset(self):
        self._supply[:, :] = 0.0
        self._demand[:, :] = 0.0

    def after(self):
        ts = self.model.timestepper.current
        node = self.node

        for scenario_index in self.model.scenarios.combinations:

            self._supply[ts.index, scenario_index.global_id] = node.flow[scenario_index.global_id]
            self._demand[ts.index, scenario_index.global_id] = node.get_max_flow(scenario_index)

        return 0

    def to_dataframe(self):
        
        raise NotImplementedError()


    def values(self):
        
        index = self.model.timestepper.datetime_index
        sc_index = self.model.scenarios.multiindex

        last_year = index[-1].year

        supply = pd.DataFrame(np.array(self._supply), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]
        demand = pd.DataFrame(np.array(self._demand), index=index, columns=sc_index).resample('Y').sum().loc[:str(last_year), :]


        # Here we calculate the years where there is a failure only considering the threshold
        failures = np.where(supply < demand*self.threshold, 1, 0)
        failures = pd.DataFrame(failures, index=demand.index, columns=demand.columns)

        return 1 - (failures.sum().round(0) / failures.shape[0])


AnnualIrrigationSupplyReliabilityScenarioRecorder.register()

