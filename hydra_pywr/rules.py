from pywr.parameters import parameter_registry, Parameter
import pandas

from pywr.nodes import Link, Storage, Output, Input, AggregatedNode
from pywr.parameters.control_curves import ControlCurveInterpolatedParameter
from pywr.parameters._thresholds import ParameterThresholdParameter
from pywr.parameters._hydropower import HydropowerTargetParameter
from pywr.schema import NodeSchema, fields
from pywr.domains.river import Catchment
import numpy as np
import marshmallow
from . import DataFrameField
from pywr import recorders
from pywr import parameters

"""
    This file is the environment in which all the node types, parameters and recorders
    defined within Hydra are run. THese are known as hydra 'rules'

    The idea is that the code being executed only has
    access to a subset of libraries, and cannot use 'os' or 'sys', for example.
    
    The imports at the top define which modules are available to the hydra rules.
"""

def exec_rules(rules):
    for rule in rules:

        if rule.find('import ') >= 0:
            raise PermissionError("Calling 'import' is not permitted. Please contact the hydra-pywr maintainer to request additional libraries be made available.")

        exec(rule)
