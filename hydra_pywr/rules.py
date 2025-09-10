"""
    Module to execute hydra rules (parameter, recorder and node subclasses) defined
    in python in Hydra.
    This is housed in a separate module to control the context (imports) which
    the executed code can use. This avoids importing lots of unused imports in
    the main exporter file
"""
import logging

import pandas
import numpy as np
import scipy

try:
    from pywr.parameters import *
    from pywr.recorders import *

    from pywr.nodes import *
    from pywr.parameters.control_curves import *
    from pywr.parameters._thresholds import *
    from pywr.parameters._hydropower import *
    from pywr.domains.river import *

    #In case use wants to namespace stuff by parameters/recorders, the recommended way.
    from pywr import recorders
    from pywr import parameters

except ImportError:
    print("Unable to find Pywr!!!")

import hydra_pywr

LOG = logging.getLogger('hydra_pywr')

"""
    This file is the environment in which all the node types, parameters and recorders
    defined within Hydra are run. THese are known as hydra 'rules'

    The idea is that the code being executed only has
    access to a subset of libraries, and cannot use 'os' or 'sys', for example.

    The imports at the top define which modules are available to the hydra rules.
"""


def exec_rules(rules):
    for rule in rules:
        LOG.info("Executing rule %s", rule.name)
        #allow importing hydra_pywr but nothing else
        if rule.value.find('import ') >= 0:
            raise PermissionError("Calling 'import' is not permitted. Please contact the hydra-pywr maintainer to request additional libraries be made available.")

        try:
            exec(rule.value)
        except Exception as e:
            LOG.exception(e)
            LOG.critical("Unable to execute rule %s. Error was: %s", rule.name, e)

    #Now find any classes that have been added, and add them to the module's
    #dict so they can be accessed by pywr
    for k, v in locals().items():
        if isinstance(v, type):
            hydra_pywr.rules.__dict__[k] = v
