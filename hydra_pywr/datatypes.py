from hydra_base.lib.HydraTypes.Types import Descriptor


PARAMETER_HYDRA_TYPE_MAP = {}
RECORDER_HYDRA_TYPE_MAP = {}

TYPE_COMPONENT_MAP = {
    "parameter": PARAMETER_HYDRA_TYPE_MAP,
    "recorder": RECORDER_HYDRA_TYPE_MAP
}

hydra_pywr_data_types = [
    {
      "tag": "data_PYWR_DATA_TYPE",
      "name": "Hydra Pywr Data Type Base"
    },
    {
      "tag": "PYWR_RECORDER",
      "name": "Pywr Recorder"
    },
    {
      "tag": "PYWR_PARAMETER",
      "name": "Pywr Parameter"
    },
    {
      "tag": "PYWR_TABLE",
      "name": "Pywr Table"
    },
    {
      "tag": "PYWR_TIMESTEPPER",
      "name": "Pywr Timestepper"
    },
    {
      "tag": "PYWR_METADATA",
      "name": "Pywr Metadata"
    },
    {
      "tag": "PYWR_SCENARIOS",
      "name": "Pywr Scenarios"
    },
    {
      "tag": "PYWR_SCENARIO_COMBINATIONS",
      "name": "Pywr Scenario Combinations"
    },
    {
      "tag": "PYWR_PY_MODULE",
      "name": "Pywr Python Module"
    },
    {
      "tag": "PYWR_NODE_OUTPUT",
      "name": "Pywr Node Output"
    },
    {
      "tag": "PYWR_PARAMETER_PATTERN",
      "name": "Pywr Parameter Pattern"
    },
    {
      "tag": "PYWR_PARAMETER_PATTERN_REFERENCE",
      "name": "Pywr Parameter Pattern REFERENCE"
    },
    {
      "tag": "PYWR_DATAFRAME",
      "name": "Pywr Dataframe Parameter",
      "component": "parameter",
      "type": "dataframeparameter"
    },
    {
      "tag": "PYWR_PARAMETER_MONTHLY_PROFILE",
      "name": "Pywr Monthly Profile Parameter",
      "component": "parameter",
      "type": "monthlyprofileparameter"
    },
    {
      "tag": "PYWR_PARAMETER_CONTROL_CURVE_INDEX",
      "name": "Pywr Control Curve Index Parameter",
      "component": "parameter",
      "type": "controlcurveindexparameter"
    },
    {
      "tag": "PYWR_PARAMETER_INDEXED_ARRAY",
      "name": "Pywr Indexed Array Parameter",
      "component": "parameter",
      "type": "indexedarrayparameter"
    },
    {
      "tag": "PYWR_PARAMETER_CONTROL_CURVE_INTERPOLATED",
      "name": "Pywr Interpolated Control Curve Parameter",
      "component": "parameter",
      "type": "controlcurveinterpolatedparameter"
    },
    {
      "tag": "PYWR_PARAMETER_AGGREGATED",
      "name": "Pywr Aggregated Parameter",
      "component": "parameter",
      "type": "aggregatedparameter"
    },
    {
      "tag": "PYWR_PARAMETER_CONSTANT_SCENARIO",
      "name": "Pywr Constant Scenario Parameter",
      "component": "parameter",
      "type": "constantscenarioparameter"
    },
    {
      "tag": "PYWR_RECORDER_FDC",
      "name": "Pywr Flow Duration Curve Recorder",
      "component": "recorder",
      "type": "flowdurationcurverecorder"
    },
    {
      "tag": "PYWR_RECORDER_SDC",
      "name": "Pywr Storage Duration Curve Recorder",
      "component": "recorder",
      "type": "storagedurationcurverecorder"
    },
    {
      "tag": "PYWR_RECORDER_FDC_DEVIATION",
      "name": "Pywr Flow Duration Curve Deviation Recorder",
      "component": "recorder",
      "type": "flowdurationcurvedeviationrecorder"
    }
]

def generate_data_types(definitions):
    for t in definitions:
        def_name = t.get("name")
        if not def_name or not isinstance(def_name, str):
            raise ValueError(f"Type definition omits <name>: {t}")
        # NB. 'name' need not be a valid Python class name; the type
        # needs to exist only to trigger the base's __init_subclass__,
        # which will also handle a missing 'tag'
        type_name = def_name.replace(" ", "")
        T = type(f"Hydra{type_name}", (Descriptor,), t)
        if comp := t.get("component", None):
            comp_type_map = TYPE_COMPONENT_MAP[comp]
            comp_type = t.get("type")
            if not comp_type or not isinstance(comp_type, str):
                raise ValueError(f"Type specifies component but not type: {t}")
            comp_type_map[comp_type] = t["tag"]


generate_data_types(hydra_pywr_data_types)
