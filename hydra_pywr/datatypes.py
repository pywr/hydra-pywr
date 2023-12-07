import json
import os
from hydra_base.lib.HydraTypes.Types import Descriptor
from hydra_base.lib.HydraTypes.Registry import typemap


REQUIRED_TYPES = {
    "PYWR_PARAMETER",
    "PYWR_RECORDER",
    "PYWR_METADATA",
    "PYWR_TIMESTEPPER"
}

PARAMETER_HYDRA_TYPE_MAP = {}
RECORDER_HYDRA_TYPE_MAP = {}

TYPE_COMPONENT_MAP = {
    "parameter": PARAMETER_HYDRA_TYPE_MAP,
    "recorder": RECORDER_HYDRA_TYPE_MAP
}


def load_type_definitions(def_file="type_definitions.json"):
    def_path = os.path.join(os.path.dirname(__file__), def_file)
    try:
        with open(def_path, 'r') as fp:
            try:
                return json.load(fp)
            except json.decoder.JSONDecodeError as jde:
                raise ValueError(f"Type definitions file '{def_file}' has invalid JSON") from jde
    except OSError as oe:
        raise ValueError(f"Unable to read type definitions in {def_path}") from oe


def generate_data_types(definitions):
    for t in definitions:
        def_name = t.get("name")
        if not def_name or not isinstance(def_name, str):
            raise ValueError(f"Type definition omits <name>: {t}")
        # NB. 'name' need not be a valid Python class name; the type
        # needs to exist only to trigger the base's __init_subclass__,
        # which will also handle a missing 'tag'
        type_name = def_name.replace(" ", "")
        _ = type(f"Hydra{type_name}", (Descriptor,), t)
        if comp := t.get("component", None):
            comp_type_map = TYPE_COMPONENT_MAP[comp]
            comp_type = t.get("type")
            if not comp_type or not isinstance(comp_type, str):
                raise ValueError(f"Type specifies component but not type: {t}")
            if comp_type in comp_type_map:
                raise ValueError(f"{comp.title()} type map already contains {comp_type}")
            comp_type_map[comp_type] = t["tag"]
    missing = [t for t in REQUIRED_TYPES if t not in typemap]
    if missing:
        raise ValueError(f"Required types are not defined: {', '.join(missing)}")


hydra_pywr_data_types = load_type_definitions()
generate_data_types(hydra_pywr_data_types)


def lookup_parameter_hydra_datatype(value):
    ptype = value.type
    if not ptype.endswith("parameter"):
        ptype += "parameter"
    return PARAMETER_HYDRA_TYPE_MAP.get(ptype, "PYWR_PARAMETER")


def lookup_recorder_hydra_datatype(value):
    rtype = value.type
    if not rtype.endswith("recorder"):
        rtype += "recorder"
    return RECORDER_HYDRA_TYPE_MAP.get(rtype, "PYWR_RECORDER")
