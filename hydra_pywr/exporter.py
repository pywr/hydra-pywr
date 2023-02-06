from collections import defaultdict
from datetime import datetime
import json
import os
import re

from pywrparser.types import (
    PywrParameter,
    PywrRecorder,
    PywrTimestepper,
    PywrMetadata,
    PywrTable,
    PywrScenario,
    PywrNode,
    PywrEdge
)

from .rules import exec_rules
from .template import PYWR_SPLIT_LINK_TYPES

from hydra_base.lib.objects import JSONObject
from hydra_base.exceptions import ResourceNotFoundError


import logging
log = logging.getLogger(__name__)

PARAMETER_TYPES = (
    "PYWR_PARAMETER",
    "PYWR_DATAFRAME"
)

RECORDER_TYPES = (
    "PYWR_RECORDER",
)

CACHE_DIR = "/tmp/hydra_pywr_cache"

"""
    Hydra => PywrNetwork
"""
class HydraToPywrNetwork():

    exclude_hydra_attrs = (
        "id", "status", "cr_date",
        "network_id", "x", "y",
        "types", "attributes", "layout",
        "network", "description"
    )

    def __init__(self, client, network, network_id, scenario_id, attributes, template, **kwargs):
        self.hydra = client
        self.data = network
        self.network_id = network_id
        self.scenario_id = scenario_id
        self.attributes = attributes
        self.template = template

        self.type_id_map = {}
        for tt in self.template.templatetypes:
            self.type_id_map[tt.id] = tt

        self.attr_unit_map = {}
        self.hydra_node_by_id = {}

        self._parameter_recorder_flags = {}
        self._inline_parameter_recorder_flags = defaultdict(dict)
        self._node_recorder_flags = {}

        self.nodes = {}
        self.edges = []
        self.parameters = {}
        self.recorders = {}
        self.tables = {}
        self.scenarios = []


    @classmethod
    def from_scenario_id(cls, client, scenario_id, template_id=None, index=0, **kwargs):
        if kwargs.get("use_cache") is True:
            scen_cache_file = f"scenario_{scenario_id}.json"
            if not os.path.exists(CACHE_DIR):
                try:
                    os.mkdir(CACHE_DIR)
                except OSError:
                    log.error(f"Unable to create scenario cache at {CACHE_DIR}: defaulting to '/tmp'")
                    cache_dir = "/tmp"
            scen_cache_path = os.path.join(CACHE_DIR, scen_cache_file)
            if os.path.exists(scen_cache_path):
                mod_ts = os.path.getmtime(scen_cache_path)
                mod_dt = datetime.fromtimestamp(int(mod_ts))
                log.info(f"Using cached scenario updated at {mod_dt}")
                with open(scen_cache_path, 'r') as fp:
                    scenario = JSONObject(json.load(fp))
            else:
                    scenario = client.get_scenario(scenario_id=scenario_id, include_data=True, include_results=False, include_metadata=True, include_attr=False)
                    with open(scen_cache_path, 'w') as fp:
                        json.dump(scenario, fp)
                    log.info(f"Cached scenario written to '{scen_cache_path}'")

            network_id = scenario.network_id
            net_cache_file = f"network_{scenario.network_id}.json"
            net_cache_path = os.path.join(CACHE_DIR, net_cache_file)
            if os.path.exists(net_cache_path):
                mod_ts = os.path.getmtime(net_cache_path)
                mod_dt = datetime.fromtimestamp(int(mod_ts))
                log.info(f"Using cached network updated at {mod_dt}")
                with open(net_cache_path, 'r') as fp:
                    network = JSONObject(json.load(fp))
            else:
                network = client.get_network(
                            network_id=network_id,
                            include_data=False,
                            include_results=False,
                            template_id=template_id,
                            include_attributes=True)
                with open(net_cache_path, 'w') as fp:
                    json.dump(JSONObject(network), fp)
                log.info(f"Cached network written to '{net_cache_path}'")
        else:
            scenario = client.get_scenario(scenario_id=scenario_id, include_data=True, include_results=False, include_metadata=True, include_attr=False)
            network_id = scenario.network_id
            network = client.get_network(
                        network_id=network_id,
                        include_data=False,
                        include_results=False,
                        template_id=template_id,
                        include_attributes=True)

        network.scenarios = [scenario]
        network.rules = client.get_resource_rules(ref_key='NETWORK', scenario_id_id=scenario_id)

        attributes = client.get_attributes()
        attributes = {attr.id: attr for attr in attributes}

        log.info(f"Retreiving template {network.types[index].template_id}")
        template = client.get_template(template_id=network.types[index].template_id)

        return cls(client, network, network_id, scenario_id, attributes, template)


    def write_rules_as_module(self):
        filename = "hydra_pywr_custom_module.py"

        prelude = (
            "from pywr import recorders",
            "from pywr import parameters",
            "import pandas",
            "import numpy as np",
            "import scipy",
            "from pywr.nodes import *",
            "from pywr.parameters.control_curves import *",
            "from pywr.parameters._thresholds import *",
            "from pywr.parameters._hydropower import *",
            "from pywr.domains.river import *"
        )

        forbidden = ("import", "eval", "exec", "__builtins__")

        with open(filename, 'w') as fp:
            for p in prelude:
                fp.write(f"{p}\n")
            fp.write("\n")
            for rule in self.data.rules:
                for forbid in forbidden:
                    if forbid in rule["value"]:
                        raise PermissionError(f"Use of <{forbid}> forbidden in custom rules.")
                fp.write(rule["value"])
                fp.write("\n\n")


    def build_pywr_network(self):
        self.build_pywr_nodes()
        self.edges = self.build_edges()
        self.parameters, self.recorders = self.build_parameters_recorders()
        self.tables = self.build_tables()
        self.timestepper = self.build_timestepper()
        self.metadata = self.build_metadata()
        self.scenarios = self.build_scenarios()

        if len(self.data.rules) > 0:
            self.write_rules_as_module()

        return self


    def build_pywr_nodes(self):

        for node in self.data["nodes"]:
            pywr_node = {"name": node["name"]}

            self.hydra_node_by_id[node["id"]] = node

            if comment := node.get("description"):
                pywr_node["comment"] = comment

            pywr_node_type = node["types"][0]["name"]

            if pywr_node_type:
                log.info(f"Building node <{node['name']}> as <{pywr_node_type}>")
                self.build_node_and_references(node, pywr_node_type)


    def build_edges(self):
        edges = []

        for hydra_edge in self.data["links"]:
            src_hydra_node = self.hydra_node_by_id[hydra_edge["node_1_id"]]
            dest_hydra_node = self.hydra_node_by_id[hydra_edge["node_2_id"]]
            # Retrieve nodes from PywrNode store to verify presence
            try:
                # NB Lookup nodes with str key: self.nodes is Dict[str:PywrNode]
                src_node = self.nodes[str(src_hydra_node["name"])]
                dest_node = self.nodes[str(dest_hydra_node["name"])]
            except KeyError:
                # Not in this template...
                continue

            verts = [src_node.name, dest_node.name]

            if hydra_edge["types"][0]["name"].lower() == "slottededge":
                for slot in ("src_slot", "dest_slot"):
                    slot_id = [attr.id for attr in hydra_edge["attributes"] if attr.name == slot][0]
                    slot_ds = self.get_dataset_by_attr_id(slot_id)
                    verts.append(slot_ds.value if slot_ds else None)

            edge = PywrEdge(verts)
            edges.append(edge)

        return edges

    def build_tables(self):
        tables = {}
        table_attr_prefix = "tbl_"
        table_subattrs = ("header", "index_col", "key", "url")
        for attr in self.data["attributes"]:
            ds = self.get_dataset_by_attr_id(attr.id)
            if not ds:
                continue
            if ds["type"].upper().startswith("PYWR_TABLE"):
                # New style Table type: single dictionary value
                value = json.loads(ds["value"])
                table = PywrTable(ds["name"], value)
                tables[table.name] = table
            elif attr.name.lower().startswith(table_attr_prefix):
                # Old style deprecated Table: multiple subattrs w common prefix
                tablename = attr.name[len(table_attr_prefix):]
                for k in table_subattrs:
                    if tablename.endswith(f".{k}"):
                        tablename = tablename.replace(f".{k}", "")
                        try:
                            ds["value"] = float(ds["value"])
                        except ValueError:
                            try:
                                ds["value"] = json.loads(ds["value"])
                            except json.decoder.JSONDecodeError:
                                pass
                        if table := tables.get(tablename):
                            table.data[k] = ds["value"]
                        else:
                            table_data = {k: ds["value"]}
                            if k != "url":  # url key required for valid Table
                                table_data.update({"url": None})
                            tables[tablename] = PywrTable(tablename, table_data)
        return tables

    def build_timestepper(self):
        timestep = {}
        ts_attr_prefix = "timestepper"
        ts_keys = ("start", "end", "timestep")

        for attr in self.data["attributes"]:
            ds = self.get_dataset_by_attr_id(attr.id)
            if ds and ds["type"].upper().startswith("PYWR_TIMESTEPPER"):
                # New style Timestep type: single dictionary value
                value = json.loads(ds["value"])
                return PywrTimestepper(value)
            elif ds:
                # Deprecated multi-attr Timestep, must aggregate
                # all subattrs then create instance
                attr_group, *subs = attr.name.split('.')
                if attr_group != ts_attr_prefix:
                    continue
                ts_key = subs[-1]
                try:
                    value = json.loads(ds["value"])
                except json.decoder.JSONDecodeError:
                    value = ds["value"]
                timestep[ts_key] = value
            else:
                continue

        ts_val = timestep.get("timestep",1)
        try:
            tv = int(float(ts_val))
        except ValueError:
            tv = ts_val
        timestep["timestep"] = tv
        return PywrTimestepper(timestep)


    def build_metadata(self):
        metadata = {
            "title": self.data['name'],
            "description": self.data['description']
        }
        for attr in self.data["attributes"]:
            ds = self.get_dataset_by_attr_id(attr.id)
            if ds and ds["type"].upper().startswith("PYWR_METADATA"):
                # New style Metadata type: single dictionary value
                value = json.loads(ds["value"])
                return PywrMetadata(value)
            elif ds:
                # Deprecated multi-attr Metadata, must aggregate
                # all subattrs then create instance
                attr_group, *subs = attr.name.split('.')
                if attr_group != "metadata":
                    continue
                meta_key = subs[-1]
                try:
                    value = json.loads(ds["value"])
                except json.decoder.JSONDecodeError:
                    value = ds["value"]
                metadata[meta_key] = value
            else:
                continue
        """
          minimum_version is an optional metadata key, but
          Pywr requires it to be a string if present.
        """
        minver = metadata.get("minimum_version")
        if minver and not isinstance(minver, str):
            metadata["minimum_version"] = str(minver)

        return PywrMetadata(metadata)


    def build_scenarios(self):
        try:
            scenarios_dataset = self.get_network_attr(self.scenario_id, self.data["id"], "scenarios")
            scenarios = [ PywrScenario(scenario) for scenario in scenarios_dataset["scenarios"] ]
        except (ResourceNotFoundError, ValueError):
            scenarios = []

        return scenarios


    def build_parameters_recorders(self):
        parameters = {} # {name: P()}
        recorders = {} # {name: R()}

        for attr in self.data.attributes:
            ds = self.get_dataset_by_attr_id(attr.id)
            if not ds:
                # This could raise instead, e.g...
                #raise ValueError(f"No dataset found for attr name {attr.name} with id {attr.id}")
                continue
            if not ds["type"].startswith(PARAMETER_TYPES + RECORDER_TYPES):
                continue
            if ds["type"].startswith(PARAMETER_TYPES):
                value = json.loads(ds["value"])
                value = unnest_parameter_key(value, key="pandas_kwargs")
                value = add_interp_kwargs(value)
                p = PywrParameter(ds["name"], value)
                assert p.name not in parameters    # Disallow overwriting
                parameters[p.name] = p
            elif ds["type"].startswith(RECORDER_TYPES):
                value = json.loads(ds["value"])
                try:
                    r = PywrRecorder(ds["name"], value)
                except:
                    raise ValueError(f"Dataset {ds['name']} is not a valid Recorder")
                recorders[r.name] = r

        return parameters, recorders


    def get_network_attr(self, scenario_id, network_id, attr_key):

        net_attr = self.hydra.get_attribute_by_name_and_dimension(name=attr_key, dimension_id=None)
        ra = self.hydra.get_resource_attributes(ref_key="network", ref_id=network_id)
        ra_id = None
        for r in ra:
            if r["attr_id"] == net_attr["id"]:
                ra_id = r["id"]

        if not ra_id:
            raise ValueError(f"Resource attribute for {attr_key} not found in scenario {scenario_id} on network {network_id}")

        data = self.hydra.get_resource_scenario(resource_attr_id=ra_id, scenario_id=scenario_id, get_parent_data=False)
        attr_data = json.loads(data["dataset"]["value"])

        return attr_data # NB: String keys


    def get_dataset_by_attr_id(self, attr_id):

        scenario = self.data.scenarios[0]
        for rs in scenario.resourcescenarios:
            if rs.resource_attr_id == attr_id:
                return rs.dataset

    def _get_resource_scenario(self, resource_attribute_id):

        for scenario in self.data["scenarios"]:
            for resource_scenario in scenario["resourcescenarios"]:
                if resource_scenario["resource_attr_id"] == resource_attribute_id:
                    return resource_scenario

        raise ValueError(f"No resource scenario found for resource attribute id: {resource_attribute_id}")


    def build_node_and_references(self, nodedata, pywr_node_type):

        for resource_attribute in nodedata["attributes"]:
            attribute = self.attributes[resource_attribute["attr_id"]]
            try:
                resource_scenario = self._get_resource_scenario(resource_attribute["id"])
            except ValueError:
                continue  # No data associated with this attribute.

            # Allow export of probable recorders
            if resource_attribute["attr_is_var"] == 'Y' and "recorder" not in attribute["name"].lower():
                continue

            attribute_name = attribute["name"]
            dataset = resource_scenario["dataset"]
            dataset_type = dataset["type"]
            value = dataset["value"]

            try:
                typedval = json.loads(value)
            except json.decoder.JSONDecodeError as e:
                typedval = value
            nodedata[attribute_name] = typedval

        nodedata["type"] = pywr_node_type
        node_attr_data = {a:v for a,v in nodedata.items() if a not in self.exclude_hydra_attrs}
        position = {"geographic": [ nodedata.get("x",0), nodedata.get("y",0) ]}
        node_attr_data["position"] = position

        if comment := nodedata.get("description"):
            node_attr_data["comment"] = comment

        node = PywrNode(node_attr_data)
        self.nodes[node.name] = node

"""
  Compatibility patches: these update the Pywr data output of
  get_pywr_data to replace deprecated syntax with that of current
  Pywr versions.
"""
def unnest_parameter_key(param_data, key="pandas_kwargs"):
    """
        Relocates all keys inside parameters' <key> arg
        to the top level of that parameter and removes the
        original <key>.
    """
    if key in param_data:
        for k, v in param_data[key].items():
            param_data[k] = v
        del param_data[key]

    return param_data

def add_interp_kwargs(param_data):
    """
        Replaces the deprecated `kind` key of interpolatedvolume
        parameters with the nested `interp_kwargs` key.
    """
    ptype = "interpolatedvolume"
    new_key = "interp_kwargs"
    if param_data["type"].lower().startswith(ptype) and "kind" in param_data:
        param_data[new_key] = {"kind": param_data["kind"]}
        del param_data["kind"]

    return param_data
