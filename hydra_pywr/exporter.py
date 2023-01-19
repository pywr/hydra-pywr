import json
import re
from collections import defaultdict

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
    def from_scenario_id(cls, client, scenario_id, template_id=None, index=0):

        scenario = client.get_scenario(scenario_id, include_data=True, include_results=False, include_metadata=True, include_attr=False)
        network_id = scenario.network_id
        network = client.get_network(network_id, include_data=True, include_results=False, template_id=None)
        network.scenarios = [scenario]
        network.rules = client.get_resource_rules('NETWORK', network_id)

        attributes = client.get_attributes()
        attributes = {attr.id: attr for attr in attributes}

        log.info(f"Retreiving template {network.types[index].template_id}")
        template = client.get_template(network.types[index].template_id)

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

        forbidden = ("import", "eval", "exec")

        with open(filename, 'w') as fp:
            for p in prelude:
                fp.write(f"{p}\n")
            fp.write("\n")
            for rule in self.data.rules:
                for forbid in forbidden:
                    if forbid in rule["value"]:
                        raise PermissionError(f"Use of {forbid} statement forbidden in custom rules.")
                fp.write(rule["value"])
                fp.write("\n\n")


    def build_pywr_network(self):
        self.build_pywr_nodes()
        self.edges = self.build_edges()
        self.parameters, self.recorders = self.build_parameters_recorders()
        self.tables = self.build_tables()
        self.timestepper, self.metadata, self.scenarios = self.build_network_attrs()

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
                log.info(f"Building node {node['name']} as {pywr_node_type}...")
                self.build_node_and_references(node, pywr_node_type)


    def build_edges(self):
        edges = []
        slot_pattern_text = r":slot\((.+?)\)"
        slot_pattern = re.compile(slot_pattern_text)

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

            matches = 0
            for match in slot_pattern.finditer(hydra_edge["name"]):
                slot = match.group(1)
                verts.append(slot)
                if matches == 1:
                    # Take max of two slots
                    break
                matches += 1

            if len(verts) == 3:
                # Slotted links must be [src_node, dest_node, src_slot, (dest_slot|null)]
                verts.append(None)

            if len(verts) > 2:
                log.info(f"slotted edge with verts {verts}")

            edge = PywrEdge(verts)
            edges.append(edge)

        return edges

    def build_tables(self):
        tables = {}
        for attr in self.data.attributes:
            ds = self.get_dataset_by_attr_id(attr.id)
            if not ds:
                continue
            if ds["type"].upper().startswith("PYWR_TABLE"):
                value = json.loads(ds["value"])
                table = PywrTable(ds["name"], value)
                tables[table.name] = table

        return tables


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

    def build_network_attrs(self):
        """ TimeStepper, Metadata, and Tables instances """

        timestep = {}
        ts_keys = ("start", "end", "timestep")

        for attr in self.data["attributes"]:
            attr_group, *subs = attr.name.split('.')
            if attr_group != "timestepper":
                continue
            dataset = self.get_dataset_by_attr_id(attr.id)
            ts_key = subs[-1]
            try:
                value = json.loads(dataset["value"])
            except json.decoder.JSONDecodeError:
                value = dataset["value"]
            timestep[ts_key] = value


        ts_val = timestep.get("timestep",1)
        try:
            tv = int(float(ts_val))
        except ValueError:
            tv = ts_val
        timestep["timestep"] = tv
        ts_inst = PywrTimestepper(timestep)

        """ Metadata """
        metadata = {"title": self.data['name'],
                    "description": self.data['description']
                   }
        for attr in self.data["attributes"]:
            attr_group, *subs = attr.name.split('.')
            if attr_group != "metadata":
                continue
            dataset = self.get_dataset_by_attr_id(attr.id)
            meta_key = subs[-1]
            try:
                value = json.loads(dataset["value"])
            except json.decoder.JSONDecodeError:
                value = dataset["value"]
            metadata[meta_key] = value

        """
          minimum_version is an optional metadata key, but
          Pywr requires it to be a string if present.
        """
        minver = metadata.get("minimum_version")
        if minver and not isinstance(minver, str):
            metadata["minimum_version"] = str(minver)

        meta_inst = PywrMetadata(metadata)

        """ Tables """

        """
        table_prefix = "tbl_"
        tables_data = defaultdict(dict)
        tables = {}
        for attr in self.data["attributes"]:
            if not attr.name.startswith(table_prefix):
                continue
            table_name, table_attr = attr.name[len(table_prefix):].split('.')
            dataset = self.get_dataset_by_attr_id(attr.id)
            try:
                value = json.loads(dataset["value"])
            except json.decoder.JSONDecodeError:
                value = dataset["value"]
            tables_data[table_name][table_attr] = value

        for tname, tdata in tables_data.items():
            tables[tname] = PywrTable(tname, tdata)
        """
        """ Scenarios """

        try:
            scenarios_dataset = self.get_network_attr(self.scenario_id, self.data["id"], "scenarios")
            scenarios = [ PywrScenario(scenario) for scenario in scenarios_dataset["scenarios"] ]
        except (ResourceNotFoundError, ValueError):
            scenarios = []

        return ts_inst, meta_inst, scenarios


    def get_network_attr(self, scenario_id, network_id, attr_key):

        net_attr = self.hydra.get_attribute_by_name_and_dimension(attr_key, None)
        ra = self.hydra.get_resource_attributes("network", network_id)
        ra_id = None
        for r in ra:
            if r["attr_id"] == net_attr["id"]:
                ra_id = r["id"]

        if not ra_id:
            raise ValueError(f"Resource attribute for {attr_key} not found in scenario {scenario_id} on network {network_id}")

        #if ra_id == 1773645:
        #    breakpoint()

        data = self.hydra.get_resource_scenario(ra_id, scenario_id, get_parent_data=False)
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

        try:
            node = PywrNode(node_attr_data)
        except Exception as e:
            breakpoint()

        self.nodes[node.name] = node
