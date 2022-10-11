import json
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
from pywrparser.types.network import PywrNetwork as NewPywrNetwork

from .rules import exec_rules

import logging
log = logging.getLogger(__name__)


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

    def __init__(self, client, network, scenario_id, attributes, template, **kwargs):
        self.hydra = client
        self.data = network
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
        network = client.get_network(scenario.network_id, include_data=True, include_results=False, template_id=None)
        network.scenarios = [scenario]
        network.rules = client.get_resource_rules('NETWORK', scenario.network_id)

        attributes = client.get_attributes()
        attributes = {attr.id: attr for attr in attributes}

        log.info(f"Retreiving template {network.types[index].template_id}")
        template = client.get_template(network.types[index].template_id)

        return cls(client, network, scenario_id, attributes, template)


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


    def build_pywr_network(self, domain=None):
        self.build_pywr_nodes()
        self.edges = self.build_edges()
        self.parameters, self.recorders = self.build_parameters_recorders()
        if domain:
            self.timestepper, self.metadata, self.scenarios = self.build_integrated_network_attrs(domain)
        else:
            self.timestepper, self.metadata, self.tables, self.scenarios = self.build_network_attrs()

        if len(self.data.rules) > 0:
            self.write_rules_as_module()

        return self


    def build_pywr_nodes(self):

        for node in self.data["nodes"]:
            pywr_node = {"name": node["name"]}

            self.hydra_node_by_id[node["id"]] = node

            if comment := node.get("description"):
                pywr_node["comment"] = comment

            # Get the type for this node from the template
            pywr_node_type = node["types"][0]["name"]
            """
            real_template_id = node["types"][0]["template_id"]

            for node_type in node["types"]:
                try:
                    #log.info(f"====\nnode: {node}")
                    if real_template_id != self.template["id"]:
                        continue
                    pywr_node_type = self.type_id_map[node_type["id"]]["name"]
                    break
                except KeyError:
                    breakpoint()
                    # Skip as not in this template...
                    pywr_node_type = None
                    continue
            """
            #log.info(f"Found node type {pywr_node_type} for node {node['name']} with nt_id {pywr_node_type['id']} on template {self.template['id']}\n====")

            #if pywr_node_type is None:
            #    raise ValueError('Template does not contain node of type "{}".'.format(pywr_node_type))

            # Skip if not in this template...
            if pywr_node_type:
                log.info(f"Building node {node['name']} as {pywr_node_type}...")
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

            edge = PywrEdge([src_node.name, dest_node.name])
            edges.append(edge)

        return edges


    def build_parameters_recorders(self):
        # attr_id = data.network.attributes[x].id
        parameters = {} # {name: P()}
        recorders = {} # {name: R()}

        for attr in self.data.attributes:
            ds = self.get_dataset_by_attr_id(attr.id)
            if not ds:
                #raise ValueError(f"No dataset found for attr name {attr.name} with id {attr.id}")
                continue
            if not ds["type"].startswith(("PYWR_PARAMETER", "PYWR_DATAFRAME", "PYWR_RECORDER")):
                continue
            if ds["type"].startswith(("PYWR_PARAMETER", "PYWR_DATAFRAME")):
                value = json.loads(ds["value"])
                p = PywrParameter(ds["name"], value)
                assert p.name not in parameters    # Disallow overwriting
                parameters[p.name] = p
            elif ds["type"].startswith("PYWR_RECORDER"):
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

        meta_inst = PywrMetadata(metadata)

        """ Tables """

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

        """ Scenarios """

        try:
            scenarios_dataset = self.get_network_attr(self.scenario_id, self.data["id"], "scenarios")
            scenarios = [ PywrScenario(scenario) for scenario in scenarios_dataset["scenarios"] ]
        except ValueError as e:
            scenarios = []

        return ts_inst, meta_inst, tables, scenarios


    def build_integrated_network_attrs(self, domain):
        domain_data_key = f"{domain}_data"
        domain_attr = self.get_attr_by_name(domain_data_key)
        dataset = self.get_dataset_by_attr_id(domain_attr.id)
        data = json.loads(dataset["value"])

        timestep = data["timestepper"]
        ts_val = timestep.get("timestep",1)
        try:
            tv = int(float(ts_val))
        except ValueError:
            tv = ts_val

        timestep["timestep"] = tv
        ts_inst = PywrTimestepper(timestep)

        metadata = data["metadata"]
        meta_inst = PywrMetadata(metadata)

        scen_insts = [ PywrScenario(s) for s in data.get("scenarios") ]

        return ts_inst, meta_inst, scen_insts


    def get_network_attr(self, scenario_id, network_id, attr_key):
        net_attr = self.hydra.get_attribute_by_name_and_dimension(attr_key, None)
        ra = self.hydra.get_resource_attributes("network", network_id)
        ra_id = None
        for r in ra:
            if r["attr_id"] == net_attr["id"]:
                ra_id = r["id"]

        if not ra_id:
            raise ValueError(f"Resource attribute for {attr_key} not found in scenario {scenario_id} on network {network_id}")

        data = self.hydra.get_resource_scenario(ra_id, scenario_id, get_parent_data=False)
        attr_data = json.loads(data["dataset"]["value"])

        return attr_data # NB: String keys


    def get_dataset_by_attr_id(self, attr_id):
        # d = data.scenarios[0].resourcescenarios[x]
        # d.resource_attr_id == attr_id
        # d.dataset

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
            if resource_attribute["attr_is_var"] == 'Y' and recorder not in attribute["name"].lower():
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
