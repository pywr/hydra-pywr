import json
from numbers import Number

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

from .datatypes import (
    lookup_parameter_hydra_datatype,
    lookup_recorder_hydra_datatype
)

from pywrparser.types.network import PywrNetwork

from hydra_base.lib.objects import JSONObject

import logging
log = logging.getLogger(__name__)


def import_json(client, filename, project_id, template_id, network_name, *args, rewrite_url_prefix=None, appdata={}, projection='EPSG:4326'):
    """
        A utility function to import a Pywr JSON file into Hydra
    """

    log.info(f'Beginning import of "{filename}" to Project ID: {project_id}')

    if filename is None:
        raise Exception("No file specified")

    if project_id is None:
        raise Exception("No project specified")

    if template_id is None:
        raise Exception("No template specified")

    pnet, errors, warnings = PywrNetwork.from_file(filename)
    if warnings:
        for component, warns in warnings.items():
            for warn in warns:
                log.info(warn)

    if errors:
        for component, errs in errors.items():
            for err in errs:
                log.info(err)
        exit(1)

    if pnet:
        pnet.add_parameter_references()
        pnet.add_recorder_references()
        pnet.promote_inline_parameters()
        pnet.promote_inline_recorders()
        pnet.attach_reference_parameters()
        pnet.attach_reference_recorders()
        #pnet.detach_parameters()


    if network_name:
        pnet.metadata.data["title"] = network_name

    if rewrite_url_prefix:
        from .utils import file_to_s3
        for elem in [*pnet.parameters.values(), *pnet.tables.values()]:
            file_to_s3(elem.data, rewrite_url_prefix)

    importer = PywrToHydraNetwork(pnet, client=client, template_id=template_id, project_id=project_id)
    importer.build_hydra_network(projection, appdata=appdata)
    network_summary = importer.add_network_to_hydra()

    log.info(f"Imported {filename} to Project ID: {project_id}. Network ID ({network_summary['id']})")

    return network_summary


def import_json_as_scenario(client, filename, network_id, *args, appdata={}):
    """
        A utility function to import a Pywr JSON file into Hydra
    """

    log.info(f'Beginning import of "{filename}" to Project ID: {network_id}')

    if filename is None:
        raise Exception("No file specified")

    if network_id is None:
        raise Exception("No project specified")

    pnet, errors, warnings = PywrNetwork.from_file(filename)
    if warnings:
        for component, warns in warnings.items():
            for warn in warns:
                log.info(warn)

    if errors:
        for component, errs in errors.items():
            for err in errs:
                log.info(err)
        exit(1)

    if pnet:
        pnet.add_parameter_references()
        pnet.add_recorder_references()
        pnet.promote_inline_parameters()
        pnet.promote_inline_recorders()

    importer = PywrToHydraScenario(pnet, client=client)
    importer.build_hydra_scenario()
    network_summary = importer.add_scenario_to_hydra()

    log.info(f"Imported {filename} to Project ID: {project_id}")

    return network_summary


class PywrTypeEncoder(json.JSONEncoder):
    def default(self, inst):
        if isinstance(inst, (PywrParameter, PywrRecorder)):
            return inst.data
        else:
            return json.JSONEncoder.default(self, inst)


"""
    PywrNetwork => hydra_network
"""

class PywrToHydraScenario():


    def __init__(self, pywr_file,
                       client=None,
                       network_id=None):
        self.filename = pywr_file
        self.hydra_client = client
        self.network_id = network_id
        self.scenarioname = "Imported Scenario"
        self.network = self.hydra_client.get_network(network_id=network_id)

    def build_hydra_scenario(self):
        self.hydra_scenario = {
            "name": self.scenarioname,
            "description": f"Imported from Pywr File {self.filename}",
            "network_id": self.network_id,
            "resource_scenarios": [],
        }

    def make_resource_attr_and_scenario(self, element, attr_name, datatype=None):

        if isinstance(element, (PywrParameter, PywrRecorder)):
            resource_scenario = self.make_paramrec_resource_scenario(element, attr_name, local_attr_id)
        elif isinstance(element, (PywrTable, PywrMetadata, PywrTimestepper)):
            resource_scenario = self.make_typed_resource_scenario(element, attr_name, local_attr_id)
        else:
            resource_scenario = self.make_resource_scenario(element, attr_name, local_attr_id)

        resource_attribute = { "id": local_attr_id,
                               "attr_id": self.get_hydra_attrid_by_name(attr_name),
                               "attr_is_var": "N"
                             }

        return resource_attribute, resource_scenario


    def make_direct_resource_attr_and_scenario(self, value, attr_name, hydra_datatype, jsonify=True):

        local_attr_id = self.get_next_attr_id()
        ds_value = json.dumps(value) if jsonify else value

        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": ds_value,
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }

        resource_attribute = { "id": local_attr_id,
                               "attr_id": self.get_hydra_attrid_by_name(attr_name),
                               "attr_is_var": "N"
                             }

        return resource_attribute, resource_scenario


    def make_typed_resource_scenario(self, element, attr_name, local_attr_id):
        hydra_datatype = self.lookup_hydra_datatype(element)
        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": element.as_json(),
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }
        return resource_scenario


    def make_network_resource_scenario(self, element, attr_name, local_attr_id):

        value = element.data[attr_name]
        hydra_datatype = self.lookup_hydra_datatype(value)

        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": value,
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }
        return resource_scenario

    def make_paramrec_resource_scenario(self, element, attr_name, local_attr_id):

        value = element.data
        hydra_datatype = self.lookup_hydra_datatype(element)

        dataset = { "name":  element.name,
                    "type":  hydra_datatype,
                    "value": json.dumps(value),
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }
        return resource_scenario

    def make_resource_scenario(self, element, attr_name, local_attr_id):

        value = element.data[attr_name]
        hydra_datatype = self.lookup_hydra_datatype(value)

        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": json.dumps(value, cls=PywrTypeEncoder),
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }

        return resource_scenario


    def lookup_hydra_datatype(self, attr_value):
        if isinstance(attr_value, Number):
            return "SCALAR"
        elif isinstance(attr_value, list):
            return "ARRAY"
        elif isinstance(attr_value, dict):
            return "DATAFRAME"
        elif isinstance(attr_value, str):
            return "DESCRIPTOR"
        elif isinstance(attr_value, PywrTable):
            return "PYWR_TABLE"
        elif isinstance(attr_value, PywrTimestepper):
            return "PYWR_TIMESTEPPER"
        elif isinstance(attr_value, PywrMetadata):
            return "PYWR_METADATA"
        elif isinstance(attr_value, PywrParameter):
            return lookup_parameter_hydra_datatype(attr_value)
        elif isinstance(attr_value, PywrRecorder):
            return lookup_recorder_hydra_datatype(attr_value)

        raise ValueError(f"Unknown data type: '{attr_value}'")

    def build_hydra_node_data(self):
        resource_scenarios = []

        for node in self.network.nodes.values():
            resource_attributes = []

            exclude = ("name", "position", "type", "comment")

            for attr_name in node.data:
                if attr_name in exclude:
                    continue
                ra, rs = self.make_resource_attr_and_scenario(node, attr_name)
                if ra["attr_id"] == None:
                    raise ValueError(f"Node '{node.name}' attr '{attr_name}' has invalid attr id: \'{ra['attr_id']}\'")
                resource_attributes.append(ra)
                resource_scenarios.append(rs)

        return resource_scenarios


    def build_hydra_link_data(self):
        resource_scenarios = []

        for edge in self.network.edges:
            src = edge.data[0]
            dest = edge.data[1]
            if len(edge.data) == 4:
                src_slot = edge.data[2]
                dest_slot = edge.data[3]
                dest_slot_text = f"::{dest_slot}" if dest_slot else ""
                name = f"{src}::{src_slot} to {dest}{dest_slot_text}"
                src_ra, src_rs = self.make_direct_resource_attr_and_scenario(
                        src_slot,
                        "src_slot",
                        "DESCRIPTOR",
                        jsonify=False
                )
                resource_scenarios.append(src_rs)
                if dest_slot:
                    dest_ra, dest_rs = self.make_direct_resource_attr_and_scenario(
                             dest_slot,
                             "dest_slot",
                             "DESCRIPTOR",
                             jsonify=False
                    )
                    resource_scenarios.append(dest_rs)

        return resource_scenarios


    def build_parameters_recorders(self):
        resource_attrs = []
        resource_scenarios = []

        for param_name, param in self.network.parameters.items():
            ra, rs = self.make_resource_attr_and_scenario(param, param_name)
            resource_attrs.append(ra)
            resource_scenarios.append(rs)

        for rec_name, rec in self.network.recorders.items():
            ra, rs = self.make_resource_attr_and_scenario(rec, rec_name)
            resource_attrs.append(ra)
            resource_scenarios.append(rs)

        return resource_attrs, resource_scenarios


    def add_scenario_to_hydra(self):
        """ Pass network to Hydra"""
        scenario = JSONObject(self.hydra_scenario)
        scenario_response = self.hydra_client.add_scenario({"scenario": scenario})
        return scenario_response


class PywrToHydraNetwork():

    default_map_projection = "EPSG:4326"

    def __init__(self, network,
                       client=None,
                       hostname=None,
                       session_id=None,
                       template_id=None,
                       project_id=None):
        self.hydra_client = client
        self.network = network
        self.hostname = hostname
        self.session_id = session_id
        self.user_id = client.user_id
        self.template_id = template_id
        self.project_id = project_id

        self._next_node_id = 0
        self._next_link_id = 0
        self._next_attr_id = 0


    def get_typeid_by_name(self, name):
        for t in self.template["templatetypes"]:
            if t["name"].lower() == name.lower():
                return t["id"]
        log.critical(f"Template {self.template_id} does not define type {name}")

    def get_hydra_network_type(self):
        for t in self.template["templatetypes"]:
            if t["resource_type"] == "NETWORK":
                return t

    def get_hydra_attrid_by_name(self, attr_name):
        if attr_name.lower() in [a.lower() for a in self.template_attributes]:
            return self.template_attributes[attr_name]

        for attr in self.hydra_attributes:
            if attr["name"].lower() == attr_name.lower():
                return attr["id"]
        log.critical(f"Attr {attr_name} not registered")

    def get_next_node_id(self):
        self._next_node_id -= 1
        return self._next_node_id

    def get_next_link_id(self):
        self._next_link_id -= 1
        return self._next_link_id

    def get_next_attr_id(self):
        self._next_attr_id -= 1
        return self._next_attr_id

    def get_node_by_name(self, name):
        for node in self.hydra_nodes:
            if node["name"] == name:
                return node

    def make_hydra_attr(self, name, desc=None):
        return { "name": name,
                 "description": desc if desc else name,
                 "project_id": self.project_id
               }

    def make_baseline_scenario(self, resource_scenarios):
        return { "name": "Baseline",
                 "description": "hydra-pywr Baseline scenario",
                 "resourcescenarios": resource_scenarios if resource_scenarios else []
               }


    def initialise_hydra_connection(self):
        print(f"Retrieving template id '{self.template_id}'...")
        self.template = self.hydra_client.get_template(template_id=self.template_id)


    def build_hydra_network(self, projection=None, appdata={}):
        if projection:
            self.projection = projection
        else:
            self.projection = self.network.metadata.data.get("projection")
            if not self.projection:
                self.projection = self.__class__.default_map_projection

        if not self.projection:
            self.project = self.hydra_client.get_project(project_id=self.project_id)
            appdata = self.project.appdata
            if "projection" in appdata:
                log.info("Setting projection from project appdata: %s", appdata["projection"])
                self.projection = appdata["projection"]
        self.initialise_hydra_connection()

        self.network.promote_inline_parameters()

        self.template_attributes = self.collect_template_attributes()
        self.hydra_attributes = self.register_hydra_attributes()

        """ Build network elements and resource_scenarios with datasets """
        self.hydra_nodes, node_scenarios = self.build_hydra_nodes()

        self.network_attributes, network_scenarios = self.build_network_attributes()

        self.hydra_links, link_scenarios = self.build_hydra_links()
        paramrec_attrs, paramrec_scenarios = self.build_parameters_recorders()

        self.network_attributes += paramrec_attrs

        self.resource_scenarios = node_scenarios + network_scenarios + link_scenarios + paramrec_scenarios

        """ Create baseline scenario with resource_scenarios """
        baseline_scenario = self.make_baseline_scenario(self.resource_scenarios)

        """ Assemble complete network """
        network_name = self.network.metadata.data["title"]
        network_description = self.network.metadata.data["description"]
        self.network_hydratype = self.get_hydra_network_type()

        self.hydra_network = {
            "name": network_name,
            "description": network_description,
            "project_id": self.project_id,
            "nodes": self.hydra_nodes,
            "links": self.hydra_links,
            "layout": None,
            "appdata": appdata,
            "scenarios": [baseline_scenario],
            "projection": self.projection,
            "attributes": self.network_attributes,
            "types": [{ "id": self.network_hydratype["id"], "child_template_id": self.template_id }]
        }
        """
        with open("network.json", 'w') as fp:
            json.dump(self.hydra_network, fp, indent=2)
        """
        return self.hydra_network

    def build_network_attributes(self):
        hydra_network_attrs = []
        resource_scenarios = []

        for component in ("timestepper", "metadata"):
            comp_inst = getattr(self.network, component)
            ra, rs = self.make_resource_attr_and_scenario(comp_inst, component)
            hydra_network_attrs.append(ra)
            resource_scenarios.append(rs)

        for table_name, table in self.network.tables.items():
            ra, rs = self.make_resource_attr_and_scenario(table, table_name)
            hydra_network_attrs.append(ra)
            resource_scenarios.append(rs)

        scenario_data = [ scenario.data for scenario in self.network.scenarios ]
        if scenario_data:
            attr_name = "scenarios"
            ra, rs = self.make_direct_resource_attr_and_scenario(
                    {"scenarios": scenario_data},
                    attr_name,
                    "PYWR_SCENARIOS"
            )
            hydra_network_attrs.append(ra)
            resource_scenarios.append(rs)

        scenario_combinations = [ s_c.data for s_c in self.network.scenario_combinations ]
        if scenario_combinations:
            attr_name = "scenario_combinations"
            ra, rs = self.make_direct_resource_attr_and_scenario(
                    {'scenario_combinations': scenario_combinations},
                    attr_name,
                    "PYWR_SCENARIO_COMBINATIONS"
            )
            hydra_network_attrs.append(ra)
            resource_scenarios.append(rs)
        return hydra_network_attrs, resource_scenarios

    def collect_template_attributes(self):
        template_attrs = {}
        for tt in self.template["templatetypes"]:
            for ta in tt["typeattrs"]:
                attr = ta["attr"]
                template_attrs[attr["name"]] = attr["id"]

        return template_attrs

    def register_hydra_attributes(self):
        typed_network_attrs = { "timestepper", "metadata", "scenarios", "scenario_combinations" }
        excluded_attrs = { 'position', 'type' }
        pending_attrs = typed_network_attrs

        for node in self.network.nodes.values():
            for attr_name in node.data:
                pending_attrs.add(attr_name)

        for param_name in self.network.parameters:
            pending_attrs.add(param_name)

        for rec_name in self.network.recorders:
            pending_attrs.add(rec_name)

        for table_name in self.network.tables:
            pending_attrs.add(table_name)

        attrs = [ self.make_hydra_attr(attr_name) for attr_name in pending_attrs - excluded_attrs.union(set(self.template_attributes.keys())) ]

        return self.hydra_client.add_attributes(attrs=attrs)


    def make_resource_attr_and_scenario(self, element, attr_name, datatype=None):
        local_attr_id = self.get_next_attr_id()

        if isinstance(element, (PywrParameter, PywrRecorder)):
            resource_scenario = self.make_paramrec_resource_scenario(element, attr_name, local_attr_id)
        elif isinstance(element, (PywrTable, PywrMetadata, PywrTimestepper)):
            resource_scenario = self.make_typed_resource_scenario(element, attr_name, local_attr_id)
        else:
            resource_scenario = self.make_resource_scenario(element, attr_name, local_attr_id)

        resource_attribute = { "id": local_attr_id,
                               "attr_id": self.get_hydra_attrid_by_name(attr_name),
                               "attr_is_var": "N"
                             }

        return resource_attribute, resource_scenario


    def make_direct_resource_attr_and_scenario(self, value, attr_name, hydra_datatype, jsonify=True):

        local_attr_id = self.get_next_attr_id()
        ds_value = json.dumps(value) if jsonify else value

        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": ds_value,
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }

        resource_attribute = { "id": local_attr_id,
                               "attr_id": self.get_hydra_attrid_by_name(attr_name),
                               "attr_is_var": "N"
                             }

        return resource_attribute, resource_scenario


    def make_typed_resource_scenario(self, element, attr_name, local_attr_id):
        hydra_datatype = self.lookup_hydra_datatype(element)
        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": element.as_json(),
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }
        return resource_scenario


    def make_network_resource_scenario(self, element, attr_name, local_attr_id):

        value = element.data[attr_name]
        hydra_datatype = self.lookup_hydra_datatype(value)

        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": value,
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }
        return resource_scenario

    def make_paramrec_resource_scenario(self, element, attr_name, local_attr_id):

        value = element.data
        hydra_datatype = self.lookup_hydra_datatype(element)

        dataset = { "name":  element.name,
                    "type":  hydra_datatype,
                    "value": json.dumps(value),
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }
        return resource_scenario

    def make_resource_scenario(self, element, attr_name, local_attr_id):

        value = element.data[attr_name]
        hydra_datatype = self.lookup_hydra_datatype(value)

        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": json.dumps(value, cls=PywrTypeEncoder),
                    "metadata": "{}",
                    "unit": "-",
                    "unit_id": None,
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }

        return resource_scenario


    def lookup_hydra_datatype(self, attr_value):
        if isinstance(attr_value, Number):
            return "SCALAR"
        elif isinstance(attr_value, list):
            return "ARRAY"
        elif isinstance(attr_value, dict):
            return "DATAFRAME"
        elif isinstance(attr_value, str):
            return "DESCRIPTOR"
        elif isinstance(attr_value, PywrTable):
            return "PYWR_TABLE"
        elif isinstance(attr_value, PywrTimestepper):
            return "PYWR_TIMESTEPPER"
        elif isinstance(attr_value, PywrMetadata):
            return "PYWR_METADATA"
        elif isinstance(attr_value, PywrParameter):
            return lookup_parameter_hydra_datatype(attr_value)
        elif isinstance(attr_value, PywrRecorder):
            return lookup_recorder_hydra_datatype(attr_value)

        raise ValueError(f"Unknown data type: '{attr_value}'")


    def build_hydra_nodes(self):
        hydra_nodes = []
        resource_scenarios = []

        for node in self.network.nodes.values():
            resource_attributes = []

            exclude = ("name", "position", "type", "comment")

            for attr_name in node.data:
                if attr_name in exclude:
                    continue
                ra, rs = self.make_resource_attr_and_scenario(node, attr_name)
                if ra["attr_id"] == None:
                    raise ValueError(f"Node '{node.name}' attr '{attr_name}' has invalid attr id: \'{ra['attr_id']}\'")
                resource_attributes.append(ra)
                resource_scenarios.append(rs)

            hydra_node = {}
            hydra_node["resource_type"] = "NODE"
            hydra_node["id"] = self.get_next_node_id()
            hydra_node["name"] = node.name
            if comment := node.data.get("comment"):
                hydra_node["description"] = comment
            hydra_node["layout"] = {}
            hydra_node["attributes"] = resource_attributes
            hydra_node["types"] = [{ "id": self.get_typeid_by_name(node.type.lower()),
                                     "child_template_id": self.template_id
                                  }]
            if "position" in node.data:
                proj_data = node.data["position"]
                for coords in proj_data.values():
                    if "geographic" in coords:
                        coords = coords["geographic"]
                    elif "schematic" in coords:
                        coords = coords["schematic"]
                    if isinstance(coords, list):
                        x, y = coords[0], coords[1]
                    elif isinstance(coords, dict):
                        hydra_node['layout']['geojson'] = coords
                hydra_node["x"] = x
                hydra_node["y"] = y
            else:
                hydra_node["x"] = 0
                hydra_node["y"] = 0

            hydra_nodes.append(hydra_node)

        return hydra_nodes, resource_scenarios


    def build_hydra_links(self):
        hydra_links = []
        resource_scenarios = []

        for edge in self.network.edges:
            link_type = {}
            resource_attributes = []
            src = edge.data[0]
            dest = edge.data[1]
            if len(edge.data) == 4:
                src_slot = edge.data[2]
                dest_slot = edge.data[3]
                dest_slot_text = f"::{dest_slot}" if dest_slot else ""
                name = f"{src}::{src_slot} to {dest}{dest_slot_text}"
                link_type["id"] = self.get_typeid_by_name("slottededge")
                link_type["child_template_id"] = self.template_id
                src_ra, src_rs = self.make_direct_resource_attr_and_scenario(
                        src_slot,
                        "src_slot",
                        "DESCRIPTOR",
                        jsonify=False
                )
                resource_attributes.append(src_ra)
                resource_scenarios.append(src_rs)
                if dest_slot:
                    dest_ra, dest_rs = self.make_direct_resource_attr_and_scenario(
                             dest_slot,
                             "dest_slot",
                             "DESCRIPTOR",
                             jsonify=False
                    )
                    resource_attributes.append(dest_ra)
                    resource_scenarios.append(dest_rs)
            else:
                name = f"{src} to {dest}"
                link_type["id"] = self.get_typeid_by_name("edge")

            hydra_link = {
                "resource_type": "LINK",
                "id": self.get_next_link_id(),
                "name": name,
                "node_1_id": self.get_node_by_name(src)["id"],
                "node_2_id": self.get_node_by_name(dest)["id"],
                "layout": {},
                "attributes": resource_attributes,
                "types": [link_type]
            }

            hydra_links.append(hydra_link)

        return hydra_links, resource_scenarios


    def build_parameters_recorders(self):
        resource_attrs = []
        resource_scenarios = []

        for param_name, param in self.network.parameters.items():
            ra, rs = self.make_resource_attr_and_scenario(param, param_name)
            resource_attrs.append(ra)
            resource_scenarios.append(rs)

        for rec_name, rec in self.network.recorders.items():
            ra, rs = self.make_resource_attr_and_scenario(rec, rec_name)
            resource_attrs.append(ra)
            resource_scenarios.append(rs)

        return resource_attrs, resource_scenarios


    def add_network_to_hydra(self):
        """ Pass network to Hydra"""
        network = JSONObject(self.hydra_network)
        network_summary = self.hydra_client.add_network({"net": network})
        return network_summary


"""
    Utilities
"""
def unwrap_list(node_data):
    return [ i[0] for i in node_data ]

def build_times(data, node="/time"):
    raw_times = data.get_node(node).read().tolist()
    """ Profile times to determine period.
        A more rigorous solution is to include a token
        indicating the period (e.g. H, D, W, or M)
        in the hdf output.
    """
    if len(raw_times) > 1 and (raw_times[0][0] == raw_times[1][0] or raw_times[-2][0] == raw_times[-1][0]):
        # Probably hours...
        times = [ f"{t[0]:02}-{t[2]:02}-{t[3]} {t[1] % 24:02}:00:00" for t in raw_times ]
    else:
        # ...assume single day values
        times = [ f"{t[0]:02}-{t[2]:02}-{t[3]}" for t in raw_times ]

    return times

def build_node_dataset(node, times, node_attr="value"):
    raw_node_data = node.read().tolist()
    node_data = unwrap_list(raw_node_data)

    series = {}
    dataset = { "value": series}

    for t,v in zip(times, node_data):
        series[t] = v

    return dataset

def build_metric(node, data):
    return data


def build_parameter_dataset(param, times, stok='_'):
    node, _, attr = param.name.partition(stok)
    raw_param_data = param.read().tolist()
    param_data = unwrap_list(raw_param_data)

    series = {}
    dataset = { node: { attr: series} }

    for t,v in zip(times, param_data):
        series[t] = v

    return dataset
