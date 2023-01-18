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

import logging
log = logging.getLogger(__name__)


PARAMETER_HYDRA_TYPE_MAP = {
    "aggregatedparameter": "PYWR_PARAMETER_AGGREGATED",
    "constantscenarioparameter": "PYWR_PARAMETER_CONSTANT_SCENARIO",
    "controlcurveindexparameter": "PYWR_PARAMETER_CONTROL_CURVE_INDEX",
    "controlcurveinterpolatedparameter": "PYWR_PARAMETER_CONTROL_CURVE_INTERPOLATED",
    "dataframeparameter": "PYWR_DATAFRAME",
    "indexedarrayparameter": "PYWR_PARAMETER_INDEXED_ARRAY",
    "monthlyprofileparameter": "PYWR_PARAMETER_MONTHLY_PROFILE"
}

RECORDER_HYDRA_TYPE_MAP = {
    "flowdurationcurvedeviationrecorder": "PYWR_RECORDER_FDC_DEVIATION"
}

class PywrTypeEncoder(json.JSONEncoder):
    def default(self, inst):
        if isinstance(inst, (PywrParameter, PywrRecorder)):
            return inst.data
        else:
            return json.JSONEncoder.default(self, inst)



"""
    PywrNetwork => hydra_network
"""
def make_hydra_attr(name, desc=None):
    return { "name": name,
             "description": desc if desc else name
           }

class PywrToHydraNetwork():

    default_map_projection = "EPSG:4326"

    def __init__(self, network,
                       hydra=None,
                       hostname=None,
                       session_id=None,
                       user_id=None,
                       template_id=None,
                       project_id=None):
        self.hydra = hydra
        self.network = network
        self.hostname = hostname
        self.session_id = session_id
        self.user_id = user_id
        self.template_id = template_id
        self.project_id = project_id

        self._next_node_id = 0
        self._next_link_id = 0
        self._next_attr_id = 0


    def get_typeid_by_name(self, name):
        for t in self.template["templatetypes"]:
            if t["name"].lower() == name.lower():
                return t["id"]

    def get_hydra_network_type(self):
        for t in self.template["templatetypes"]:
            if t["resource_type"] == "NETWORK":
                return t

    def get_hydra_attrid_by_name(self, attr_name):
        if attr_name in self.template_attributes:
            return self.template_attributes[attr_name]

        for attr in self.hydra_attributes:
            if attr["name"] == attr_name:
                return attr["id"]

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

    def make_baseline_scenario(self, resource_scenarios):
        return { "name": "Baseline",
                 "description": "hydra-pywr Baseline scenario",
                 "resourcescenarios": resource_scenarios if resource_scenarios else []
               }


    def initialise_hydra_connection(self):
        if not self.hydra:
            from hydra_client.connection import JSONConnection
            self.hydra = JSONConnection(self.hostname, session_id=self.session_id, user_id=self.user_id)

        print(f"Retrieving template id '{self.template_id}'...")
        self.template = self.hydra.get_template(self.template_id)


    def build_hydra_network(self, projection=None):
        if projection:
            self.projection = projection
        else:
            self.projection = self.network.metadata.data.get("projection")
            if not self.projection:
                self.projection = self.__class__.default_map_projection

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
            "scenarios": [baseline_scenario],
            "projection": self.projection,
            "attributes": self.network_attributes,
            "types": [{ "id": self.network_hydratype["id"], "child_template_id": self.template_id }]
        }
        """
        breakpoint()
        with open("network.json", 'w') as fp:
            json.dump(self.hydra_network, fp, indent=2)
        """
        return self.hydra_network

    def build_network_attributes(self):
        exclude_metadata_attrs = ("title", "description", "projection")
        hydra_network_attrs = []
        resource_scenarios = []

        for attr_name in self.network.timestepper.data:
            ra, rs = self.make_resource_attr_and_scenario(self.network.timestepper, f"timestepper.{attr_name}")
            hydra_network_attrs.append(ra)
            resource_scenarios.append(rs)

        for attr_name in (a for a in self.network.metadata.data if a not in exclude_metadata_attrs):
            ra, rs = self.make_resource_attr_and_scenario(self.network.metadata, f"metadata.{attr_name}")
            hydra_network_attrs.append(ra)
            resource_scenarios.append(rs)

        for table_name, table in self.network.tables.items():
            ra, rs = self.make_resource_attr_and_scenario(table, table_name)
            hydra_network_attrs.append(ra)
            resource_scenarios.append(rs)
            """
            for attr_name in table.data:
                ra, rs = self.make_resource_attr_and_scenario(table, table_name)
                hydra_network_attrs.append(ra)
                resource_scenarios.append(rs)
            """

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

        return hydra_network_attrs, resource_scenarios

    def collect_template_attributes(self):
        template_attrs = {}
        for tt in self.template["templatetypes"]:
            for ta in tt["typeattrs"]:
                attr = ta["attr"]
                template_attrs[attr["name"]] = attr["id"]

        return template_attrs

    def register_hydra_attributes(self):
        timestepper_attrs = { 'timestepper.start', 'timestepper.end', 'timestepper.timestep'}
        excluded_attrs = { 'position', 'type' }
        pending_attrs = timestepper_attrs

        pending_attrs.add("scenarios")

        for node in self.network.nodes.values():
            for attr_name in node.data:
                pending_attrs.add(attr_name)

        for param_name in self.network.parameters:
            pending_attrs.add(param_name)

        for rec_name in self.network.recorders:
            pending_attrs.add(rec_name)

        for meta_attr in self.network.metadata.data:
            pending_attrs.add(f"metadata.{meta_attr}")

        for table_name, table in self.network.tables.items():
            for attr_name in table.data.keys():
                pending_attrs.add(table_name)

        attrs = [ make_hydra_attr(attr_name) for attr_name in pending_attrs - excluded_attrs.union(set(self.template_attributes.keys())) ]

        return self.hydra.add_attributes(attrs)


    def make_resource_attr_and_scenario(self, element, attr_name, datatype=None):
        local_attr_id = self.get_next_attr_id()

        if isinstance(element, (PywrParameter, PywrRecorder)):
            resource_scenario = self.make_paramrec_resource_scenario(element, attr_name, local_attr_id)
        elif isinstance(element, PywrTable):
            resource_scenario = self.make_table_resource_scenario(element, attr_name, local_attr_id)
        elif isinstance(element, (PywrMetadata, PywrTimestepper)):
            base, name = attr_name.split('.')
            resource_scenario = self.make_network_resource_scenario(element, name, local_attr_id)
        else:
            resource_scenario = self.make_resource_scenario(element, attr_name, local_attr_id)

        resource_attribute = { "id": local_attr_id,
                               "attr_id": self.get_hydra_attrid_by_name(attr_name),
                               "attr_is_var": "N"
                             }

        return resource_attribute, resource_scenario


    def make_direct_resource_attr_and_scenario(self, value, attr_name, hydra_datatype):

        local_attr_id = self.get_next_attr_id()

        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": json.dumps(value),
                    "metadata": "{}",
                    "unit": "-",
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


    def make_table_resource_scenario(self, element, attr_name, local_attr_id):
        hydra_datatype = self.lookup_hydra_datatype(element)
        dataset = { "name":  attr_name,
                    "type":  hydra_datatype,
                    "value": element.as_json(),
                    "metadata": "{}",
                    "unit": "-",
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
                    "hidden": 'N'
                  }

        resource_scenario = { "resource_attr_id": local_attr_id,
                              "dataset": dataset
                            }

        return resource_scenario


    def lookup_parameter_hydra_datatype(self, value):
        ptype = value.type
        if not ptype.endswith("parameter"):
            ptype += "parameter"

        return PARAMETER_HYDRA_TYPE_MAP.get(ptype, "PYWR_PARAMETER")


    def lookup_recorder_hydra_datatype(self, value):
        rtype = value.type
        if not rtype.endswith("recorder"):
            rtype += "recorder"

        return RECORDER_HYDRA_TYPE_MAP.get(rtype, "PYWR_RECORDER")


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
        elif isinstance(attr_value, PywrParameter):
            return self.lookup_parameter_hydra_datatype(attr_value)
        elif isinstance(attr_value, PywrRecorder):
            return self.lookup_recorder_hydra_datatype(attr_value)

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
                    raise ValueError(f"Node '{node.name}' attr '{attr_name}' has invalid attr id: '{attr_id}'")
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
                    x, y = coords[0], coords[1]
                hydra_node["x"] = x
                hydra_node["y"] = y

            hydra_nodes.append(hydra_node)

        return hydra_nodes, resource_scenarios


    def build_hydra_links(self):
        hydra_links = []
        resource_scenarios = []

        for edge in self.network.edges:
            resource_attributes = []

            src = edge.data[0]
            dest = edge.data[1]
            name = f"{src} to {dest}"

            hydra_link = {}
            hydra_link["resource_type"] = "LINK"
            hydra_link["id"] = self.get_next_link_id()
            hydra_link["name"] = name
            hydra_link["node_1_id"] = self.get_node_by_name(src)["id"]
            hydra_link["node_2_id"] = self.get_node_by_name(dest)["id"]
            hydra_link["layout"] = {}
            hydra_link["resource_attributes"] = resource_attributes
            hydra_link["types"] = [{ "id": self.get_typeid_by_name("edge") }]

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
        self.hydra.add_network(self.hydra_network)


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
