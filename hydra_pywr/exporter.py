import json

from collections import defaultdict

from hydra_base.lib.HydraTypes.Registry import typemap as hydra_typemap

from hydra_pywr.template import PYWR_SPLIT_LINK_TYPES

from hydra_pywr_common.datatypes import PywrParameter, PywrRecorder

from hydra_pywr_common.types import PywrDataReference

import hydra_pywr_common

from hydra_pywr_common.types.nodes import(
    PywrNode
)
from hydra_pywr_common.types.base import(
    PywrEdge
)

from hydra_pywr_common.types.fragments.network import(
    Timestepper,
    Metadata,
    Table,
    Scenario
)

from .core import BasePywrHydra

from hydra_pywr_common.types.fragments.config import IntegratedConfig
#from hydra_pywr_common.types.parameters import *
#from hydra_pywr_common.types.recorders import *

import logging
log = logging.getLogger(__name__)

COST_ALIASES = ['allocation penalty', 'allocation_penalty', 'Allocation Penalty']
EXCLUDE_HYDRA_ATTRS = ("id", "status", "cr_date", "network_id", "x", "y",
                       "types", "attributes", "layout", "network", "description")


class PywrHydraExporter(BasePywrHydra):
    def __init__(self, client, data, scenario_id, attributes, template):

        super().__init__()

        self.data = data
        self.scenario_id = scenario_id
        self.attributes = attributes
        self.client = client
        self.template = template

        self.type_id_map = {}
        for tt in self.template.templatetypes:
            self.type_id_map[tt.id] = tt

        self.attr_unit_map = {}
        self.hydra_node_lookup = {}

        self._parameter_recorder_flags = {}
        self._inline_parameter_recorder_flags = defaultdict(dict)
        self._node_recorder_flags = {}

        self.timestepper = None
        self.metadata = None

        self.nodes = {}
        self.edges = []
        self.parameters = {}
        self.recorders = {}
        self.tables = {}
        self.scenarios = []

        self._pattern_templates = None


    @classmethod
    def from_scenario_id(cls, client, scenario_id, template_id=None, index=0, **kwargs):
        scenario = client.get_scenario(scenario_id, include_data=True, include_results=False, include_metadata=False, include_attr=False)
        # Fetch the network
        network = client.get_network(scenario.network_id, include_data=False, include_results=False, template_id=template_id)

        network.scenarios = [scenario]

        # Fetch all the attributes
        attributes = client.get_attributes()
        attributes = {attr.id: attr for attr in attributes}

        template = None

        if template_id is not None:
            template = client.get_template(template_id)
        #elif len(network.types) == 1:
        else:
            template = client.get_template(network.types[index].template_id)


        return cls(client, network, scenario_id, attributes, template, **kwargs)


    def get_pywr_data(self, domain=None):
        self.generate_pywr_nodes()
        self.edges = self.build_edges()

        if domain is not None:
            self.timestepper, self.metadata, self.scenarios = self.build_integrated_network_attrs(domain)
        else:
            self.timestepper, self.metadata, self.tables = self.build_network_attrs()

        return self


    def _get_all_resource_attributes(self):
        """
            Get all the complex mode attributes in the network so that they
            can be used for mapping to resource scenarios later.
        """

        for a in self.data['attributes']:
            yield a

        for rtype in ('nodes', 'links', 'resourcegroups'):
            for o in self.data[rtype]:
                for a in o['attributes']:
                    yield a

    def _get_resource_scenario(self, resource_attribute_id):

        # TODO this just returns the first resource scenario that is found.
        for scenario in self.data['scenarios']:
            for resource_scenario in scenario['resourcescenarios']:
                if resource_scenario['resource_attr_id'] == resource_attribute_id:
                    return resource_scenario

        raise ValueError('No resource scenario found for resource attribute id: {}'.format(resource_attribute_id))


    def generate_pywr_nodes(self):
        """ Generator returning a Pywr dict for each node in the network. """

        for node in self.data['nodes']:
            # Create the basic information.
            pywr_node = {'name': node['name']}

            self.hydra_node_lookup[node['id']] = node

            if node.get('description', None) is not None:
                pywr_node['comment'] = node['description']

            # Get the type for this node from the template
            pywr_node_type = None
            for node_type in node['types']:
                try:
                    pywr_node_type = self.type_id_map[node_type['id']]['name']
                except KeyError:
                    # Skip as not in this template...
                    continue

            #if pywr_node_type is None:
            #    raise ValueError('Template does not contain node of type "{}".'.format(pywr_node_type))


            # Skip as not in this template...
            if pywr_node_type:
                self.build_node_and_references(node, pywr_node_type)


    def build_node_and_references(self, nodedata, pywr_node_type):

        for resource_attribute in nodedata['attributes']:
            attribute = self.attributes[resource_attribute['attr_id']]
            try:
                resource_scenario = self._get_resource_scenario(resource_attribute['id'])
            except ValueError:
                continue  # No data associated with this attribute.

            if resource_attribute['attr_is_var'] == 'Y':
                continue

            attribute_name = attribute['name']
            dataset = resource_scenario['dataset']
            dataset_type = dataset['type']
            value = dataset['value']
            try:
                typedval = json.loads(value)
                if isinstance(typedval, dict) and typedval.get('__recorder__') is not None:
                    self._parameter_recorder_flags[attribute_name] = typedval.pop('__recorder__')
                #remove the contents of the pandas_kwargs sub-dict ane put them on the parent dict
                if isinstance(typedval, dict) and typedval.get('pandas_kwargs') is not None:
                    for k, v in typedval.get('pandas_kwargs').items():
                        typedval[k] = v
                    del(typedval['pandas_kwargs'])
                    print(typedval['parse_dates'])
            except json.decoder.JSONDecodeError as e:
                typedval = value
            nodedata[attribute_name] = typedval

        nodedata["type"] = pywr_node_type
        node_attr_data = {a:v for a, v in nodedata.items() if a not in EXCLUDE_HYDRA_ATTRS}
        position = {"geographic": [nodedata.get("x", 0), nodedata.get("y", 0)]}
        node_attr_data["position"] = position

        dev_node = PywrNode.NodeFactory(node_attr_data)

        self.nodes[dev_node.name] = dev_node
        self.parameters.update(dev_node.parameters)
        self.recorders.update(dev_node.recorders)


    def normalise(self, name):
        return name.lower().replace("_", "").replace(" ", "")

    def build_edges(self):
        edges = {}

        for hydra_edge in self.data["links"]:
            src_hydra_node = self.hydra_node_lookup[hydra_edge["node_1_id"]]
            dest_hydra_node = self.hydra_node_lookup[hydra_edge["node_2_id"]]

            from_node_types = self.get_type_map(src_hydra_node)
            node_type_names = set([nt.lower() for nt in from_node_types.values()])
            # Retrieve nodes from PywrNode store to verify presence
            try:
                src_node = self.nodes[src_hydra_node["name"]]
                dest_node = self.nodes[dest_hydra_node["name"]]
            except KeyError:
                # Not in this template...
                continue


            if len(set(PYWR_SPLIT_LINK_TYPES).intersection(node_type_names)) > 0:
                #This is a split link, which has 4 entries, the third one bing the name
                #of the source node and the last one being None,
                slot_name = dest_node.name
                for s in src_node.slot_names.get_value():
                    #TODO: THis is a massive hack. How wan we identify the correct slot
                    #names if they are not node names in the model???
                    if self.normalise(s).endswith(self.normalise(dest_node.name)):
                        slot_name = s
                edge = PywrEdge((src_node.name, dest_node.name, slot_name, None))
            else:
                edge = PywrEdge((src_node.name, dest_node.name))  # NB Call ctor directly with tuple here, no factory
            edges[edge.name] = edge

        return edges

    def get_type_map(self, resource):
        """
        for a given resource (node, link, group) get the type id:name map for it
        ex: node.types = [{id: 1, name: type1}, {id: 11, name: type11}
        returns:
            {
             1: type1
             11: type11
            }
        """
        type_map = {}

        for t in resource.get('types', []):
            type_map[t['id']] = self.type_id_map[t['id']]['name']

        return type_map

    def build_integrated_network_attrs(self, domain):
        domain_data_key = f"{domain}_data"
        domain_attr = self.get_attr_by_name(domain_data_key)
        resource_scenario = self._get_resource_scenario(domain_attr.id)
        dataset = resource_scenario["dataset"]
        data = json.loads(dataset["value"])
        timestep = data["timestepper"]

        ts_val = timestep.get("timestep",1)
        try:
            tv = int(float(ts_val))
        except ValueError:
            tv = ts_val
        timestep["timestep"] = tv
        ts_inst = Timestepper(timestep)

        metadata = data["metadata"]
        meta_inst = Metadata(metadata)

        scenarios = data["scenarios"]
        scen_insts = [ Scenario(s) for s in scenarios ]

        return ts_inst, meta_inst, scen_insts


    def get_integrated_config(self, config_key="config"):
        config_attr = self.client.get_attribute_by_name_and_dimension(config_key, None)
        ra = self.client.get_resource_attributes("network", self.data.id)
        ra_id = None
        for r in ra:
            if r["attr_id"] == config_attr["id"]:
                ra_id = r["id"]

        data = self.client.get_resource_scenario(ra_id, self.scenario_id, get_parent_data=False)
        attr_data = json.loads(data["dataset"]["value"])

        return IntegratedConfig(attr_data)


    def get_attr_by_name(self, name):
        for attr in self.data["attributes"]:
            if attr.name == name:
                return attr

        raise KeyError(f"No attr named '{name}'")


    def build_network_attrs(self):
        """ TimeStepper and Metadata instances """

        timestep = {}
        ts_keys = ("start", "end", "timestep")

        for attr in self.data["attributes"]:
            attr_group, *subs = attr.name.split('.')
            if attr_group != "timestepper":
                continue
            resource_scenario = self._get_resource_scenario(attr.id)
            dataset = resource_scenario["dataset"]
            #ts_key = attr.name.split('.')[-1]
            ts_key = subs[-1]
            timestep[ts_key] = dataset["value"]


        ts_val = timestep.get("timestep", 1)
        try:
            tv = int(float(ts_val))
        except ValueError:
            tv = ts_val
        timestep["timestep"] = tv
        ts_inst = Timestepper(timestep)

        """ Metadata """
        metadata = {"title": self.data['name'],
                    "description": self.data['description']
                   }
        for attr in self.data["attributes"]:
            attr_group, *subs = attr.name.split('.')
            if attr_group != "metadata":
                continue
            resource_scenario = self._get_resource_scenario(attr.id)
            dataset = resource_scenario["dataset"]
            meta_key = subs[-1]
            metadata[meta_key] = dataset["value"]

        meta_inst = Metadata(metadata)

        """ Tables """
        tables_data = defaultdict(dict)
        tables = {}
        for attr in self.data["attributes"]:
            if not attr.name.startswith("tbl_"):
                continue
            table_name, table_attr = attr.name[4:].split('.')
            resource_scenario = self._get_resource_scenario(attr.id)
            dataset = resource_scenario["dataset"]
            tables_data[table_name][table_attr] = dataset["value"]

        for tname, tdata in tables_data.items():
            tables[tname] = Table(tdata)

        """ Parameters """
        print(self.data.keys())
        for attr in self.data["attributes"]:
            resource_scenario = self._get_resource_scenario(attr.id)
            dataset = resource_scenario["dataset"]
            dataset_type = hydra_typemap[dataset.type.upper()]
            is_parameter_or_recorder = False
            try:
                data = json.loads(dataset.value)
                if not isinstance(data, dict):
                    raise ValueError("Not a dict")

                if data.get('type') is not None:
                    is_parameter_or_recorder = True

                if data.get('__recorder__') is not None:
                    self._parameter_recorder_flags[attr.name] = data.pop('__recorder__')
                #remove the contents of the pandas_kwargs sub-dict ane put them on the parent dict
                if data.get('pandas_kwargs') is not None:
                    for k, v in data.get('pandas_kwargs').items():
                        data[k] = v
                    del(data['pandas_kwargs'])
            except ValueError as e:
                print(e)
                pass

            if is_parameter_or_recorder is True:
                parameter = PywrDataReference.ReferenceFactory(attr.name, dataset.value)
                if isinstance(parameter, hydra_pywr_common.types.base.PywrRecorder): #just in case this is somehow mis-categorised
                    self.recorders[attr.name] = parameter
                else:
                    self.parameters[dataset.name] = parameter

        """ Recorders """
        for attr in self.data["attributes"]:
            resource_scenario = self._get_resource_scenario(attr.id)
            dataset = resource_scenario["dataset"]
            dataset_type = hydra_typemap[dataset.type.upper()]
            if issubclass(dataset_type, PywrRecorder):
                recorder = PywrDataReference.ReferenceFactory(attr.name, dataset.value)
                if isinstance(recorder, hydra_pywr_common.types.base.PywrRecorder): #just in case this is somehow mis-categorised
                    self.recorders[attr.name] = recorder
                else:
                    self.parameters[dataset.name] = recorder

        return ts_inst, meta_inst, tables


    def get_scenario_data(self):

        for resource_attribute in self.data['attributes']:
            attribute = self.attributes[resource_attribute['attr_id']]

            try:
                resource_scenario = self._get_resource_scenario(resource_attribute['id'])
            except ValueError:
                continue
            dataset = resource_scenario['dataset']
            value = dataset['value']

            data_type = dataset['type'].lower()

            if data_type != PywrScenarios.tag.lower():
                continue

            return json.loads(value)
        return None
