import json
import os
import logging

from collections import defaultdict

from hydra_base.lib.HydraTypes.Registry import typemap as hydra_typemap
from hydra_base.lib.objects import JSONObject

from hydra_pywr.template import PYWR_SPLIT_LINK_TYPES

from hydra_pywr_common.datatypes import PywrParameter, PywrRecorder, PywrTable

from hydra_pywr_common.types import PywrDataReference

from hydra_pywr.rules import exec_rules

import hydra_pywr_common

from hydra_pywr_common.types.base import(
    PywrNode, PywrEdge
)

from hydra_pywr_common.types.fragments.network import(
    Timestepper,
    Metadata,
    Scenario
)

from .core import BasePywrHydra

from hydra_pywr_common.types.fragments.config import IntegratedConfig
#from hydra_pywr_common.types.parameters import *
#from hydra_pywr_common.types.recorders import *


COST_ALIASES = ['allocation penalty', 'allocation_penalty', 'Allocation Penalty']
EXCLUDE_HYDRA_ATTRS = ("id", "status", "cr_date", "network_id", "x", "y",
                       "types", "attributes", "layout", "network", "description")


class PywrHydraExporter(BasePywrHydra):
    def __init__(self, client, data, scenario_id, attributes, template, **kwargs):

        super().__init__()

        self.data = data
        self.network_id = data['id']
        self.scenario_id = scenario_id
        self.attributes = attributes
        self.client = client
        self.template = template

        self.log = logging.getLogger(__name__)

        self.type_id_map = {}
        for tt in self.template.templatetypes:
            self.type_id_map[tt.id] = tt

        self.attr_unit_map = {}
        self.hydra_node_lookup = {}

        self._parameter_recorder_flags = {}
        self._inline_parameter_recorder_flags = defaultdict(dict)
        self._node_recorder_flags = {}

        self.partial = kwargs.get('partial', False)

        self.timestepper = None
        self.metadata = None

        self.nodes = {}
        self.edges = []
        self.parameters = {}
        self.recorders = {}
        self.tables = {}
        self.scenarios = []

        self._pattern_templates = None

        self.reference_model = None

        self.excluded_node_ids = []
        self.excluded_nodes = []
        # parameters which are used by the sub-netwwork in partial mode.
        # Any paramteers not in this list are deleted.
        self.parameters_to_keep = []

        #this is a list of parameters which are referenced from nodes that will
        #be removed in partial mode. Some may be kept if they are refrenced from nodes which are to
        #be kept
        self.parameters_to_remove = []

        self.keep_color = kwargs.get('keep_color', [])
        if self.keep_color is not None:
            if isinstance(self.keep_color, str):
                if self.keep_color.find(',') > 0: # is it comma separated values?
                    self.keep_color = self.keep_color.split(',')
                else:
                    # just one color. Then make a list
                    self.keep_color = [self.keep_color]

        self.paramkeys = ('params', 'parameters', 'index_parameter')


    @classmethod
    def from_scenario_id(cls, client, scenario_id, template_id=None, index=0, **kwargs):
        cache_file = f'/tmp/scenario_{scenario_id}.json'
        if kwargs.get('use_cache') is True and os.path.exists(cache_file):
            logging.info("Using cached scenario")
            with open(cache_file, 'r') as f:
                scenario = JSONObject(json.load(f))
        else:
            scenario = client.get_scenario(scenario_id=scenario_id, include_data=True, include_results=False, include_metadata=False, include_attr=False)
            with open(cache_file, 'w') as f:
                json.dump(scenario, f)
        # Fetch the network
        network_cache_file = f'/tmp/network_{scenario.network_id}.json'
        if kwargs.get('use_cache') is True and os.path.exists(network_cache_file):
            logging.info("Using cached network")
            with open(network_cache_file, 'r') as f:
                network = JSONObject(json.load(f))
        else:
            network = client.get_network(
                network_id=scenario.network_id,
                include_data=False,
                include_results=False,
                template_id=template_id,
                include_attributes=True)

            with open(network_cache_file, 'w') as f:
                json.dump(JSONObject(network), f)

        network.scenarios = [scenario]


        rules = client.get_resource_rules(ref_key='NETWORK', ref_id=scenario.network_id, scenario_id=scenario.id)

        network.rules = rules

        # Fetch all the attributes
        attributes = client.get_attributes(
        network_id=network.id,
        project_id = network.project_id,
        include_global=True
        )
        attributes = {attr.id: attr for attr in attributes}

        template = None

        if template_id is None:
            template_id = network.types[index].template_id

        template_cache_file = f'/tmp/template_{template_id}.json'
        if kwargs.get('use_cache') is True and os.path.exists(template_cache_file):
            logging.info("Using cached template")

            with open(template_cache_file, 'r') as f:
                template = JSONObject(json.load(f))
        else:
            template = client.get_template(template_id=template_id)
            with open(template_cache_file, 'w') as f:
                json.dump(template, f)

        return cls(client, network, scenario_id, attributes, template, **kwargs)


    def get_pywr_data(self, domain=None, reference_model=None):
        if reference_model is not None:
            self.reference_model = json.load(reference_model)


        if domain is not None:
            self.timestepper, self.metadata, self.scenarios = self.build_integrated_network_attrs(domain)
        else:
            self.timestepper, self.metadata, self.scenarios = self.build_network_attrs()

        self.generate_pywr_nodes()
        self.edges = self.build_edges()

        self.remove_unused_parameters()

        self.exec_rules()

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

    def _get_resource_scenario(self, resource_attribute, node=None):
        # TODO this just returns the first resource scenario that is found.
        for scenario in self.data['scenarios']:
            for resource_scenario in scenario['resourcescenarios']:
                if resource_scenario['resource_attr_id'] == resource_attribute['id']:
                    return resource_scenario
        name = ""
        nodename = ""
        if node is not None:
            nodename = node.name
        if hasattr(resource_attribute, 'name'):
            name = resource_attribute.name
        elif isinstance(resource_attribute, dict) and resource_attribute.get('name') is not None:
            name = resource_attribute['name']
        raise ValueError('No resource scenario found on {} {} for resource attribute: "{}" ({})'.format(
            resource_attribute['ref_key'], nodename, name, resource_attribute['id']))
        return None

    def get_reference_node_names(self):
        if self.reference_model is not None:
            return [n['name'].lower() for n in self.reference_model['nodes']]
        return []

    def get_link_lookup(self):
        """
            compile a dict, keyed on node ID of all the links attached
            to that node.
        """
        lookup = defaultdict(list)
        for l in self.data['links']:
            lookup[l['node_1_id']].append(l)
            lookup[l['node_2_id']].append(l)
        return lookup

    def exec_rules(self):

        rules = [r for r in self.data['rules'] if r.status.lower() == 'a']

        self.log.info("Exec-ing {} rules".format(len(rules)))

        exec_rules(rules)

    def generate_pywr_nodes(self):
        """ Generator returning a Pywr dict for each node in the network. """
        #only export node names which match the reference model
        reference_node_names = self.get_reference_node_names()

        #a dict mapping the links attached to each node.
        link_lookup = self.get_link_lookup()

        for node in self.data['nodes']:

            #this might occur if the node was excluded by virtue of being related
            #to another node which is being excluded
            if node['id'] in self.excluded_node_ids:
                continue

            if self.reference_model is not None:
                if node['name'].lower() not in reference_node_names:
                    continue
            # Create the basic information.
            pywr_node = {'name': node['name']}

            #if any of the links in or out are red, then keep this node.
            keep_node = False
            if link_lookup.get(node['id']) is None:
                #this is an unconnected node, so we need to keep it.
                keep_node = True
            else:
                for l in link_lookup.get(node['id'], []):
                    if l.layout is not None and l.layout.get('color') is not None:
                        if self.keep_color  == []:#no specific colour specified
                            keep_node = True
                        elif l.layout['color'] in self.keep_color:
                            #colour(s) specified, and this link matches.
                            keep_node = True

            #only keep nodes which have a link connected to them which has a layout
            if self.partial is True and keep_node is False:
                self.log.info("Excluding node %s", node['name'])
                self.excluded_node_ids.append(node['id'])
                self.excluded_nodes.append(node)


                #identify parameters on this node which should be excluded as they're no longer referenced

                for resource_attribute in node['attributes']:
                    if resource_attribute['attr_is_var'] == 'Y':
                        continue

                    attribute = self.attributes[resource_attribute['attr_id']]
                    try:
                        resource_scenario = self._get_resource_scenario(resource_attribute, node)
                    except ValueError:
                        continue  # No data associated with this attribute.
                    value = resource_scenario['dataset']['value']
                    self.flag_parameter_to_remove(value)

                continue

            self.hydra_node_lookup[node['id']] = node

            if node.get('description', None) is not None:
                pywr_node['comment'] = node['description']


            pywr_node_type = None
            for node_type in node['types']:
                try:
                    pywr_node_type = self.type_id_map[node_type['id']]['name']
                except KeyError:
                    # Skip as not in this template...
                    continue

            node_type_attribute_names = [a.attr.name for a in self.type_id_map[node_type['id']].typeattrs]
            #if pywr_node_type is None:
            #    raise ValueError('Template does not contain node of type "{}".'.format(pywr_node_type))


            # Skip as not in this template...
            if pywr_node_type:
                self.build_node_and_references(node, pywr_node_type, node_type_attribute_names)

        self.remove_orphan_nodes()

    def remove_orphan_nodes(self):
        """
            If a node in partial mode is being removed, then the nodes to which it references
            should also be removed.
        """

        nodes_to_remove = []
        for name, node in self.nodes.items():
            if hasattr(node, 'nodes'):
                #if all of the nodes to which this node references are not in the exported list of nodes
                #then don't export this node as it's redundant.
                for n in node.nodes.get_value():
                    if n in self.nodes:
                        break
                else:
                    nodes_to_remove.append(name)

        #remove all the nodes flagged for removal.
        for node_to_remove in nodes_to_remove:
            if node_to_remove in self.nodes:
                del self.nodes[node_to_remove]

    def flag_parameter_to_remove(self, value):
        """
            If the value passed in is a parameter, flag is as a candidate for removal
        """
        self._flag_parameter(value, self.parameters_to_remove)

    def flag_parameter_to_keep(self, value):
        """
            If the value passed in is a parameter, flag is as one which must remain
        """
        self._flag_parameter(value, self.parameters_to_keep)

    def _flag_parameter(self, value, context):
        """
            Either set a parameter to be removed or to be kept depending on the context
            passed in. The context is either self.parameters_to_remove or self.parameters_to_keep
        """
        if self.partial is not True:
            return
        try:
            #flag any parameters which are referenced by a paramter
            #defined on this node
            if isinstance(value, str):
                typedval = json.loads(value)
            else:
                typedval = value

            if isinstance(typedval, dict):
                for paramkey in self.paramkeys:
                    referencedparams = typedval.get(paramkey, [])
                    if isinstance(referencedparams, list):# eg 'params'
                        for p in typedval.get(paramkey, []):
                            context.append(p)
                            self.flag_related_parameters(p, context)
                    elif isinstance(referencedparams, str): # eg 'index_parameter'
                        context.append(referencedparams)
                        self.flag_related_parameters(p, context)
            elif isinstance(typedval, str):
                if self.parameters.get(typedval) is not None:
                    context.append(typedval)
        except json.decoder.JSONDecodeError as e:
            #flag a parameter defined on this node as being
            #primed for removal
            typedval = value
            if self.parameters.get(typedval) is not None:
                context.append(typedval)
                self.flag_related_parameters(typedval, context)

    def flag_related_parameters(self, param_name, context):
        """
            Take a given parameter name and flag the paramters related to this one
            to be either removed or kept (depending on the context).
            The context is either self.parameters_to_remove or self.parameters_to_keep
        """

        if not isinstance(param_name, dict):#
            #this paramter name is not a parameter reference, but an embedded parameter
            #so ignore it
            return

        if self.parameters.get(param_name) is None:
            return

        param = self.parameters[param_name].get_value()
        for paramkey in self.paramkeys:
            referencedparams = param.get(paramkey, [])
            if isinstance(referencedparams, list):# eg 'params'
                 for p in referencedparams:
                    context.append(p)
                    self.flag_related_parameters(p, context)
            elif isinstance(referencedparams, str): # eg 'index_parameter'
                context.append(referencedparams)
                self.flag_related_parameters(p, context)

    def flag_orphan_parameters(self):
        """
            Reove parameters which refer to nodes that aren't there an ymore, but which
            aren't referenced from the node, but instead reference the node in the
            parameter using a key like 'storage_node' or 'node'
        """
        for pname in self.parameters:
            p = self.parameters[pname].get_value()
            for noderef in ['storage_node', 'node']:
                if p.get(noderef) is not None:
                    if p[noderef] not in self.nodes:
                        self.parameters_to_remove.append(pname)

    def remove_unused_parameters(self):
        """
        Remove any parameters which are referenced from nodes which are not included in the export
        """

        if self.partial is not True:
            return

        #remove parameters which refer to a node, but the node desn't refer to them
        self.flag_orphan_parameters()

        #remove any parameters from the 'remove' list which are present in the 'keep' list.
        #this means they are referenced on nodes both inside and outside the partial network
        #and therefore should be kept
        self.parameters_to_remove = set(self.parameters_to_remove) - set(self.parameters_to_keep)

        for unused_parameter in self.parameters_to_remove:
            if self.parameters.get(unused_parameter) is not None:# maybe it was already removed?
                self.log.warning(unused_parameter)
                del(self.parameters[unused_parameter])

        self.log.info("%s parameters removed", len(self.parameters_to_remove))


    def build_node_and_references(self, nodedata, pywr_node_type, node_type_attribute_names):

        for resource_attribute in nodedata['attributes']:
            attribute = self.attributes[resource_attribute['attr_id']]
            try:
                resource_scenario = self._get_resource_scenario(resource_attribute, nodedata)
            except ValueError:
                continue  # No data associated with this attribute.

            if resource_attribute['attr_is_var'] == 'Y':
                continue

            attribute_name = attribute['name']
            dataset = resource_scenario['dataset']
            value = dataset['value']

            try:
                typedval = json.loads(value)
                if isinstance(typedval, dict) and typedval.get('__recorder__') is not None:
                    self._parameter_recorder_flags[f"__{nodedata['name']}__:{attribute_name}"] = typedval.pop('__recorder__')
                #remove the contents of the pandas_kwargs sub-dict ane put them on the parent dict
                if isinstance(typedval, dict) and typedval.get('pandas_kwargs') is not None:
                    for k, v in typedval.get('pandas_kwargs').items():
                        typedval[k] = v
                    del(typedval['pandas_kwargs'])

                self.flag_parameter_to_keep(value)

            except json.decoder.JSONDecodeError as e:
                typedval = value
                if self.parameters.get(typedval) is not None:
                    self.parameters_to_keep.append(typedval)

            nodedata[attribute_name] = typedval

        nodedata["type"] = pywr_node_type
        node_attr_data = {a:v for a, v in nodedata.items() if a not in EXCLUDE_HYDRA_ATTRS}
        position = {"geographic": [nodedata.get("x", 0), nodedata.get("y", 0)]}
        node_attr_data["position"] = position
        node_attr_data['intrinsic_attrs'] = node_type_attribute_names

        dev_node = PywrNode.NodeFactory(node_attr_data)

        self.nodes[dev_node.name] = dev_node
        self.parameters.update(dev_node.parameters)
        self.recorders.update(dev_node.recorders)
        #add node-level parameters to the used parameters list so they're not deleted
        self.parameters_to_keep.extend(dev_node.parameters.keys())

    def normalise(self, name):
        return name.lower().replace("_", "").replace(" ", "")

    def build_edges(self):
        edges = {}

        for hydra_edge in self.data["links"]:
            src_id = hydra_edge["node_1_id"]
            dest_id = hydra_edge["node_2_id"]
            if self.reference_model is not None:
                #ignore links to nodes which we are excluding
                src_hydra_node = self.hydra_node_lookup.get(src_id)
                dest_hydra_node = self.hydra_node_lookup.get(dest_id)
                if src_hydra_node is None or dest_hydra_node is None:
                    continue
            else:
                if src_id in self.excluded_node_ids or dest_id in self.excluded_node_ids:
                    continue
                src_hydra_node = self.hydra_node_lookup[src_id]
                dest_hydra_node = self.hydra_node_lookup[dest_id]


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
        resource_scenario = self._get_resource_scenario(domain_attr)
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
            try:
                resource_scenario = self._get_resource_scenario(attr)
            except ValueError:
                self.log.warning("No value for %s", attr.name)
                continue
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
        metadata = {
            "hydra_network_id": self.data['name'],
            "hydra_scenario_id": self.scenario_id,
            "title": self.data['name'],
            "description": self.data['description']
        }
        for attr in self.data["attributes"]:
            attr_group, *subs = attr.name.split('.')
            if attr_group != "metadata":
                continue
            resource_scenario = self._get_resource_scenario(attr)
            dataset = resource_scenario["dataset"]
            meta_key = subs[-1]
            metadata[meta_key] = dataset["value"]

        meta_inst = Metadata(metadata)

        """ Scenarios """
        scenarios = []
        for attr in self.data["attributes"]:
            if attr.name == 'scenarios':
                try:
                    resource_scenario = self._get_resource_scenario(attr)
                except ValueError as e:
                    self.log.warning("No value found for scenario attribute")
                    continue

                try:
                    dataset = resource_scenario["dataset"]
                    val  = dataset['value']
                    scenarios = json.loads(val)['scenarios']
                except ValueError as e:
                    self.log.warning("An error occurred getting data for scenarios")

                continue


        """ Tables """
        for attr in self.data["attributes"]:
            try:
                resource_scenario = self._get_resource_scenario(attr)
            except ValueError as e:
                self.log.debug("No value found for network attribute %s", attr.name)
                continue

            dataset = resource_scenario["dataset"]
            dataset_type = hydra_typemap[dataset.type.upper()]
            #backward compatibility
            if attr.name.startswith('tbl_'):
                tablename = attr.name.replace('tbl_', '')
                for k in ('index_col', 'key', 'url', 'header'):
                    if tablename.endswith(f'.{k}'):
                        tablename = tablename.replace(f'.{k}', '')
                        try:
                            dataset['value'] = int(float(dataset.value))
                        except:
                            try:
                                dataset['value'] = json.loads(dataset.value)
                            except:
                                pass
                        if self.tables.get(tablename):
                            self.tables[tablename][k] = dataset.value
                        else:
                            self.tables[tablename] = {k: dataset.value}
            elif issubclass(dataset_type, PywrTable):
                self.tables[attr.name] = json.loads(dataset.value)

        """ Parameters """
        for attr in self.data["attributes"]:
            try:
                resource_scenario = self._get_resource_scenario(attr)
            except ValueError as e:
                self.log.debug("No value found for network attribute %s", attr.name)
                continue

            #Ignore tables
            if attr.name.startswith('tbl') or attr.name == 'scenarios':
                continue

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
                dataset.value = json.dumps(data)
            except ValueError as e:
                self.log.warning(f"{attr.name} : {e}")

            if is_parameter_or_recorder is True:
                parameter = PywrDataReference.ReferenceFactory(attr.name, dataset.value)
                if isinstance(parameter, hydra_pywr_common.types.base.PywrRecorder): #just in case this is somehow mis-categorised
                    self.recorders[attr.name] = parameter
                else:
                    self.parameters[dataset.name] = parameter

        """ Recorders """
        for attr in self.data["attributes"]:
            try:
                resource_scenario = self._get_resource_scenario(attr)
            except ValueError as e:
                self.log.debug("No value found for network attribute %s", attr.name)
                continue
            dataset = resource_scenario["dataset"]
            dataset_type = hydra_typemap[dataset.type.upper()]
            if issubclass(dataset_type, PywrRecorder):
                recorder = PywrDataReference.ReferenceFactory(attr.name, dataset.value)
                if isinstance(recorder, hydra_pywr_common.types.base.PywrRecorder): #just in case this is somehow mis-categorised
                    self.recorders[attr.name] = recorder
                else:
                    self.parameters[dataset.name] = recorder


        return ts_inst, meta_inst, scenarios


    def get_scenario_data(self):

        for resource_attribute in self.data['attributes']:
            attribute = self.attributes[resource_attribute['attr_id']]

            try:
                resource_scenario = self._get_resource_scenario(resource_attribute)
            except ValueError:
                continue
            dataset = resource_scenario['dataset']
            value = dataset['value']

            data_type = dataset['type'].lower()

            if data_type != PywrScenarios.tag.lower():
                continue

            return json.loads(value)
        return None
