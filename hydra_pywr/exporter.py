import json
from past.builtins import basestring
from .template import PYWR_SPLIT_LINK_TYPES, PYWR_EDGE_LINK_NAME, PYWR_CONSTRAINED_EDGE_LINK_NAME
from .core import BasePywrHydra
from hydra_pywr_common import PywrParameter, PywrRecorder, PywrParameterPattern, PywrParameterPatternReference,\
    PywrNodeOutput, PywrScenarios, PywrScenarioCombinations
from pywr.nodes import NodeMeta
from hydra_base.lib.HydraTypes.Registry import typemap
import jinja2
from collections import defaultdict
from .rules import exec_rules

from hydra_pywr_common.types.nodes import(
    PywrNode
)
from hydra_pywr_common.types.base import(
    PywrParameter as DevPywrParameter,
    PywrEdge
)

from hydra_pywr_common.types.fragments.network import(
    Timestepper,
    Metadata
)

from hydra_pywr_common.types.parameters import *
from hydra_pywr_common.types.recorders import *

import pudb


import logging
log = logging.getLogger(__name__)

COST_ALIASES = ['allocation penalty', 'allocation_penalty', 'Allocation Penalty']
EXCLUDE_HYDRA_ATTRS = ("id", "status", "cr_date", "network_id", "x", "y",
                       "types", "attributes", "layout", "network", "description")

class PatternContext(object):
    """ Container for arbitrary attributes in pattern rendering. """
    pass


class PywrHydraExporter(BasePywrHydra):
    def __init__(self, client, data, attributes, template):
        super().__init__()
        self.data = data
        self.attributes = attributes
        self.client = client
        self.template = template

        self.type_id_map = {}
        for tt in self.template.templatetypes:
            self.type_id_map[tt.id] = tt

        self.attr_unit_map = {}
        #Lookup of ID to hydra node
        self.hydra_node_lookup = {}

        self._parameter_recorder_flags = {}
        self._inline_parameter_recorder_flags = defaultdict(dict)
        self._node_recorder_flags = {}

        self.nodes = {}
        self.edges = {}
        self.parameters = {}
        self.recorders = {}

        self._pattern_templates = None


    @classmethod
    def from_scenario_id(cls, client, scenario_id, template_id=None, **kwargs):
        scenario = client.get_scenario(scenario_id, include_data=True, include_results=False, include_metadata=False, include_attr=False)
        # Fetch the network
        network = client.get_network(scenario.network_id, include_data=False, include_results=False, template_id=template_id)

        network.scenarios = [scenario]

        # Fetch all the attributes
        attributes = client.get_attributes()
        attributes = {attr.id: attr for attr in attributes}


        rules = client.get_resource_rules('NETWORK', scenario.network_id)

        network.rules = rules

        template = None

        if template_id is not None:
            template = client.get_template(template_id)
        elif len(network.types) == 1:
            template = client.get_template(network.types[0].template_id)


        # We also need the template to get the node types
        #template = client.get_template_by_name(pywr_template_name())
        return cls(client, network, attributes, template, **kwargs)

    def make_attr_unit_map(self):
        """
            Create a mapping between an attribute ID and its unit, as defined
            in the template
        """
        for templatetype in self.template.templatetypes:
            for typeattr in templatetype.typeattrs:
                self.attr_unit_map[typeattr.attr_id] = typeattr.unit_id

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

    def get_pywr_data(self):

        pywr_data = {
            'metadata': {'title': self.data['name'], 'description': self.data['description']}
        }

        # First find any patterns and create jinja2 templates for them.
        self.create_parameter_pattern_templates()

        """
        pgroup_data = {}
        for pkey, pvalue in self.generate_group_data("parameters", decode_from_json=True):
            print(pkey, pvalue)
            pgroup_data[pkey] = pvalue

        print(pgroup_data)
        exit(77)
        """

        # TODO see proposed changes to metadata and timestepper data.
        for group_name in ('metadata', 'timestepper', 'recorders', 'parameters'):
            # Recorders and parameters are JSON encoded.
            decode_from_json = group_name in ('recorders', 'parameters')

            group_data = {}
            for key, value in self.generate_group_data(group_name, decode_from_json=decode_from_json):
                group_data[key] = value

            # Only make the section if it contains data.
            if len(group_data) > 0:
                if group_name in pywr_data:
                    pywr_data[group_name].update(group_data)
                else:
                    pywr_data[group_name] = group_data

        scenarios = self.get_scenario_data()
        if scenarios is not None:
            pywr_data['scenarios'] = scenarios['scenarios']

        #this is executed here to allow the generate_pywr_nodes access to node
        #schema definitions.
        self.exec_rules()

        scenario_combinations = self.get_scenario_combinations_data()
        if scenario_combinations is not None:
            pywr_data['scenario_combinations'] = scenario_combinations['scenario_combinations']

        nodes = []
        for node, parameters, recorders in self.generate_pywr_nodes():
            nodes.append(node)

            if len(parameters) > 0:
                if 'parameters' not in pywr_data:
                    pywr_data['parameters'] = {}
                pywr_data['parameters'].update(parameters)

            if len(recorders) > 0:
                if 'recorders' not in pywr_data:
                    pywr_data['recorders'] = {}
                pywr_data['recorders'].update(recorders)
        pywr_data['nodes'] = nodes

        self.edges = self.build_edges()

        edges = []
        for edge, (node, parameters, recorders) in self.generate_pywr_edges():
            edges.append(edge)
            if node is not None:
                pywr_data['nodes'].append(node)

                if len(parameters) > 0:
                    if 'parameters' not in pywr_data:
                        pywr_data['parameters'] = {}
                    pywr_data['parameters'].update(parameters)

                if len(recorders) > 0:
                    if 'recorders' not in pywr_data:
                        pywr_data['recorders'] = {}
                    pywr_data['recorders'].update(recorders)

        pywr_data['edges'] = edges

        self.timestepper, self.metadata = self.build_network_attrs()
        return self
        #return pywr_data

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

    def exec_rules(self):

        rules = [r for r in self.data['rules'] if r.status.lower() == 'a']

        log.info("Exec-ing {} rules".format(len(rules)))

        exec_rules(rules)

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
                pywr_node_type = self.type_id_map[node_type['id']]['name']
            if pywr_node_type is None:
                raise ValueError('Template does not contain node of type "{}".'.format(pywr_node_type))

            # ***
            self.build_node_and_references(node, pywr_node_type)
            #print(DevPywrParameter.parameter_type_map)
            #import pudb; pudb.set_trace()
            #self.build_parameters()
            #exit(55)
            # ***

            pywr_node_attrs, parameters, recorders = self._generate_component_attributes(node, pywr_node_type)
            pywr_node.update(pywr_node_attrs)

            if node['x'] is not None and node['y'] is not None:
                # Finally add coordinates from hydra
                if 'position' not in pywr_node:
                    pywr_node['position'] = {}
                pywr_node['position'].update({'geographic': [node['x'], node['y']]})

            yield pywr_node, parameters, recorders


    def build_node_and_references(self, nodedata, pywr_node_type):
        pywr_node = {'type': pywr_node_type}
        parameters = {}
        recorders = {}

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
            #print(attribute)
            #print(value)
            try:
                #print(f"value: {value} ({type(value)}")
                typedval = json.loads(value)
            except json.decoder.JSONDecodeError as e:
                #print(f"*** JSON PARSE ERROR *** {nodedata['name']}:{attribute_name} = {value} ({dataset_type})")
                #print(e)
                typedval = value
            nodedata[attribute_name] = typedval

        nodedata["type"] = pywr_node_type
        node_attr_data = {a:v for a,v in nodedata.items() if a not in EXCLUDE_HYDRA_ATTRS}
        position = { "geographic": [ nodedata.get("x",0), nodedata.get("y",0) ] }
        node_attr_data["position"] = position
        node_attr_data["comment"] = nodedata.get("description", "")

        dev_node = PywrNode.NodeFactory(node_attr_data)

        if dev_node.name == "Delta_Cotton":
            print(dev_node.parameters)
            print(dev_node.max_flow.__dict__)
            #exit(55)
            #pudb.set_trace()
            #pass
        if dev_node.name == "link_48":
            print(dev_node)
            print(dev_node.recorders)
            print(dev_node.__dict__)
            print(node_attr_data)
        if dev_node.name == "YattaThika3A":
            print(dev_node)
            print(dev_node.recorders)
            print(dev_node.__dict__)
            print(node_attr_data)
            print(nodedata)

        print(dev_node)
        print(dev_node.__dict__)
        print(dev_node.parameters)
        print(dev_node.recorders)

        self.nodes[dev_node.name] = dev_node
        self.parameters.update(dev_node.parameters)
        self.recorders.update(dev_node.recorders)

        print()


    def build_edges(self):
        edges = {}

        for hydra_edge in self.data["links"]:
            src_hydra_node = self.hydra_node_lookup[hydra_edge["node_1_id"]]
            dest_hydra_node = self.hydra_node_lookup[hydra_edge["node_2_id"]]
            # Retrieve nodes from PywrNode store to verify presence
            src_node = self.nodes[src_hydra_node["name"]]
            dest_node = self.nodes[dest_hydra_node["name"]]

            edge = PywrEdge((src_node.name, dest_node.name))  # NB Call ctor directly with tuple here, no factory
            edges[edge.name] = edge

            print(edge)
            print(edge.__dict__)
            print()

        return edges


    def build_network_attrs(self):
        """ TimeStepper and Metadata instances """

        print(self.data.keys())

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

            """
            print(resource_scenario)
            print(attr)
            print(f"{attr.name}: {dataset['value']}")
            """
            print()

        print(timestep)
        ts_inst = Timestepper(timestep)
        print(ts_inst)
        print(ts_inst.__dict__)
        print(ts_inst.start, ts_inst.end, ts_inst.timestep)

        """ Metadata """
        metadata = {"title": self.data['name'],
                    "description": self.data['description']
                   }
        for attr_name in self.data["attributes"]:
            attr_group, *subs = attr.name.split('.')
            if attr_group != "metadata":
                continue
            resource_scenario = self._get_resource_scenario(attr.id)
            dataset = resource_scenario["dataset"]
            meta_key = subs[-1]
            metadata[meta_key] = dataset["value"]

        meta_inst = Metadata(metadata)
        print(meta_inst)
        print(meta_inst.__dict__)
        print(meta_inst.title, meta_inst.description)

        return ts_inst, meta_inst


    def generate_pywr_edges(self):
        """ Generator returning a Pywr tuple for each link/edge in the network. """

        # Only make "real" edges in the Pywr model using the main link type with name PYWR_EDGE_LINK_NAME.
        # Other link types are for virtual or data connections and should not be added to the list of Pywr edges.
        for link in self.data['links']:
            for link_type in link['types']:
                link_type_name = self.type_id_map[link_type['id']]['name']
                if link_type_name in (PYWR_EDGE_LINK_NAME, PYWR_CONSTRAINED_EDGE_LINK_NAME):
                    break
            else:
                continue  # Skip this link type

            node_from = self.hydra_node_lookup[link['node_1_id']]
            node_to = self.hydra_node_lookup[link['node_2_id']]

            from_node_types = self.get_type_map(node_from)

            node_type_names = set([nt.lower() for nt in from_node_types.values()])

            if link_type_name == PYWR_EDGE_LINK_NAME:
                #if the node type is a split link, then add the slot name to the link
                #The target node name is used as the slot reference.

                if len(set(PYWR_SPLIT_LINK_TYPES).intersection(node_type_names)) > 0:
                    yield [node_from['name'], node_to['name'], node_to['name'], None], (None, {}, {})
                else:
                    yield [node_from['name'], node_to['name']], (None, {}, {})

            elif link_type_name == PYWR_CONSTRAINED_EDGE_LINK_NAME:
                pywr_node_type = 'link'
                pywr_node = {'name': link['name']}

                pywr_node_attrs, parameters, recorders = self._generate_component_attributes(link, pywr_node_type)
                pywr_node.update(pywr_node_attrs)

                # Yield the two edges and one corresponding node
                yield [node_from['name'], pywr_node['name']], (pywr_node, parameters, recorders)
                yield [pywr_node['name'], node_to['name']], (None, {}, {})

    def generate_group_data(self, group_name, decode_from_json=False):
        """ Generator returning a key and dict value for meta keys. """

        for resource_attribute in self.data['attributes']:

            attribute = self.attributes[resource_attribute['attr_id']]
            attribute_name = attribute['name']

            try:
                resource_scenario = self._get_resource_scenario(resource_attribute['id'])
            except ValueError:
                continue
            dataset = resource_scenario['dataset']
            value = dataset['value']

            data_type = dataset['type']
            hydra_type = typemap[data_type.upper()]

            if group_name == 'parameters':
                if not issubclass(hydra_type, PywrParameter):
                    continue
            elif group_name == 'recorders':
                if not issubclass(hydra_type, PywrRecorder):
                    continue
            else:
                if not attribute_name.startswith('{}.'.format(group_name)):
                    continue
                attribute_name = attribute_name.split('.', 1)[-1]

            if decode_from_json:
                value = json.loads(value)

            # TODO check this. It should not happen as described below.
            # Hydra opportunistically converts everything to native types
            # Some of the Pywr data should remain as string despite looking like a float/int
            if attribute_name == 'timestep' and group_name == 'timestepper':
                try:
                    value = int(value)
                except ValueError:
                    pass

            yield attribute_name, value

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

    def get_scenario_combinations_data(self):

        for resource_attribute in self.data['attributes']:
            attribute = self.attributes[resource_attribute['attr_id']]

            try:
                resource_scenario = self._get_resource_scenario(resource_attribute['id'])
            except ValueError:
                continue
            dataset = resource_scenario['dataset']
            value = dataset['value']

            data_type = dataset['type'].lower()

            if data_type != PywrScenarioCombinations.tag.lower():
                continue

            return json.loads(value)
        return None

    def _generate_component_attributes(self, component, pywr_node_type):

        node_klass = NodeMeta.node_registry[pywr_node_type]
        schema = node_klass.Schema()

        pywr_node = {'type': pywr_node_type}
        parameters = {}
        recorders = {}

        # Then add any corresponding attributes / data
        for resource_attribute in component['attributes']:
            attribute = self.attributes[resource_attribute['attr_id']]
            try:
                resource_scenario = self._get_resource_scenario(resource_attribute['id'])
            except ValueError:
                continue  # No data associated with this attribute.

            if resource_attribute['attr_is_var'] == 'Y':
                continue

            attribute_name = attribute['name']

            if attribute_name in COST_ALIASES:
                attribute_name = 'cost'

            dataset = resource_scenario['dataset']
            dataset_type = dataset['type']
            value = dataset['value']

            hydra_type = typemap[dataset_type.upper()]


            if attribute_name in schema.fields:
                #TODO: This is repeated. fix.
                if issubclass(hydra_type, PywrParameterPatternReference):
                    # Is a pattern of parameters
                    context = self._make_component_pattern_context(component, pywr_node_type)
                    parameters.update(self.generate_parameters_from_patterns(value, context))
                elif issubclass(hydra_type, PywrParameter):
                    component_name = self.make_node_attribute_component_name(
                        component['name'],
                        attribute_name
                    )

                    # Must be a parameter
                    param_value = json.loads(value)
                    try:
                        recorder_flags = param_value.pop('__recorder__')
                    except (KeyError, AttributeError):
                        pass
                    else:
                        self._parameter_recorder_flags[component_name] = recorder_flags

                    parameters[component_name] = param_value

                    value = component_name




                # The attribute is part of the node definition
                if isinstance(value, basestring):
                    try:
                        value = json.loads(value)
                    except json.decoder.JSONDecodeError:
                        pass
                    else:
                        # Check for any recorder flags "__recorder__"
                        try:
                            recorder_flags = value.pop('__recorder__')
                        except (KeyError, AttributeError, TypeError):
                            pass
                        else:
                            self._inline_parameter_recorder_flags[component['name']][attribute_name] = recorder_flags
                    finally:
                        pywr_node[attribute_name] = value

                else:
                    pywr_node[attribute_name] = value
            else:
                # Otherwise the attribute is either a parameter or recorder
                # defined as a node attribute (for convenience).
                component_name = self.make_node_attribute_component_name(
                    component['name'],
                    attribute_name
                )
                if issubclass(hydra_type, PywrNodeOutput):
                    value = json.loads(value)
                    try:
                        recorder_flags = value.pop('__recorder__')
                    except (KeyError, AttributeError):
                        pass
                    else:
                        self._node_recorder_flags[component['name']] = recorder_flags
                elif issubclass(hydra_type, PywrParameterPatternReference):
                    # Is a pattern of parameters
                    context = self._make_component_pattern_context(component, pywr_node_type)
                    parameters.update(self.generate_parameters_from_patterns(value, context))
                elif issubclass(hydra_type, PywrParameter):
                    # Must be a parameter
                    value = json.loads(value)
                    try:
                        recorder_flags = value.pop('__recorder__')
                    except (KeyError, AttributeError):
                        pass
                    else:
                        self._parameter_recorder_flags[component_name] = recorder_flags
                    parameters[component_name] = value
                elif issubclass(hydra_type, PywrRecorder):
                    # Must be a recorder
                    recorders[component_name] = json.loads(value)
                else:
                    pass
                    # Any other type we do not support as a non-schema nodal attribute
                    # raise ValueError('Hydra dataset type "{}" not supported as a non-schema'
                    #                 ' attribute on a Pywr node.'.format(dataset_type))

        return pywr_node, parameters, recorders

    def create_parameter_pattern_templates(self):
        """ Create Jinja2 templates for each parameter pattern. """

        templates = {}

        for resource_attribute in self.data['attributes']:

            attribute = self.attributes[resource_attribute['attr_id']]
            attribute_name = attribute['name']

            try:
                resource_scenario = self._get_resource_scenario(resource_attribute['id'])
            except ValueError:
                continue
            dataset = resource_scenario['dataset']
            value = dataset['value']

            data_type = dataset['type']

            if data_type.upper() != PywrParameterPattern.tag:
                continue

            pattern_template = jinja2.Template(value)
            templates[attribute_name] = pattern_template

        self._pattern_templates = templates

    def _make_component_pattern_context(self, component, pywr_node_type):
        """ Create the context for rendering parameter patterns. """

        node_klass = NodeMeta.node_registry[pywr_node_type]
        schema = node_klass.Schema()

        context = PatternContext()
        context.name = component['name']
        context.id = component['id']
        context.description = component['description']

        data = PatternContext()

        for resource_attribute in component['attributes']:
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

            hydra_type = typemap[dataset_type.upper()]
            if issubclass(hydra_type, (PywrParameter, PywrRecorder)):
                # Ignore Pywr parameter definitions
                continue

            if isinstance(value, basestring):
                try:
                    value = json.loads(value)
                except json.decoder.JSONDecodeError:
                    pass

            setattr(data, attribute_name, value)
        context.data = data
        return context

    def generate_parameters_from_patterns(self, pattern_name, context):

        template = self._pattern_templates[pattern_name]
        # TODO make this work for non-node types
        data = template.render(node=context)
        parameters = json.loads(data)
        return parameters
