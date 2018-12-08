import json
from past.builtins import basestring
from .template import PYWR_EDGE_LINK_NAME, PYWR_CONSTRAINED_EDGE_LINK_NAME
from .core import BasePywrHydra
from hydra_pywr_common import PywrParameter, PywrRecorder, PywrPythonModule
from pywr.nodes import NodeMeta
from hydra_base.lib.HydraTypes.Registry import typemap


class PywrHydraExporter(BasePywrHydra):
    def __init__(self, data, attributes, template):
        super().__init__()
        self.data = data
        self.attributes = attributes
        self.template = template

    @classmethod
    def from_network_id(cls, client, network_id, scenario_id):
        # Fetch the network
        network = client.get_network(network_id, include_data='Y', scenario_ids=[scenario_id])
        # Fetch all the attributes
        attributes = client.get_attributes()
        attributes = {attr.id: attr for attr in attributes}

        # We also need the template to get the node types
        #template = client.get_template_by_name(pywr_template_name())
        return cls(network, attributes, None)

    def get_pywr_data(self):

        pywr_data = {
            'metadata': {'title': self.data['name'], 'description': self.data['description']}
        }

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

        # Export any modules
        modules = list(self.generate_pywr_modules())

        return pywr_data, modules

    def _get_resource_scenario(self, resource_attribute_id):

        # TODO this just returns the first resource scenario that is found.
        for scenario in self.data['scenarios']:
            for resource_scenario in scenario['resourcescenarios']:
                if resource_scenario['resource_attr_id'] == resource_attribute_id:
                    return resource_scenario

        raise ValueError('No resource scenario found for resource attribute id: {}'.format(resource_attribute_id))

    def _get_node(self, node_id):

        for node in self.data['nodes']:
            if node['id'] == node_id:
                return node

        raise ValueError('No node found with node_id: {}'.format(node_id))

    def generate_pywr_nodes(self):
        """ Generator returning a Pywr dict for each node in the network. """

        for node in self.data['nodes']:
            # Create the basic information.
            pywr_node = {'name': node['name']}

            if node.get('description', None) is not None:
                pywr_node['comment'] = node['description']

            # Get the type for this node from the template
            pywr_node_type = None
            for node_type in node['types']:
                pywr_node_type = node_type['name']
            if pywr_node_type is None:
                raise ValueError('Template does not contain node of type "{}".'.format(pywr_node_type))

            pywr_node_attrs, parameters, recorders = self._generate_component_attributes(node, pywr_node_type)
            pywr_node.update(pywr_node_attrs)

            if node['x'] is not None and node['y'] is not None:
                # Finally add coordinates from hydra
                if 'position' not in pywr_node:
                    pywr_node['position'] = {}
                pywr_node['position'].update({'geographic': [node['x'], node['y']]})

            yield pywr_node, parameters, recorders

    def generate_pywr_edges(self):
        """ Generator returning a Pywr tuple for each link/edge in the network. """

        # Only make "real" edges in the Pywr model using the main link type with name PYWR_EDGE_LINK_NAME.
        # Other link types are for virtual or data connections and should not be added to the list of Pywr edges.
        for link in self.data['links']:
            for link_type in link['types']:
                if link_type['name'] in (PYWR_EDGE_LINK_NAME, PYWR_CONSTRAINED_EDGE_LINK_NAME):
                    break
            else:
                continue  # Skip this link type

            if link_type['name'] == PYWR_EDGE_LINK_NAME:
                node_from = self._get_node(link['node_1_id'])
                node_to = self._get_node(link['node_2_id'])
                yield [node_from['name'], node_to['name']], (None, {}, {})

            elif link_type['name'] == PYWR_CONSTRAINED_EDGE_LINK_NAME:
                node_from = self._get_node(link['node_1_id'])
                node_to = self._get_node(link['node_2_id'])

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

            if group_name == 'parameters':
                if data_type != PywrParameter.tag:
                    continue
            elif group_name == 'recorders':
                if data_type != PywrRecorder.tag:
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
                value = int(value)

            yield attribute_name, value

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

            dataset = resource_scenario['dataset']
            dataset_type = dataset['type']
            value = dataset['value']

            if attribute_name in schema.fields:
                # The attribute is part of the node definition
                if isinstance(value, basestring):
                    try:
                        pywr_node[attribute_name] = json.loads(value)
                    except json.decoder.JSONDecodeError:
                        pywr_node[attribute_name] = value
                else:
                    pywr_node[attribute_name] = value
            else:
                # Otherwise the attribute is either a parameter or recorder
                # defined as a node attribute (for convenience).
                hydra_type = typemap[dataset_type]
                component_name = self.make_node_attribute_component_name(component['name'], attribute_name)
                if issubclass(hydra_type, PywrParameter):
                    # Must be a parameter
                    parameters[component_name] = json.loads(value)
                elif issubclass(hydra_type, PywrRecorder):
                    # Must be a recorder
                    recorders[component_name] = json.loads(value)
                else:
                    # Any other type we do not support as a non-schema nodal attribute
                    raise ValueError('Hydra dataset type "{}" not supported as a non-schema'
                                     ' attribute on a Pywr node.'.format(dataset_type))

        return pywr_node, parameters, recorders

    def generate_pywr_modules(self):
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

            print(dataset)
            import pdb; pdb.set_trace()

            if data_type != PywrPythonModule.tag:
                continue

            value = json.loads(value)

            yield attribute_name, value


