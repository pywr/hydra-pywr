import json
import warnings
from past.builtins import basestring
from .template import PYWR_PROTECTED_NODE_KEYS, pywr_template_name, load_template_config
from .core import BasePywrHydra, data_type_from_field
from pywr.nodes import NodeMeta
from hydra_pywr_common import data_type_from_component_type, data_type_from_parameter_value, PywrParameter
from hydra_base.exceptions import HydraError
import logging
log = logging.getLogger(__name__)


class PywrHydraImporter(BasePywrHydra):

    def __init__(self, client, data, template):
        super().__init__()
        self.template = template
        self.client = client

        if isinstance(data, basestring):
            # argument is a filename
            path = data
            with open(path, "r") as f:
                data = json.load(f)
        elif hasattr(data, 'read'):
            # argument is a file-like object
            data = json.load(data)

        self.data = data
        self.attr_unit_map = {}
        self.dimensions = {}
        #maps the name of an attribute to its dimension (if it has one)
        self.attr_dimension_map = {}
        self.attr_name_map = self.make_attr_name_map()
        self.attribute_name_id_map = {}

        #If any node types in the input file are not available in the current
        #environment (custom pywr nodes) then either log a warning and use default or throw an exception.
        self.ignore_type_errors = False

        self.next_node_id = -1

    @classmethod
    def from_client(cls, client, data, template_id):
        template = client.get_template(template_id)
        return cls(client, data, template)

    @property
    def name(self):
        try:
            name = self.data['metadata']['title']
        except KeyError:
            name = 'A Pywr model.'
            warnings.warn('Pywr model data contains no name metadata. Using default name: "{}"'.format(name))
        return name

    @property
    def description(self):
        try:
            description = self.data['metadata']['description']
        except KeyError:
            description = ''
        return description

    def make_attr_unit_map(self):
        """
            Create a mapping between an attribute ID and its unit, as defined
            in the template
        """

        if self.template is None:
            log.info("Cannot build unit map as no template was specified.")
            return

        for templatetype in self.template.templatetypes:
            for typeattr in templatetype.typeattrs:
                self.attr_unit_map[typeattr.attr_id] = typeattr.unit_id

        log.info("Units map created")

    def import_data(self, project_id, projection=None, ignore_type_errors=False):

        self.ignore_type_errors = ignore_type_errors

        self.make_attr_unit_map()

        # First the attributes must be added.
        attributes = list(self.add_attributes_request_data())

        # The response attributes have ids now.
        response_attributes = self.client.add_attributes(attributes)

        # Convert to a simple dict for local processing.
        self.attribute_name_id_map = {a.name.lower(): a.id for a in response_attributes}

        # Now we try to create the network
        network = self.add_network_request_data(project_id, projection=projection)
        hydra_network = self.client.add_network(network)

        # Get the added scenario_id. There should only be one scenario
        assert len(hydra_network['scenarios']) == 1
        scenario_id = hydra_network['scenarios'][0]['id']

        return hydra_network.id, scenario_id

    def make_attr_name_map(self):
        """
            Create a mapping between an attribute's name and itself, as defined
            in the template
        """
        attr_name_map = {}
        for templatetype in self.template.templatetypes:
            for typeattr in templatetype.typeattrs:
                attr = self.client.get_attribute_by_id(typeattr.attr_id)
                attr_name_map[attr.name] = attr
                # Populate the dimension mapping
                self.attr_dimension_map[attr.name] = attr.dimension_id

        return attr_name_map

    def add_attributes_request_data(self):
        """ Generate the data for adding attributes to Hydra. """


        # Yield attributes from the timestepper ...
        for attr in self.attributes_from_meta():
            yield attr

        # Yield the attributes from the nodes ...
        for attr in self.attributes_from_nodes():
            yield attr

        # ... now the attributes associated with the recorders and parameters.
        for key in ('recorders', 'parameters'):
            if key not in self.data:
                continue
            for attr in self.attributes_from_component_dict(key):
                yield attr

    def add_network_request_data(self, project_id, projection=None):
        """ Return a dictionary of the data required for adding a network to Hydra. """

        # Get the network type
        for template_type in self.template['templatetypes']:
            if template_type['resource_type'] == 'NETWORK':
                network_template_type = template_type
                break
        else:
            raise ValueError('No NETWORK resource type found in template.')

        network_template_type_id = network_template_type['id']

        # TODO add tables and scenarios.

        nodes, links, resource_scenarios = self.convert_nodes_and_edges()

        network_attributes = []
        for component_key in ('recorders', 'parameters'):
            generator = self.generate_component_resource_scenarios(component_key, encode_to_json=True)
            for resource_attribute, resource_scenario in generator:
                network_attributes.append(resource_attribute)
                resource_scenarios.append(resource_scenario)

        # TODO timestepper data is on the scenario.
        for component_key in ('metadata', 'timestepper'):
            generator = self.generate_component_resource_scenarios(component_key, encode_to_json=False)
            for resource_attribute, resource_scenario in generator:
                network_attributes.append(resource_attribute)
                resource_scenarios.append(resource_scenario)

        if 'scenarios' in self.data:
            resource_attribute, resource_scenario = self._make_dataset_resource_attribute_and_scenario('scenarios',
                                                                                                       {'scenarios': self.data['scenarios']},
                                                                                'PYWR_SCENARIOS', self.attribute_name_id_map['scenarios'],
                                                                                encode_to_json=True)
            network_attributes.append(resource_attribute)
            resource_scenarios.append(resource_scenario)

        if 'scenario_combinations' in self.data:
            resource_attribute, resource_scenario = self._make_dataset_resource_attribute_and_scenario('scenario_combinations',
                                                                                                       {'scenario_combinations': self.data['scenario_combinations']},
                                                                                'PYWR_SCENARIO_COMBINATIONS', self.attribute_name_id_map['scenario_combinations'],
                                                                                encode_to_json=True)
            network_attributes.append(resource_attribute)
            resource_scenarios.append(resource_scenario)

        scenario = self.make_scenario(resource_scenarios)

        data = {
            "name": self.name,
            "description": self.description,
            "project_id": project_id,
            "links": links,
            "nodes": nodes,
            "layout": None,
            "scenarios": [scenario, ],
            "projection": projection,
            "attributes": network_attributes,
            'types': [{'id': network_template_type_id}]
        }
        return data

    def make_scenario(self, resource_scenarios=None):
        """ Make the request data for a Hydra scenario. """

        if resource_scenarios is None:
            resource_scenarios = []

        scenario = {
            "name": "Baseline",
            "description": "Baseline scenario (auto-generated by Pywr app)",
            "resourcescenarios": resource_scenarios
        }
        return scenario

    def attributes_from_nodes(self):
        """ Generator to convert Pywr nodes data in to Hydra attribute data.

        This function is intended to be used to convert Pywr components (e.g. recorders, parameters, etc.)  data
        in to a format that can be imported in to Hydra. The Pywr component data is a dict of dict with each
        sub-dict represent a single component (see the "recorder" or "parameters" section of the Pywr JSON format). This
        function returns Hydra data to add a Attribute for each of the components in the outer dict.
        """
        nodes = self.data['nodes']

        attributes = set()


        for node in nodes:
            node_type = node['type'].lower()
            node_klass = NodeMeta.node_registry.get(node_type)
            if node_klass is None:
                msg = f"Node type {node_type} not recognised."
                if self.ignore_type_errors is False:
                    raise HydraError(msg)
                log.warning(msg + "Attempting to add default values.")
                for name in node.keys():
                    if name not in PYWR_PROTECTED_NODE_KEYS:
                        attributes.add(name)
                continue
            schema = node_klass.Schema()

            # Create an attribute for each field in the schema.
            for name, field in schema.fields.items():
                if name in PYWR_PROTECTED_NODE_KEYS:
                    continue
                attributes.add(name)

        for attr in sorted(attributes):
            yield self.attr_name_map.get(attr, {
                'name': attr,
                'description': '',
                'dimension_id' : self.attr_dimension_map.get(attr)
            })

    def attributes_from_meta(self):
        """ Generator to convert Pywr timestepper data in to Hydra attribute data. """
        if 'scenarios' in self.data:
            yield self.attr_name_map.get('scenarios', {'name': 'scenarios', 'description': ''})

        if 'scenario_combinations' in self.data:
            yield self.attr_name_map.get('scenario_combinations', {'name': 'scenario_combinations', 'description': ''})

        for meta_key in ('metadata', 'timestepper'):
            meta_data = self.data.get(meta_key)
            if meta_data is None:
                log.warning("No meta data found for key %s", meta_key)
                continue
            for key in meta_data.keys():
                # Prefix these names with Pywr JSON section.
                attr_name = '{}.{}'.format(meta_key, key)
                yield self.attr_name_map.get(attr_name, {'name': attr_name,'description': ''})

    def _get_template_type_by_name(self, name, resource_type=None):
        for template_type in self.template['templatetypes']:
            if name == template_type['name']:
                if resource_type is None or template_type['resource_type'] == resource_type:
                    return template_type
        msg = 'Template does not contain node of type "{}".'.format(name)
        if self.ignore_type_errors:
            log.warning(msg)
            return {}
        else:
            raise HydraError(msg)


    def convert_nodes_and_edges(self):
        """ Convert a tuple of (nodes, links) of Hydra data based on the given Pywr data. """

        pywr_nodes = self.data['nodes']
        pywr_edges = self.data['edges']

        def find_node_id(node_name):
            for hydra_node in hydra_nodes:
                if hydra_node['name'] == node_name:
                    return hydra_node['id']
            raise ValueError('Node name "{}" not found in node data.'.format(node_name))

        # TODO make this object properties
        node_id = -1
        link_id = -1
        hydra_nodes = []
        hydra_links = []  # Note the change in nomenclature pywr->edges, hydra->links
        hydra_resource_scenarios = []

        # First generate the hydra node data
        for pywr_node in pywr_nodes:

            try:
                comment = pywr_node['comment']
            except KeyError:
                comment = None

            # Get the type for this node from the template
            # Pywr keeps a registry of lower case node types.
            pywr_node_type = pywr_node['type'].lower()
            node_template_type = self._get_template_type_by_name(pywr_node_type, 'NODE')
            node_template_type_id = node_template_type.get('id')

            # Now make the attributes
            resource_attributes = []
            for resource_attribute, resource_scenario in self.generate_node_resource_scenarios(pywr_node):
                resource_attributes.append(resource_attribute)
                hydra_resource_scenarios.append(resource_scenario)

            # Try to get geometry from the pywr_node
            geometry = None
            try:
                geometry = pywr_node['position']['geographic']
            except KeyError:
                pass

            x, y = None, None
            if geometry is not None:
                if isinstance(geometry, list):
                    x, y = geometry
                    geometry = None  # Don't save this a layout
                elif isinstance(geometry, dict):
                    from shapely.geometry import shape
                    rpoint = shape(geometry).representative_point()
                    x = rpoint.x
                    y = rpoint.y
                else:
                    raise ValueError(f'Node "{pywr_node["name"]}" position data not supported.')

            hydra_node = {
                'id': node_id,
                'name': pywr_node['name'],
                'description': comment,
                'layout': {
                    'geojson': geometry
                },
                'x': x,  # TODO add some tests with coordinates.
                'y': y,
                'attributes': resource_attributes,
                'types': [{'id': node_template_type_id}] if node_template_type_id else None
            }

            hydra_nodes.append(hydra_node)
            node_id -= 1

        # All Pywr edges have the same type
        edge_template_type = self._get_template_type_by_name('edge', 'LINK')
        edge_template_type_id = edge_template_type.get('id')

        for pywr_edge in pywr_edges:

            # TODO slots
            if len(pywr_edge) > 2:
                log.warning('Edges with slot definitions are not currently supported. %s', pywr_edge)
                pywr_edge = pywr_edge[0:2]

            node_1_name, node_2_name = pywr_edge

            hydra_link = {
                'id': link_id,
                'name': "{} to {}".format(node_1_name, node_2_name),
                'description': None,
                'layout': None,
                'node_1_id': find_node_id(node_1_name),
                'node_2_id': find_node_id(node_2_name),
                'attributes': [],  # Links have no resource attributes
                'types': [{'id': edge_template_type_id}] if edge_template_type_id else None
            }
            hydra_links.append(hydra_link)
            link_id -= 1

        return hydra_nodes, hydra_links, hydra_resource_scenarios

    def generate_node_resource_scenarios(self, pywr_node):

        for ra, rs in self.generate_node_schema_resource_scenarios(pywr_node):
            yield ra, rs

        for component_key in ('parameters', 'recorders'):
            for ra, rs in self.generate_node_component_resource_scenarios(pywr_node, component_key,
                                                                          encode_to_json=True):
                yield ra, rs

    def generate_node_schema_resource_scenarios(self, pywr_node):
        """ Generate resource attribute, resource scenario and datasets for a Pywr node.

        """
        node_name = pywr_node['name']
        node_type = pywr_node['type'].lower()
        node_klass = NodeMeta.node_registry.get(node_type)

        msg = f"Node type {node_klass} not recognised."

        if node_klass is None and self.ignore_type_errors is False:
            raise HydraError(msg)

        if node_klass is None:
            log.warning(msg + " Using 'descriptor' as default data type")
            fields = dict((n, 'descriptor') for n in pywr_node.keys())
        else:
            schema = node_klass.Schema()
            fields = schema.fields

        # Create an attribute for each field in the schema.
        for name, field in fields.items():
            if name not in pywr_node:
                continue  # Skip missing fields

            if name in PYWR_PROTECTED_NODE_KEYS:
                continue
            # Non-protected keys represent data that must be added to Hydra.
            if isinstance(field, str):
                data_type = field #the default field if it can't find the class
            else:
                data_type = data_type_from_field(field)

            if data_type == PywrParameter.tag.lower():
                # If the field is defined as general parameter then the actual
                # type might be something more specific.
                try:
                    data_type = data_type_from_parameter_value(pywr_node[name]).tag
                except ValueError:
                    log.warning(f'No Hydra data type for Pywr field "{name}"'
                                f' on node type "{node_type}" found.')

                #TODO: hack to ignore these when they reference parameters elsewhere
                if data_type.lower() == 'descriptor' and pywr_node[name].find(f"__{node_name}__") >= 0:

                    log.warn(f"Ignoring descriptor %s on attribute %s, node %s as this it is assumed this is defined as a parameter, and so will be set as an attribute through the parameters.", pywr_node[name], name, node_name)
                    continue

            # Key is the attribute name. The attributes need to already by added to the
            # database and hence have a valid id.
            attribute_id = self.attribute_name_id_map[name]

            unit_id = self.attr_unit_map.get(attribute_id)

            yield self._make_dataset_resource_attribute_and_scenario(name,
                                                                     pywr_node[name],
                                                                     data_type,
                                                                     attribute_id,
                                                                     unit_id=unit_id,
                                                                     encode_to_json=True)

    def generate_node_component_resource_scenarios(self, pywr_node, component_key,
                                                   **kwargs):

        try:
            components = self.data[component_key]
        except KeyError:
            components = {}

        node_name = pywr_node['name']

        for component_name, component_data in components.items():
            # Filter components to only include those that should be stored at the node level
            if not self.is_component_a_node_attribute(component_name, node_name):
                continue

            data_type = data_type_from_component_type(component_key, component_data['type']).tag
            attribute_name = self._attribute_name(component_key, component_name)

            # This the attribute corresponding to the component.
            # It should have a positive id and already be entered in the hydra database.
            attribute_id = self.attribute_name_id_map[attribute_name.lower()]

            unit_id = self.attr_unit_map.get(attribute_id)

            yield self._make_dataset_resource_attribute_and_scenario(attribute_name,
                                                                     component_data,
                                                                     data_type,
                                                                     attribute_id,
                                                                     unit_id=unit_id,
                                                                     **kwargs)

    def _attribute_name(self, component_key, component_name):
        if component_key in ('parameters', 'recorders'):
            if self._node_attribute_component_delimiter in component_name:
                attribute_name = component_name.split(self._node_attribute_component_delimiter, 1)[-1]
            else:
                attribute_name = component_name
        elif component_key == 'timestepper':
            attribute_name = '{}.{}'.format(component_key, component_name)
        else:
            attribute_name = component_key

        return attribute_name

    def attributes_from_component_dict(self, component_key):
        """ Generator to convert Pywr components data in to Hydra attribute data.

        This function is intended to be used to convert Pywr components
        (e.g. recorders, parameters, etc.) data in to a format that can be imported in to Hydra.
        The Pywr component data is a dict of dict with each sub-dict represent a single component
        (see the "recorder" or "parameters" section of the Pywr JSON format). This
        function returns Hydra data to add a Attribute for each of the components in the outer dict.


        """
        components = self.data[component_key]
        for component_name in components.keys():
            attribute_name = self._attribute_name(component_key, component_name)

            yield self.attr_name_map.get(attribute_name, {
                'name': attribute_name,
                'description': '',
                'dimension_id' : self.attr_dimension_map.get(attribute_name)
            })

    def generate_component_resource_scenarios(self, component_key, **kwargs):
        """ Convert from Pywr components to resource attributes and resource scenarios.

        This function is intended to be used to convert Pywr components
        (e.g. recorders, parameters, etc.) data into a format that can be imported in to Hydra.
        The Pywr component data is a dict of dict with each sub-dict represent a
        single component (see the "recorder" or "parameters" section of the Pywr JSON format).
        This function returns a list of resource attributes and resource scenarios.
        These can be used to import the data
        to Hydra.

        """
        try:
            components = self.data[component_key]
        except KeyError:
            components = {}

        #Recorders and parameters can result in duplicate attributes.
        #To avoid this, we keep track of duplicates and add them as a list within
        #a single attribute, and change the data type
        attribute_data_registry = {}

        for component_name, component_data in components.items():

            if component_key.lower() == 'metadata':
                if component_name in ('title', 'description', 'minimum_version'):
                    # These names are saved on the hydra network directly (name and descripton)
                    # therefore do not add as a attributes as well.
                    continue

            # Determine whether this component should be store on as a node attribute.
            if component_key in ('parameters', 'recorders') and \
                    self.is_component_a_node_attribute(component_name):
                continue

            # Determine the data type
            if component_key in ('parameters', 'recorders'):
                data_type = data_type_from_component_type(component_key, component_data['type']).tag
            else:
                data_type = 'DESCRIPTOR'

            attribute_name = self._attribute_name(component_key, component_name)

            # This the attribute corresponding to the component.
            # It should have a positive id and already be entered in the hydra database.
            attribute_id = self.attribute_name_id_map[attribute_name.lower()]

            attribute_data = {'attribute_name':attribute_name,
                              'data':{component_name:component_data},
                              'data_type':data_type,
                              'attribute_id':attribute_id}

            if attribute_data_registry.get(attribute_name):
                attribute_data_registry[attribute_name]['data'][component_name] = component_data
            else:
                attribute_data_registry[attribute_name] = attribute_data

        for attribute_name, attribute_data in attribute_data_registry.items():
            if attribute_name == 'capacity:capex':
                log.info(f"capacity:capex: {attribute_data}")
            attribute_name = attribute_data['attribute_name']
            data = attribute_data['data']
            if len(data) == 1:
                data = list(data.values())[0]
            data_type = attribute_data['data_type']
            attribute_id = attribute_data['attribute_id']
            unit_id = self.attr_unit_map.get(attribute_id)
            yield self._make_dataset_resource_attribute_and_scenario(attribute_name,
                                                                     data,
                                                                     data_type,
                                                                     attribute_id,
                                                                     unit_id=unit_id,
                                                                     **kwargs)
