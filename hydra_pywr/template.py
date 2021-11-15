""" Module to generate a Hydra template from Pywr.
"""
from pywr.domains.river import *
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayStorageRecorder
import pywr
import os
import json
import copy
from hydra_base.exceptions import HydraError

PYWR_EDGE_LINK_NAME = 'edge'
PYWR_SPLIT_LINK_TYPES = ['riversplit', 'riversplitwithgauge', 'multisplitlink']
PYWR_CONSTRAINED_EDGE_LINK_NAME = 'constrained edge'
PYWR_CONSTRAINED_EDGE_ATTRIBUTES = ('min_flow', 'max_flow', 'cost')

PYWR_PROTECTED_NODE_KEYS = ('name', 'comment', 'type', 'position')

PYWR_ARRAY_RECORDER_ATTRIBUTES = {
    NumpyArrayNodeRecorder: 'simulated_flow',
    NumpyArrayStorageRecorder: 'simulated_volume'
}

PYWR_OUTPUT_ATTRIBUTES = list(PYWR_ARRAY_RECORDER_ATTRIBUTES.values())
PYWR_TIMESTEPPER_ATTRIBUTES = ('start', 'end', 'timestep')
PYWR_DEFAULT_DATASETS = {
    'start': {'data_type': 'descriptor', 'val': '2018-01-01', 'name': 'Default start date'},
    'end': {'data_type': 'descriptor', 'val': '2018-12-31', 'name': 'Default end date'},
    'timestep': {'data_type': 'scalar', 'val': 1, 'name': 'Default timestep'},
}

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'template_configs')


def _load_layouts():
    with open(os.path.join(os.path.dirname(__file__), 'node_layouts.json')) as fh:
        return json.load(fh)
PYWR_LAYOUTS = _load_layouts()


def get_layout(node_klass):

    layout = copy.deepcopy(PYWR_LAYOUTS['__default__'])
    try:
        node_specific_layout = PYWR_LAYOUTS[node_klass.__name__.lower()]
    except KeyError:
        node_specific_layout = {}
    layout.update(node_specific_layout)
    return layout


class TemplateExistsError(ValueError):
    pass


def pywr_template_name(config_name):
    """ The name of the Hydra template for Pywr. """
    return 'Pywr {} template (version: {})'.format(config_name, pywr.__version__)



def generate_pywr_template(attribute_ids, default_data_set_ids, config):

    template_types = [
        # Default link type (not constraint)
        {
            'name': PYWR_EDGE_LINK_NAME,
            'resource_type': 'LINK',
            'typeattrs': [],
            # Default layout for links
            'layout': {"linestyle": "solid", "width": "7", "color": "#000000", "hidden": "N"}
        },
        # Constraint link
        {
            'name': PYWR_CONSTRAINED_EDGE_LINK_NAME,
            'resource_type': 'LINK',
            'typeattrs': [
                {
                    'attr_id': attribute_ids[name],
                    'data_type': 'scalar',
                    'description': '',
                    'is_var': 'N'
                }
                for name in PYWR_CONSTRAINED_EDGE_ATTRIBUTES
            ],
            # Default layout for links
            'layout': {"linestyle": "solid", "width": "7", "color": "#CA3013", "hidden": "N"}
        },
        # Timestepper attributes
        {
            'name': 'Pywr {}'.format(config['name']),
            'resource_type': 'NETWORK',
            'typeattrs': [
                {
                    'attr_id': attribute_ids['timestepper.{}'.format(name)],
                    'data_type': 'descriptor' if name != 'timestep' else 'scalar',
                    'description': '',
                    'default_dataset_id': default_data_set_ids[name],
                    'is_var': 'N'
                }
                for name in PYWR_TIMESTEPPER_ATTRIBUTES
            ]
        }
    ]

    # Get any white or black listed nodes from the template configuration.
    node_whitelist = config['nodes'].get('whitelist', None)
    if node_whitelist is not None:
        node_whitelist = [n.lower() for n in node_whitelist]
    node_blacklist = config['nodes'].get('blacklist', None)
    if node_blacklist is not None:
        node_blacklist = [n.lower() for n in node_blacklist]
    #
    for t in generate_pywr_node_templates(attribute_ids, whitelist=node_whitelist,
                                          blacklist=node_blacklist):
        template_types.append(t)

    # TODO add layout
    template = {
        'name': pywr_template_name(config['name']),
        'templatetypes': template_types,
    }

    return template


def add_default_datasets(client):

    default_data_set_ids = {}
    for attribute_name, dataset in PYWR_DEFAULT_DATASETS.items():
        hydra_dataset = client.add_dataset(flush=True, **dataset)
        default_data_set_ids[attribute_name] = hydra_dataset['id']
    return default_data_set_ids


def register_template(client, config_name='full', update=False):
    """ Register the template with Hydra. """
    config = load_template_config(config_name)

    # check to see if the template exists first.
    template_name = pywr_template_name(config['name'])
    try:
        existing_template = client.get_template_by_name(template_name)
    except HydraError:
        existing_template = None

    attributes = [a for a in generate_pywr_attributes()]

    # The response attributes have ids now.
    response_attributes = client.add_attributes(attributes)

    # Now add the default datasets
    default_data_set_ids = add_default_datasets(client)

    # Convert to a simple dict for local processing.
    attribute_ids = {a.name: a.id for a in response_attributes}

    template = generate_pywr_template(attribute_ids, default_data_set_ids, config)

    if existing_template is None:
        # No template. Add a new one.
        client.add_template(template)
    else:
        if not update:
            raise TemplateExistsError('Template with name f{template_name} already exists.')

        # Map existing template types to new ones by name
        for new_tt in template['templatetypes']:
            for existing_tt in existing_template['templatetypes']:
                if new_tt['name'] == existing_tt['name'] and \
                        new_tt['resource_type'] == existing_tt['resource_type']:
                    new_tt['id'] = existing_tt['id']
                    new_tt['template_id'] = existing_template['id']

                    for ta in new_tt['typeattrs']:
                        ta['type_id'] = new_tt['id']

        template['id'] = existing_template['id']
        client.update_template(template)


def unregister_template(client, config_name='full'):
    """ Unregister the template with Hydra. """

    config = load_template_config(config_name)
    template = client.get_template_by_name(pywr_template_name(config['name']))
    client.delete_template(template['id'])


def load_template_config(config_name):
    with open(os.path.join(CONFIG_DIR, '{}.json'.format(config_name))) as fh:
        config = json.load(fh)
    return config


def migrate_network_template(client, network_id, template_id=None, template_name=None):
    """ Migrate an existing network to a new template.

    This will remove all existing templates from the network.
    """
    if template_id is None and template_name is None:
        raise ValueError('One of either `template_id` or `template_name` must be given.')

    if template_id is None:
        new_template = client.get_template_by_name(template_name)
        template_id = new_template['id']

    # Get the existing network
    network = client.get_network(network_id)
    # Remove all existing templates.
    for network_type in network['types']:
        client.remove_template_from_network(network_id, network_type['template_id'], 'N')

    # Apply new template
    client.apply_template_to_network(template_id, network_id)
    network = client.get_network(network_id)

    # Check the template has been applied correctly.
    assert len(network['types']) == 1
    for network_type in network['types']:
        assert network_type['template_id'] == template_id
