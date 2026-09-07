from helpers import *
from fixtures import *
from hydra_base_fixtures import *
from hydra_pywr.exporter import HydraToPywrNetwork, export_json
from hydra_pywr.runner import run_network_scenario
from hydra_pywr.template import pywr_template_name, PYWR_TIMESTEPPER_ATTRIBUTES
from pywrparser.types.network import PywrNetwork
from pywr.model import Model
import json


def test_export(db_with_pywr_network, logged_in_client, tmp_path):
    client = logged_in_client

    _, pywr_scenario_id, pywr_json_filename = db_with_pywr_network

    outfile = export_json(client, str(tmp_path), pywr_scenario_id, use_cache=False,
                          json_sort_keys=False, json_indent=2)

    with open(outfile) as fh:
        pywr_data_exported = json.load(fh)

    # Check transformed data is about right
    with open(pywr_json_filename) as fh:
        pywr_data = json.load(fh)

    assert_identical_pywr_data(pywr_data, pywr_data_exported)

    m = Model.load(pywr_data)
    m.run()


def test_runner(db_with_pywr_network, logged_in_client, tmp_path):
    client = logged_in_client

    _, pywr_scenario_id, pywr_json_filename = db_with_pywr_network

    runner = run_network_scenario(client, pywr_scenario_id, template_id=None, data_dir=str(tmp_path))

    assert runner.model is not None


def test_create_empty_network(db_with_template, projectmaker, logged_in_client, tmp_path):
    client = logged_in_client

    project = projectmaker.create()
    template = client.get_template_by_name(pywr_template_name('Full'))

    # Find the network type in this template.
    # There is only one of these in the template.
    for template_type in template['templatetypes']:
        if template_type['resource_type'] == 'NETWORK':
            template_type_id = template_type['id']
            break
    else:
        raise ValueError('No network type found in this template!')

    # This is a minimal network with no data and a scenario
    network_data = {
        'name': 'empty', 'description': 'empty network', 'project_id': project.id, 'types': [{'id': template_type_id}],
        'scenarios': [{
            "name": "Baseline",
            "description": "",
            "resourcescenarios": []
        }]
    }

    hydra_network = client.add_network(network_data)

    scenario_id = hydra_network['scenarios'][0]['id']

    exporter = HydraToPywrNetwork.from_scenario_id(client, scenario_id, data_dir=str(tmp_path))
    pywr_network_data = exporter.build_pywr_network()
    pywr_data_exported = PywrNetwork(pywr_network_data).as_dict()

    assert 'timestepper' in pywr_data_exported

    for key in PYWR_TIMESTEPPER_ATTRIBUTES:
        assert key in pywr_data_exported['timestepper']

    assert 'metadata' in pywr_data_exported
    assert 'title' in pywr_data_exported['metadata']
    assert 'description' in pywr_data_exported['metadata']

    assert 'nodes' in pywr_data_exported
    assert 'edges' in pywr_data_exported
