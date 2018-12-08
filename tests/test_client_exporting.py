from helpers import *
from fixtures import *
from hydra_base_fixtures import *
from hydra_pywr.exporter import PywrHydraExporter
from hydra_pywr.runner import PywrHydraRunner
from hydra_pywr.template import pywr_template_name, PYWR_TIMESTEPPER_ATTRIBUTES
from pywr.model import Model
import json


def test_export(db_with_pywr_network, logged_in_client):
    client = logged_in_client

    pywr_network_id, pywr_scenario_id, pywr_json_filename = db_with_pywr_network
    exporter = PywrHydraExporter.from_network_id(client, pywr_network_id, pywr_scenario_id)
    pywr_data_exported = exporter.get_pywr_data()

    # Check transformed data is about right
    with open(pywr_json_filename) as fh:
        pywr_data = json.load(fh)

    assert_identical_pywr_data(pywr_data, pywr_data_exported)

    m = Model.load(pywr_data)
    m.run()



def test_runner(db_with_pywr_network, logged_in_client):
    client = logged_in_client

    pywr_network_id, pywr_scenario_id, pywr_json_filename = db_with_pywr_network

    runner = PywrHydraRunner.from_network_id(client, pywr_network_id, pywr_scenario_id)

    runner.load_pywr_model()
    runner.run_pywr_model()
    runner.save_pywr_results(client)


def test_create_empty_network(db_with_template, projectmaker, logged_in_client):
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

    exporter = PywrHydraExporter.from_network_id(client, hydra_network.id, scenario_id)
    pywr_data_exported = exporter.get_pywr_data()

    assert 'timestepper' in pywr_data_exported

    for key in PYWR_TIMESTEPPER_ATTRIBUTES:
        assert key in pywr_data_exported['timestepper']

    assert 'metadata' in pywr_data_exported
    assert 'title' in pywr_data_exported['metadata']
    assert 'description' in pywr_data_exported['metadata']

    assert 'nodes' in pywr_data_exported
    assert 'edges' in pywr_data_exported


def test_export_with_module(db_with_custom_pywr_network, logged_in_client):
    client = logged_in_client

    pywr_network_id, pywr_scenario_id, pywr_json_filename, pywr_module_filename = db_with_custom_pywr_network
    exporter = PywrHydraExporter.from_network_id(client, pywr_network_id, pywr_scenario_id)
    pywr_data_exported, module = exporter.get_pywr_data()

    # Check transformed data is about right
    with open(pywr_json_filename) as fh:
        pywr_data = json.load(fh)

    assert_identical_pywr_data(pywr_data, pywr_data_exported)

    m = Model.load(pywr_data)
    m.run()
