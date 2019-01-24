from helpers import *
from fixtures import *
from hydra_base_fixtures import *
from hydra_pywr.importer import PywrHydraImporter
from hydra_pywr.exporter import PywrHydraExporter
from hydra_base.lib.objects import Dataset
import os
import pytest


@pytest.fixture()
def pywr_with_demand_pattern(model_directory, db_with_template, projectmaker, logged_in_client):
    client = logged_in_client

    # Create the basic pywr model
    project = projectmaker.create()
    pywr_json_filename = os.path.join(model_directory, 'simple1.json')
    importer = PywrHydraImporter.from_client(client, pywr_json_filename, 'full')
    network_id, scenario_id = importer.import_data(client, project.id)

    # Create the demand pattern
    pattern_attr = client.add_attribute({'name': 'demand_pattern'})
    ra = client.add_resource_attribute('NETWORK', network_id, pattern_attr['id'], 'N')

    with open(os.path.join(model_directory, 'simple_demand_pattern.json')) as fh:
        pattern_str = fh.read()

    pattern_data = Dataset({
        'name': 'demand_pattern',
        'value': pattern_str,
        "hidden": "N",
        "type": 'PYWR_PARAMETER_PATTERN',
        "unit": "-",
    })

    client.add_data_to_attribute(scenario_id, ra['id'], pattern_data)

    # Assign the pattern to one of the nodes
    node = client.get_node_by_name(network_id, 'demand1')
    pattern_ref_attr = client.add_attribute({'name': 'demand'})
    ra = client.add_resource_attribute('NODE', node['id'], pattern_ref_attr['id'], 'N')

    pattern_ref_data = Dataset({
        'name': 'demand',
        'value': 'demand_pattern',
        'hidden': 'N',
        'type': 'PYWR_PARAMETER_PATTERN_REF'
    })

    client.add_data_to_attribute(scenario_id, ra['id'], pattern_ref_data)

    #
    population_attr = client.add_attribute({'name': 'population'})
    ra = client.add_resource_attribute('NODE', node['id'], population_attr['id'], 'N')

    population_data = Dataset({
        'name': 'population',
        'value': 3.14,
        'hidden': 'N',
        'type': 'SCALAR'
    })

    client.add_data_to_attribute(scenario_id, ra['id'], population_data)

    return network_id, scenario_id


def test_simple_demand_patter(pywr_with_demand_pattern, logged_in_client):
    client = logged_in_client
    pywr_network_id, pywr_scenario_id = pywr_with_demand_pattern

    exporter = PywrHydraExporter.from_network_id(client, pywr_network_id, pywr_scenario_id)
    pywr_data_exported = exporter.get_pywr_data()

    assert 'parameters' in pywr_data_exported

    parameters = pywr_data_exported['parameters']

    assert 'demand1-population' in parameters
    assert parameters['demand1-population']['value'] == 3.14


