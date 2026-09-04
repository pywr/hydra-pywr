import os
import pytest
import hydra_base
from hydra_base import JSONObject
from hydra_pywr.importer import import_json
from hydra_pywr.template import register_template, load_template_config, pywr_template_name
from hydra_client.connection import JSONConnection
from hydra_base_fixtures import testdb_uri


@pytest.fixture()
def client(testdb_uri):
    return JSONConnection(app_name="Test Pywr application.", db_url=testdb_uri)


@pytest.fixture()
def logged_in_client(client):
    root_user_id = client.login('root', '')
    return client


@pytest.fixture
def model_directory():
    return os.path.join(os.path.dirname(__file__), 'models')

@pytest.fixture()
def data_directory():
    return os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture()
def simple1(model_directory):
    return os.path.join(model_directory, 'simple1.json')


@pytest.fixture(params=[
    'simple1.json',
    'reservoir2.json',
    'parameter_reference.json',
    'hydra_generated.json',
    'hydropower.json',
    'simple_with_scenario.json'
])
def pywr_json_filename(request, model_directory):
    return os.path.join(model_directory, request.param)


@pytest.fixture()
def db_with_template(db_with_users, logged_in_client):
    register_template(logged_in_client)


@pytest.fixture()
def db_with_pywr_network(pywr_json_filename, db_with_template, projectmaker, logged_in_client):
    client = logged_in_client

    project = projectmaker.create()

    config = load_template_config('full')
    template = client.get_template_by_name(pywr_template_name(config['name']))

    network_summary = import_json(client, pywr_json_filename, project.id, template['id'], None)

    network_id = network_summary['id']
    new_network = client.get_network(network_id=network_id, include_attributes=False, include_data=False)
    scenario_id = new_network['scenarios'][0]['id']

    return network_id, scenario_id, pywr_json_filename


@pytest.fixture
def hydropower_verification_model(model_directory):
    return os.path.join(model_directory, "hydropower_verification.json")
