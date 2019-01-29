import os
import pytest
import hydra_base
from hydra_base import JSONObject
from hydra_pywr.importer import PywrHydraImporter
from hydra_pywr.template import pywr_template_name, register_template
from hydra_client.connection import JSONConnection
from hydra_base_fixtures import testdb_uri


@pytest.fixture()
def client(testdb_uri):
    return JSONConnection(app_name="Test Pywr application.", db_url=testdb_uri)


@pytest.fixture()
def logged_in_client(client):
    root_user_id = client.login('root', '')
    return client


@pytest.fixture()
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
    'national_model.json',
])
def pywr_json_filename(request, model_directory):
    return os.path.join(model_directory, request.param)


@pytest.fixture()
def session_with_pywr_network(pywr_json_filename, session_with_pywr_template, projectmaker, root_user_id):
    project = projectmaker.create()

    template = JSONObject(hydra_base.get_template_by_name(pywr_template_name('Full')))

    importer = PywrHydraImporter(pywr_json_filename, template)

    # First the attributes must be added.
    attributes = [JSONObject(a) for a in importer.add_attributes_request_data()]

    # The response attributes have ids now.
    response_attributes = hydra_base.add_attributes(attributes)

    # Convert to a simple dict for local processing.
    attribute_ids = {a.name: a.id for a in response_attributes}

    # Now we try to create the network
    network = importer.add_network_request_data(attribute_ids, project.id)
    hydra_network = hydra_base.add_network(JSONObject(network), user_id=root_user_id)

    return hydra_network.id, pywr_json_filename


@pytest.fixture()
def db_with_template(db_with_users, logged_in_client):
    register_template(logged_in_client)


@pytest.fixture()
def db_with_pywr_network(pywr_json_filename, db_with_template, projectmaker, logged_in_client):
    client = logged_in_client

    project = projectmaker.create()

    importer = PywrHydraImporter.from_client(client, pywr_json_filename, 'full')
    network_id, scenario_id = importer.import_data(client, project.id)

    return network_id, scenario_id, pywr_json_filename

