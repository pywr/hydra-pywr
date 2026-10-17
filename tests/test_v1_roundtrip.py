"""
Baseline tests for the Pywr v1 path: register the template, import a v1 JSON model into Hydra and
export it again. Uses a temporary SQLite database (see hydra_base_fixtures.testdb_uri); nothing
here may be pointed at a MySQL or live database.
"""
import json
import os

import pytest

from hydra_base_fixtures import db_backend, testdb_uri, engine, db_empty, db_with_users  # noqa: F401
from local_client import LocalHydraClient, import_template_file
from hydra_pywr.importer import import_json
from hydra_pywr.exporter import HydraToPywrNetwork

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')


@pytest.fixture()
def v1_client(testdb_uri, db_with_users):  # noqa: F811
    assert testdb_uri.startswith('sqlite:///'), "v1 round-trip tests must use a temporary SQLite database"
    client = LocalHydraClient(app_name="hydra-pywr v1 round-trip tests", db_url=testdb_uri)
    client.login('root', '')
    return client


@pytest.fixture()
def v1_template(v1_client):
    return import_template_file(v1_client, 'pywr_parent_template_v1.json')


@pytest.mark.parametrize('model', ['simple1.json', 'reservoir2.json', 'parameter_reference.json',
                                   'simple_with_scenario.json'])
def test_import_export_roundtrip(model, v1_client, v1_template, tmp_path):
    path = os.path.join(MODEL_DIR, model)
    with open(path) as fh:
        original = json.load(fh)

    project = v1_client.add_project({'name': 'v1 roundtrip %s' % model, 'description': 'test'})
    summary = import_json(v1_client, path, project['id'], v1_template['id'], None)

    network = v1_client.get_network(network_id=summary['id'], include_attributes=False, include_data=False)
    scenario_id = network['scenarios'][0]['id']

    exporter = HydraToPywrNetwork.from_scenario_id(v1_client, scenario_id, data_dir=str(tmp_path))
    exported = exporter.build_pywr_network()

    # build_pywr_network returns the exporter, with the Pywr components held as attributes.
    exported_nodes = exported.nodes.values() if isinstance(exported.nodes, dict) else exported.nodes
    exported_names = {n['name'] if isinstance(n, dict) else n.name for n in exported_nodes}
    assert exported_names == {n['name'] for n in original['nodes']}
    assert len(exported.edges) == len(original['edges'])
