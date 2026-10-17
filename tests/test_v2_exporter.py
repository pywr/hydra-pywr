"""
Tests for the Pywr v2 exporter (hydra_pywr.exporter_v2).

A small network is built in Hydra from the Pywr v2 parent template, exported to Pywr v2 JSON,
validated against the v2 schema and run. Requires the Pywr v2 wheel (`pip install --pre pywr`),
which cannot share a Python environment with Pywr v1, so the module is skipped when v1 is installed.
Uses a temporary SQLite database; nothing here may be pointed at a MySQL or live database.
"""
import json
import os

import pytest

pywr = pytest.importorskip("pywr")
if not hasattr(pywr, "ModelSchema"):
    pytest.skip("Pywr v2 is required for the v2 exporter tests (found Pywr v1)", allow_module_level=True)

from hydra_base_fixtures import db_backend, testdb_uri, engine, db_empty, db_with_users  # noqa: F401,E402
from local_client import LocalHydraClient, import_template_file  # noqa: E402
from hydra_pywr.exporter_v2 import HydraToPywrV2Network, export_json_v2  # noqa: E402


@pytest.fixture()
def v2_client(testdb_uri, db_with_users):  # noqa: F811
    assert testdb_uri.startswith('sqlite:///'), "v2 exporter tests must use a temporary SQLite database"
    client = LocalHydraClient(app_name="hydra-pywr v2 exporter tests", db_url=testdb_uri)
    client.login('root', '')
    return client


@pytest.fixture()
def v2_template(v2_client):
    return import_template_file(v2_client, 'pywr_parent_template_v2.json')


class NetworkBuilder:
    """Builds a Hydra network payload (nodes, links, datasets) from a v2 template."""

    def __init__(self, client, template, project_id):
        self.client = client
        self.template = template
        self.project_id = project_id
        self.types = {t['name']: t for t in template['templatetypes']}
        self.nodes, self.links, self.network_ras = [], [], []
        self.resource_scenarios = []
        self._ra = 0
        self._id = 0
        self._attr_ids = {}
        for t in template['templatetypes']:
            for ta in t['typeattrs']:
                self._attr_ids[ta['attr']['name']] = ta['attr_id']

    def _next(self):
        self._id -= 1
        return self._id

    def _attr_id(self, name):
        if name not in self._attr_ids:
            created = self.client.add_attributes(attrs=[{'name': name, 'description': name,
                                                         'project_id': self.project_id}])
            self._attr_ids[name] = created[0]['id']
        return self._attr_ids[name]

    def _resource_attr(self, name, value, datatype='descriptor'):
        self._ra -= 1
        ra = {'id': self._ra, 'attr_id': self._attr_id(name), 'attr_is_var': 'N'}
        self.resource_scenarios.append({
            'resource_attr_id': self._ra,
            'dataset': {'name': name, 'type': datatype, 'value': value, 'metadata': '{}',
                        'unit': '-', 'unit_id': None, 'hidden': 'N'}})
        return ra

    def node(self, name, node_type, **attrs):
        node = {'resource_type': 'NODE', 'id': self._next(), 'name': name, 'x': 0, 'y': 0, 'layout': {},
                'types': [{'id': self.types[node_type]['id'], 'child_template_id': self.template['id']}],
                'attributes': [self._resource_attr(k, json.dumps(v)) for k, v in attrs.items()]}
        self.nodes.append(node)
        return node

    def link(self, src, dest):
        self.links.append({'resource_type': 'LINK', 'id': self._next(), 'name': '%s to %s' % (src['name'], dest['name']),
                           'node_1_id': src['id'], 'node_2_id': dest['id'], 'layout': {}, 'attributes': [],
                           'types': [{'id': self.types['edge']['id'], 'child_template_id': self.template['id']}]})

    def network_attr(self, name, value):
        self.network_ras.append(self._resource_attr(name, value))

    def create(self, name='v2 test network'):
        network_type = next(t for t in self.template['templatetypes'] if t['resource_type'] == 'NETWORK')
        payload = {
            'name': name, 'description': 'hydra-pywr v2 exporter test', 'project_id': self.project_id,
            'nodes': self.nodes, 'links': self.links, 'layout': None, 'appdata': {}, 'projection': 'EPSG:4326',
            'attributes': self.network_ras,
            'scenarios': [{'name': 'Baseline', 'description': 'test', 'resourcescenarios': self.resource_scenarios}],
            'types': [{'id': network_type['id'], 'child_template_id': self.template['id']}],
        }
        summary = self.client.add_network({'net': payload})
        network = self.client.get_network(network_id=summary['id'], include_attributes=False, include_data=False)
        return network['id'], network['scenarios'][0]['id']


@pytest.fixture()
def simple_network(v2_client, v2_template):
    """Catchment -> Link -> Output, with a Literal max_flow and a negative-cost demand."""
    project = v2_client.add_project({'name': 'v2 exporter tests', 'description': 'test'})
    b = NetworkBuilder(v2_client, v2_template, project['id'])
    catchment = b.node('inflow', 'Catchment', flow=5)
    link = b.node('link', 'Link')
    demand = b.node('demand', 'Output', max_flow=10, cost=-10)
    b.link(catchment, link)
    b.link(link, demand)
    b.network_attr('timestepper.start', '2020-01-01')
    b.network_attr('timestepper.end', '2020-01-10')
    b.network_attr('timestepper.timestep', '1')
    return b.create()


def test_v2_template_node_types(v2_template):
    """The template imports with the expected v2 node types and no network-level attributes."""
    names = {t['name'] for t in v2_template['templatetypes']}
    assert {'Catchment', 'Link', 'Output', 'Reservoir', 'River', 'Turbine'} <= names
    network_types = [t for t in v2_template['templatetypes'] if t['resource_type'] == 'NETWORK']
    assert len(network_types) == 1
    assert network_types[0]['typeattrs'] == []


def test_export_builds_valid_v2_model(v2_client, simple_network):
    network_id, scenario_id = simple_network
    exporter = HydraToPywrV2Network.from_scenario_id(v2_client, scenario_id)
    model = exporter.build()

    assert {n['meta']['name'] for n in model['network']['nodes']} == {'inflow', 'link', 'demand'}
    assert {n['type'] for n in model['network']['nodes']} == {'Catchment', 'Link', 'Output'}
    assert len(model['network']['edges']) == 2
    assert model['time']['start'] == '2020-01-01'  # the v2 schema calls the timestepper 'time'

    nodes = {n['meta']['name']: n for n in model['network']['nodes']}
    assert nodes['inflow']['flow'] == {'type': 'Literal', 'value': 5}
    assert nodes['demand']['max_flow'] == {'type': 'Literal', 'value': 10}
    assert nodes['demand']['cost'] == {'type': 'Literal', 'value': -10}

    # The exported JSON must be accepted by the Pywr v2 schema.
    pywr.ModelSchema.from_json_string(json.dumps(model))


def test_exported_model_runs(v2_client, simple_network, tmp_path):
    network_id, scenario_id = simple_network
    outfile = export_json_v2(v2_client, str(tmp_path), scenario_id)
    assert os.path.exists(outfile)

    schema = pywr.ModelSchema.from_path(outfile)
    model = schema.build(str(tmp_path), None)
    result = model.run("clp")
    assert result is not None

    # The exporter adds a Memory output over all nodes, named 'all'. The catchment supplies 5 per
    # day to a demand that can take up to 10, so every node in the chain carries 5 for each of the
    # 10 days. Results come back as a long-format Arrow record batch.
    network_result = result.network_result
    assert 'all' in network_result.output_names()
    batch = network_result.to_record_batch('all')
    assert {'time_start', 'name', 'attribute', 'value'} <= set(batch.schema.names)
    assert batch.num_rows == 30
    rows = batch.to_pylist()
    assert {r['name'] for r in rows} == {'inflow', 'link', 'demand'}
    assert {round(r['value'], 6) for r in rows} == {5.0}
