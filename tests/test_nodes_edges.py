"""
The unit tests in this module test the internal behaviour of the Pywr-Hydra application.

"""
from hydra_pywr.importer import PywrHydraImporter
import pytest
import json
from fixtures import logged_in_client


@pytest.fixture()
def pywr_nodes_edges():
    """ Example node and edge data from Pywr.

    This data looks like "nodes" and "edges" section of a Pywr JSON file.
    """
    nodes_edges = {
        "nodes": [
            {
                "name": "supply1",
                "type": "Input",
                "max_flow": 15
            },
            {
                "name": "link1",
                "type": "Link"
            },
            {
                "name": "demand1",
                "type": "Output",
                "max_flow": 10,
                "cost": -10
            }
        ],
        "edges": [
            ["supply1", "link1"],
            ["link1", "demand1"]
        ]
    }
    return nodes_edges


@pytest.fixture()
def pywr_nodes_edges_importer(logged_in_client, pywr_nodes_edges):
    # Note the use of a fake template here because we're not testing nodes/links.
    return PywrHydraImporter(logged_in_client, pywr_nodes_edges, {'templatetypes': []})


def test_nodes_to_attributes(pywr_nodes_edges_importer):
    importer = pywr_nodes_edges_importer

    attributes = importer.attributes_from_nodes()
    attribute_names = [a['name'] for a in attributes]

    for key in ('max_flow', 'cost'):
        assert key in attribute_names

    for key in ('name', 'comment', 'type'):
        assert key not in attribute_names

