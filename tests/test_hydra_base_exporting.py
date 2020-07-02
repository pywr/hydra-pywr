"""
The unit tests in here interact directly with hydra-base (rather than using the Web API).
"""
from helpers import *
from fixtures import *
from hydra_base_fixtures import *
from hydra_pywr.importer import PywrHydraImporter
from hydra_pywr.exporter import PywrHydraExporter
from hydra_pywr.template import pywr_template_name
import hydra_base
import pytest
import json


def test_export(session_with_pywr_network, root_user_id):
    pywr_network_id, pywr_json_filename = session_with_pywr_network

    # Fetch the network
    network = hydra_base.get_network(pywr_network_id, user_id=root_user_id, include_data='Y')
    # Fetch all the attributes
    attributes = hydra_base.get_attributes(user_id=root_user_id)
    # TODO not sure why this returns SQLAlchemy object?
    # TODO rename this to map/lookup
    attributes = {attr.id: JSONObject(attr) for attr in attributes}

    # We also need the template to get the node types
    template = JSONObject(hydra_base.get_template_by_name(pywr_template_name('Full'), user_id=root_user_id))

    exporter = PywrHydraExporter(network, attributes, template)

    pywr_data_exported = exporter.get_pywr_data()

    # Check transformed data is about right
    with open(pywr_json_filename) as fh:
        pywr_data = json.load(fh)

    assert_identical_pywr_data(pywr_data, pywr_data_exported)
