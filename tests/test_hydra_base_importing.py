"""
The unit tests in here interact directly with hydra-base (rather than using the Web API).
"""
from helpers import *
from fixtures import *
from hydra_base_fixtures import *
from hydra_pywr.importer import PywrHydraImporter
from hydra_pywr.template import generate_pywr_attributes, generate_pywr_template, pywr_template_name, \
    PYWR_DEFAULT_DATASETS, load_template_config
import hydra_base
import pytest
import json


def test_add_network(pywr_json_filename, session_with_pywr_template, projectmaker, root_user_id):
    project = projectmaker.create()

    template = JSONObject(hydra_base.get_template_by_name(pywr_template_name('Full')))

    importer = PywrHydraImporter(pywr_json_filename, template)

    # First the attributes must be added.
    attributes = [JSONObject(a) for a in importer.add_attributes_request_data()]

    # The response attributes have ids now.
    response_attributes = hydra_base.add_attributes(attributes)

    # Convert to a simple dict for local processing.
    # TODO change this variable name to map or lookup
    attribute_ids = {a.name: a.id for a in response_attributes}

    # Now we try to create the network
    network = importer.add_network_request_data(attribute_ids, project.id)

    # Check transformed data is about right
    with open(pywr_json_filename) as fh:
        pywr_data = json.load(fh)

    assert_hydra_pywr(network, pywr_data)
    hydra_network = hydra_base.add_network(JSONObject(network), user_id=root_user_id)


def test_add_template(session, root_user_id):

    attributes = [JSONObject(a) for a in generate_pywr_attributes()]

    # The response attributes have ids now.
    response_attributes = hydra_base.add_attributes(attributes)

    # Convert to a simple dict for local processing.
    attribute_ids = {a.name: a.id for a in response_attributes}

    default_data_set_ids = {}
    for attribute_name, dataset in PYWR_DEFAULT_DATASETS.items():
        hydra_dataset = hydra_base.add_dataset(flush=True, **dataset)
        default_data_set_ids[attribute_name] = hydra_dataset.id    

    config = load_template_config('full')
    template = generate_pywr_template(attribute_ids, default_data_set_ids, config)

    hydra_base.add_template(JSONObject(template))
