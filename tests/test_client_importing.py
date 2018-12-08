from helpers import *
from fixtures import *
from hydra_base_fixtures import *
from hydra_pywr.importer import PywrHydraImporter
from hydra_pywr.template import register_template
import pytest
import os


def test_add_network(pywr_json_filename, db_with_template, projectmaker, logged_in_client):
    client = logged_in_client

    project = projectmaker.create()

    importer = PywrHydraImporter.from_client(client, pywr_json_filename, 'full')
    importer.import_data(client, project.id)


def test_add_template(db_with_users, logged_in_client):
    register_template(logged_in_client)


def test_add_network_with_module(model_directory, db_with_template, projectmaker, logged_in_client):

    client = logged_in_client

    project = projectmaker.create()

    pywr_json_filename = os.path.join(model_directory, 'custom.json')
    pywr_module_filename = os.path.join(model_directory, 'custom_node.py')

    importer = PywrHydraImporter.from_client(client, pywr_json_filename, 'full',
                                             modules=[pywr_module_filename])
    importer.import_data(client, project.id)
