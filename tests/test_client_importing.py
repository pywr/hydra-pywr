from helpers import *
from fixtures import *
from hydra_base_fixtures import *
from hydra_pywr.importer import import_json
from hydra_pywr.template import register_template, load_template_config, pywr_template_name
import pytest


def test_add_network(pywr_json_filename, db_with_template, projectmaker, logged_in_client):
    client = logged_in_client

    project = projectmaker.create()

    config = load_template_config('full')
    template = client.get_template_by_name(pywr_template_name(config['name']))

    import_json(client, pywr_json_filename, project.id, template['id'], None)


def test_add_template(db_with_users, logged_in_client):
    register_template(logged_in_client)


