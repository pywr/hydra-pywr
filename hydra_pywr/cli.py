import click
import pandas

from hydra_client.connection import RemoteJSONConnection

from hydra_client.click import hydra_app

from . import runner
from . import exporter
from . import importer

from .template import register_template, unregister_template, migrate_network_template, TemplateExistsError
from . import utils


def get_client(**kwargs):
    return RemoteJSONConnection(app_name='Pywr Hydra App', **kwargs)


def get_logged_in_client(context):
    session = context.get("session")
    client = get_client(url=context.get('hostname'), session_id=session)
    if client.user_id is None and session is None:
        client.login(username=context["username"], password=context["password"])
    return client


def start_cli():
    cli(obj={}, auto_envvar_prefix='HYDRA_PYWR')


@click.group()
@click.pass_obj
@click.option('-u', '--username', type=str, default=None)
@click.option('-p', '--password', type=str, default=None)
@click.option('-h', '--hostname', type=str, default=None)
@click.option('-s', '--session', type=str, default=None)
def cli(obj, username, password, hostname, session):
    """ CLI for the Pywr-Hydra application. """

    obj['hostname'] = hostname
    obj['username'] = username
    obj['password'] = password
    obj['session'] = session


@hydra_app(category='import', name='Import Pywr JSON')
@cli.command(name='import', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('--filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-p', '--project-id', type=int)
@click.option('--template-id', type=int)
@click.option('--projection', type=str, default=None)
@click.option('--network-name', type=str, default=None)
@click.option('--rewrite-url-prefix', type=str, default=None)
def import_json(obj, filename, project_id, template_id, projection, network_name, rewrite_url_prefix, *args):
    """ Import a Pywr JSON file into Hydra. """

    client = get_logged_in_client(obj)

    importer.import_json(client,
                         filename,
                         project_id,
                         template_id,
                         network_name,
                         *args,
                         rewrite_url_prefix=rewrite_url_prefix,
                         projection=projection)


@hydra_app(category='export', name='Export to Pywr JSON')
@cli.command(name='export', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('--data-dir', default='/tmp')
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('--use-cache', is_flag=True)
@click.option('--json-sort-keys/--no-json-sort-keys', default=False)
@click.option('--json-indent', type=int, default=2)
def export_json(obj, data_dir, scenario_id, use_cache, json_sort_keys, json_indent):
    """ Export a Pywr JSON from Hydra. """
    client = get_logged_in_client(obj)
    exporter.export_json(client,
                         data_dir,
                         scenario_id,
                         use_cache,
                         json_sort_keys,
                         json_indent)


@cli.command(name="run-file", context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.argument("filename", type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('--domain', type=str, default="water")
@click.option('--output-file', type=str, default="output.csv")
def run_file(obj, filename, domain, output_file):
    runner.run_file(filename, domain, output_file)

@cli.command(name="purge-cache", context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.option("--cache-path", type=click.Path(file_okay=False, dir_okay=True, exists=True))
def purge_cache(cache_path):
    from hydra_pywr.filecache import FileCache
    fc = FileCache(cache_path)
    fc.purge_all()


@hydra_app(category='model', name='Run Pywr')
@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('-t', '--template-id', type=int, default=None)
@click.option('--domain', type=str, default="water")
@click.option('--output-frequency', type=str, default=None)
@click.option('--solver', type=str, default=None)
@click.option('--data-dir', default='/tmp')
def run(obj, scenario_id, template_id, domain, output_frequency, solver, data_dir):
    """ Export, run and save a Pywr model from Hydra. """
    client = get_logged_in_client(obj)

    if scenario_id is None:
        raise Exception('No scenario specified')

    runner.run_network_scenario(client,
                                scenario_id,
                                template_id,
                                domain,
                                output_frequency,
                                data_dir=data_dir)


"""
  Miscellaneous Utilities - to be reviewed and/or relocated
"""
@hydra_app(category='network_utility', name='Step model')
@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('-n', '--network-id', type=int, default=None)
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('--child-scenario-ids', type=int, default=None, multiple=True)
def step_model(obj, network_id, scenario_id, child_scenario_ids):
    client = get_logged_in_client(obj)
    utils.apply_final_volumes_as_initial_volumes(client, scenario_id, child_scenario_ids)
    utils.progress_start_end_dates(client, network_id, scenario_id)


@hydra_app(category='network_utility', name='Apply initial volumes')
@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('--child-scenario-ids', type=int, default=None, multiple=True)
def apply_initial_volumes_to_other_networks(obj, scenario_id, child_scenario_ids):
    client = get_logged_in_client(obj)
    utils.apply_final_volumes_as_initial_volumes(client, scenario_id, child_scenario_ids)


@hydra_app(category='network_utility', name='Step forward the game')
@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('--child-scenario-ids', type=int, default=None, multiple=True)
@click.option('--filename', type=click.Path(file_okay=True, dir_okay=False))
@click.option('--attribute-name', type=str, default=None)
@click.option('--index-col', type=str, default=None)
@click.option('--column-name', type=str, default=None)
@click.option('--data-type', type=str, default='PYWR_DATAFRAME')
@click.option('--create-new/--no-create-new', default=False)
def step_game(obj, scenario_id, child_scenario_ids, filename, attribute_name, index_col,
              column_name, data_type, create_new):
    client = get_logged_in_client(obj)

    # Create new scenarios in each of the networks
    new_scenario_ids = list(utils.clone_scenarios(client, child_scenario_ids))

    # Update the initial volumes
    utils.apply_final_volumes_as_initial_volumes(client, scenario_id, new_scenario_ids)
    # Load the new data
    dataframe = pandas.read_csv(filename, index_col=index_col, parse_dates=True)
    # Update the time-step and data for each scenario
    for new_scenario_id in new_scenario_ids:
        utils.import_dataframe(client, dataframe, new_scenario_id, attribute_name,
                               create_new=create_new, data_type=data_type, column=column_name)
        utils.progress_start_end_dates(client, new_scenario_id)


@cli.group()
def template():
    pass


@template.command('register')
@click.option('-c', '--config', type=str, default='full')
@click.option('--update/--no-update', default=False)
@click.pass_obj
def template_register(obj, config, update):
    """ Register a Pywr template with Hydra. """

    client = get_logged_in_client(obj)
    try:
        register_template(client, config_name=config, update=update)
    except TemplateExistsError:
        click.echo('The template is already registered. To force an updated use the --update option.')


@template.command('unregister')
@click.option('-c', '--config', type=str, default='full')
@click.pass_obj
def template_unregister(obj, config):
    """ Unregister a Pywr template with Hydra. """
    client = get_logged_in_client(obj)
    if click.confirm('Are you sure you want to remove the template? '
                     'This will invalidate any existing networks that use the template.'):
        unregister_template(client, config_name=config)


@template.command('migrate')
@click.argument('network-id', type=int)
@click.option('--template-name', type=str, default=None)
@click.option('--template-id', type=int, default=None)
@click.pass_obj
def template_migrate(obj, network_id, template_name, template_id):
    client = get_logged_in_client(obj)
    if click.confirm('Are you sure you want to migrate network {} to a new template?'.format(network_id)):
        migrate_network_template(client, network_id, template_name=template_name, template_id=template_id)
