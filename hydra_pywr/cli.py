import click
import json
import os
from hydra_client.connection import JSONConnection
from .exporter import PywrHydraExporter
from .runner import PywrHydraRunner
from .importer import PywrHydraImporter
from .template import register_template, unregister_template, migrate_network_template, TemplateExistsError
from . import utils
from hydra_client.click import hydra_app, make_plugins, write_plugins


def get_client(hostname, **kwargs):
    return JSONConnection(app_name='Pywr Hydra App', db_url=hostname, **kwargs)


def get_logged_in_client(context, user_id=None):
    session = context['session']
    client = get_client(context['hostname'], session_id=session, user_id=user_id)
    if client.user_id is None:
        client.login(username=context['username'], password=context['password'])
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
@cli.command(name='import')
@click.pass_obj
@click.option('--filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-p', '--project-id', type=int)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('--template-id', type=int, default=None)
@click.option('--projection', type=str, default=None)
@click.option('--run/--no-run', default=False)
@click.option('--solver', type=str, default=None)
@click.option('--check-model/--no-check-model', default=True)
def import_json(obj, filename, project_id, user_id, template_id, projection, run, solver, check_model):
    """ Import a Pywr JSON file into Hydra. """
    click.echo(f'Beginning import of "{filename}"! Project ID: {project_id}')
    client = get_logged_in_client(obj, user_id=user_id)
    importer = PywrHydraImporter.from_client(client, filename, template_id)
    network_id, scenario_id = importer.import_data(client, project_id, projection=projection)

    click.echo(f'Successfully imported "{filename}"! Network ID: {network_id}, Scenario ID: {scenario_id}')

    if run:
        run_network_scenario(client, network_id, scenario_id, solver=solver, check_model=check_model)


@hydra_app(category='export', name='Export to Pywr JSON')
@cli.command(name='export')
@click.pass_obj
@click.option('--data-dir', default='/tmp')
@click.option('-n', '--network-id', type=int, default=None)
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('--json-indent', type=int, default=2)
@click.option('--json-sort-keys/--no-json-sort-keys', default=False)
def export_json(obj, data_dir, network_id, scenario_id, user_id, json_sort_keys, json_indent):
    """ Export a Pywr JSON from Hydra. """
    client = get_logged_in_client(obj, user_id=user_id)
    exporter = PywrHydraExporter.from_network_id(client, network_id, scenario_id)

    data = exporter.get_pywr_data()
    title = data['metadata']['title']

    filename = os.path.join(data_dir, f'{title}.json')
    with open(filename, mode='w') as fh:
        json.dump(data, fh, sort_keys=json_sort_keys, indent=json_indent)

    click.echo(f'Successfully exported "{filename}"! Network ID: {network_id}, Scenario ID: {scenario_id}')


@hydra_app(category='model', name='Run Pywr')
@cli.command()
@click.pass_obj
@click.option('-n', '--network-id', type=int, default=None)
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('--output-frequency', type=str, default=None)
@click.option('--solver', type=str, default=None)
@click.option('--check-model/--no-check-model', default=True)
def run(obj, network_id, scenario_id, user_id, output_frequency, solver, check_model):
    """ Export, run and save a Pywr model from Hydra. """
    client = get_logged_in_client(obj, user_id=user_id)
    run_network_scenario(client, network_id, scenario_id, output_frequency=output_frequency,
                         solver=solver, check_model=check_model)


def run_network_scenario(client, network_id, scenario_id, output_frequency=None, solver=None, check_model=True):
    runner = PywrHydraRunner.from_network_id(client, network_id, scenario_id,
                                             output_resample_freq=output_frequency)

    runner.load_pywr_model(solver=solver)
    runner.run_pywr_model(check=check_model)
    runner.save_pywr_results(client)

    click.echo(f'Pywr model run success! Network ID: {network_id}, Scenario ID: {scenario_id}')


@hydra_app(category='network_utility', name='Step model')
@cli.command()
@click.pass_obj
@click.option('-n', '--network-id', type=int, default=None)
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('-u', '--user-id', type=int, default=None)
def step_model(obj, network_id, scenario_id, user_id):
    client = get_logged_in_client(obj, user_id=user_id)
    utils.apply_final_volumes_as_initial_volumes(client, network_id, scenario_id)
    utils.progress_start_end_dates(client, network_id, scenario_id)


@hydra_app(category='network_utility', name='Apply initial volumes')
@cli.command()
@click.pass_obj
@click.option('-n', '--network-id', type=int, default=None)
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('-t', '--target-network-ids', multiple=True, type=int, default=None)
@click.option('-t', '--target-scenario-ids', multiple=True, type=int, default=None)
def apply_initial_volumes_to_other_networks(obj, network_id, scenario_id, user_id, target_network_ids,
                                            target_scenario_ids):

    client = get_logged_in_client(obj, user_id=user_id)

    for target_network_id, target_scenario_id in zip(target_network_ids, target_scenario_ids):
        utils.apply_final_volumes_as_initial_volumes(client, target_network_id, target_scenario_id,
                                                     source_network_id=network_id, source_scenario_id=scenario_id)



@cli.command()
@click.pass_obj
@click.argument('docker-image', type=str)
def register(obj, docker_image):
    """ Register the app with the Hydra installation. """
    plugins = make_plugins(cli, 'hydra-pywr', docker_image=docker_image)
    app_name = docker_image.replace('/', '-').replace(':', '-')
    write_plugins(plugins, app_name)


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
