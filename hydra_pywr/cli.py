import click
import json
import os
from hydra_client.connection import JSONConnection
from .exporter import PywrHydraExporter
from .runner import PywrHydraRunner
from .template import register_template, unregister_template, migrate_network_template, TemplateExistsError
from . import utils
from hydra_client.click import hydra_app, make_plugins, write_plugins
import pandas

from hydra_pywr_common.types.network import(
    PywrNetwork,
    PywrIntegratedNetwork
)

from hydra_pywr_common.lib.writers import(
    PywrJsonWriter,
    PywrHydraWriter,
    PywrHydraIntegratedWriter,
    PywrIntegratedJsonWriter,
    IntegratedOutputWriter
)

from hydra_pywr_common.lib.runners import(
    IntegratedModelRunner,
    write_output
)

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
@cli.command(name='import', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('--filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-p', '--project-id', type=int)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('--template-id', type=int)
@click.option('--projection', type=str, default=None)
@click.option('--run/--no-run', default=False)
@click.option('--solver', type=str, default=None)
@click.option('--check-model/--no-check-model', default=True)
@click.option('--ignore-type-errors', is_flag=True, default=False)
def import_json(obj, filename, project_id, user_id, template_id, projection, run, solver, check_model, ignore_type_errors, *args):
    """ Import a Pywr JSON file into Hydra. """
    click.echo(f'Beginning import of "{filename}" to Project ID: {project_id}')

    if filename is None:
        raise Exception("No file specified")

    if project_id is None:
        raise Exception("No project specified")

    if template_id is None:
        raise Exception("No template specified")

    pnet = PywrNetwork.from_source_file(filename)
    hwriter = PywrHydraWriter(pnet, user_id=user_id, template_id=template_id, project_id=project_id)
    hwriter.build_hydra_network(projection)


@hydra_app(category='import', name='Import Integrate Pynsim JSON from combined file')
@cli.command(name='integrated-import', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('--filename', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-p', '--project-id', type=int)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('--water-template-id', type=int)
@click.option('--energy-template-id', type=int)
@click.option('--projection', type=str, default=None)
def integrated_import_combinedjson(obj, filename, project_id, user_id, water_template_id, energy_template_id, projection, *args):
    """ Import a Pynsim combined JSON file into Hydra. """
    if filename is None:
        raise Exception("No file specified")

    if project_id is None:
        raise Exception("No project specified")

    if not (water_template_id and energy_template_id):
        raise Exception("No template specified")

    pin = PywrIntegratedNetwork.from_combined_file(filename)
    writer = PywrHydraIntegratedWriter(pin, user_id=user_id, water_template_id=water_template_id, energy_template_id=energy_template_id, project_id=project_id)
    writer.build_hydra_integrated_network(projection=projection)
    breakpoint()
    writer.add_network_to_hydra()

    click.echo(f"Imported {filename} to Project ID: {project_id}")


@hydra_app(category='export', name='Export to Pywr JSON')
@cli.command(name='export', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('--data-dir', default='/tmp')
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('--json-indent', type=int, default=2)
@click.option('--json-sort-keys/--no-json-sort-keys', default=False)
def export_json(obj, data_dir, scenario_id, user_id, json_sort_keys, json_indent):
    """ Export a Pywr JSON from Hydra. """
    client = get_logged_in_client(obj, user_id=user_id)
    exporter = PywrHydraExporter.from_scenario_id(client, scenario_id)

    network_id = exporter.data.id

    data = exporter.get_pywr_data("energy")

    pnet = PywrNetwork(data)
    writer = PywrJsonWriter(pnet)
    output = writer.as_dict()

    outfile = os.path.join(data_dir, f"{pnet.title.replace(' ', '_')}.json")
    with open(outfile, mode='w') as fp:
        json.dump(output, fp, sort_keys=json_sort_keys, indent=2)

    click.echo(f"Network: {network_id}, Scenario: {scenario_id} exported to `{outfile}`")

@hydra_app(category='export', name='Export to IntegratedPywrJSON')
@cli.command(name='integrated-export', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('--data-dir', default='/tmp')
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('--json-sort-keys/--no-json-sort-keys', default=False)
def integrated_export(obj, data_dir, scenario_id, user_id, json_sort_keys):
    """ Export an integrated Pywr Water/Energy Network from Hydra to Pynsim json. """
    client = get_logged_in_client(obj, user_id=user_id)
    w_exporter = PywrHydraExporter.from_scenario_id(client, scenario_id, index=0)
    e_exporter = PywrHydraExporter.from_scenario_id(client, scenario_id, index=1)

    network_id = w_exporter.data.id

    water_data = w_exporter.get_pywr_data("water")
    water_template_id = w_exporter.template["id"]
    water_net = PywrNetwork(water_data)
    energy_data = e_exporter.get_pywr_data("energy")
    energy_template_id = e_exporter.template["id"]
    energy_net = PywrNetwork(energy_data)

    config = w_exporter.get_integrated_config()
    pin = PywrIntegratedNetwork(water_net, energy_net, config)
    writer = PywrIntegratedJsonWriter(pin)
    output = writer.as_dict()
    dests = writer.write_as_pynsim()

    for engine in dests["engines"]:
        click.echo(f"{engine} output written to {dests[engine]['file']}")

    outfile = os.path.join(data_dir, "combined_export.json")
    with open(outfile, mode='w') as fp:
        json.dump(output, fp, sort_keys=json_sort_keys, indent=2)

    click.echo(f"Network: {network_id}, Scenario: {scenario_id} exported to `{outfile}`")
    dests["water"]["template_id"] = water_template_id
    dests["energy"]["template_id"] = energy_template_id
    return dests


@hydra_app(category='model', name='Run Integrated Model')
@cli.command(name="integrated-run", context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_context
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('--output-frequency', type=str, default=None)
@click.option('--solver', type=str, default=None)
@click.option('--check-model/--no-check-model', default=True)
@click.option('--data-dir', default=None)
def integrated_run(ctx, scenario_id, user_id, output_frequency, solver, check_model, data_dir):
    dests = ctx.invoke(integrated_export, user_id=user_id, scenario_id=scenario_id)
    pynsim_config = dests["config"]
    imr = IntegratedModelRunner(pynsim_config)
    imr.run_subprocess()

    for engine in dests["engines"]:
        h5output = f"results/{engine}_Outputs.h5"
        write_output(f"Importing results for {engine} engine from {h5output}...")
        template_id = dests[engine]["template_id"]
        iow = IntegratedOutputWriter(scenario_id, template_id, h5output, engine, user_id=2)
        iow.build_hydra_output()


@hydra_app(category='model', name='Run Pywr')
@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('-t', '--template-id', type=int, default=None)
@click.option('-u', '--user-id', type=int, default=None)
@click.option('--output-frequency', type=str, default=None)
@click.option('--solver', type=str, default=None)
@click.option('--check-model/--no-check-model', default=True)
@click.option('--data-dir', default=None)
def run(obj, scenario_id, template_id, user_id, output_frequency, solver, check_model, data_dir):
    """ Export, run and save a Pywr model from Hydra. """
    client = get_logged_in_client(obj, user_id=user_id)

    if scenario_id is None:
        raise Exception('No scenario specified')

    run_network_scenario(client, scenario_id, template_id, output_frequency=output_frequency,
                         solver=solver, check_model=check_model, data_dir=data_dir)

def run_network_scenario(client, scenario_id, template_id, output_frequency=None, solver=None, check_model=True, data_dir=None):

    runner = PywrHydraRunner.from_scenario_id(client, scenario_id,
                                             template_id=template_id,
                                             output_resample_freq=output_frequency)

    pywr_data = runner.load_pywr_model(solver=solver)

    network_id = runner.data.id

    if data_dir is not None:
        save_pywr_file(pywr_data, data_dir, network_id, scenario_id)

    runner.run_pywr_model(check=check_model)
    runner.save_pywr_results()

    click.echo(f'Pywr model run success! Network ID: {network_id}, Scenario ID: {scenario_id}')


def save_pywr_file(data, data_dir, network_id=None, scenario_id=None):
    """
    Save pywr json data to the specified directory
    """
    if data_dir is None:
        click.echo("No data dir specified. Returning.")

    title = data['metadata']['title']

    #check if the output folder exists and create it if not
    if not os.path.isdir(data_dir):
        #exist_ok sets unix the '-p' functionality to create the whole path
        os.makedirs(data_dir, exist_ok=True)

    filename = os.path.join(data_dir, f'{title}.json')
    with open(filename, mode='w') as fh:
        json.dump(data, fh, sort_keys=True, indent=2)

    click.echo(f'Successfully exported "{filename}"! Network ID: {network_id}, Scenario ID: {scenario_id}')


@hydra_app(category='network_utility', name='Step model')
@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('-n', '--network-id', type=int, default=None)
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('--child-scenario-ids', type=int, default=None, multiple=True)
@click.option('-u', '--user-id', type=int, default=None)
def step_model(obj, network_id, scenario_id, child_scenario_ids, user_id):
    client = get_logged_in_client(obj, user_id=user_id)
    utils.apply_final_volumes_as_initial_volumes(client, scenario_id, child_scenario_ids)
    utils.progress_start_end_dates(client, network_id, scenario_id)


@hydra_app(category='network_utility', name='Apply initial volumes')
@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True))
@click.pass_obj
@click.option('-s', '--scenario-id', type=int, default=None)
@click.option('--child-scenario-ids', type=int, default=None, multiple=True)
@click.option('-u', '--user-id', type=int, default=None)
def apply_initial_volumes_to_other_networks(obj, scenario_id, child_scenario_ids, user_id):
    client = get_logged_in_client(obj, user_id=user_id)
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
@click.option('-u', '--user-id', type=int, default=None)
def step_game(obj, scenario_id, child_scenario_ids, filename, attribute_name, index_col,
              column_name, data_type, create_new, user_id):
    client = get_logged_in_client(obj, user_id=user_id)

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


@cli.command()
@click.pass_obj
@click.argument('docker-image', type=str, default=None)
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
