import os
import pandas
from urllib.parse import urlparse
from hydra_network_utils import data as data_utils

import logging
log = logging.getLogger(__name__)

def import_dataframe(client, dataframe, scenario_id, attribute_id, create_new=False, data_type='PYWR_DATAFRAME', column=None):

    scenario = client.get_scenario(scenario_id, include_data=False)
    network_id = scenario['network_id']

    data_utils.import_dataframe(client, dataframe, network_id, scenario_id, attribute_id, column,
                                         create_new=create_new, data_type=data_type)


def clone_scenarios(client, scenarios_ids):
    for scenario_id in scenarios_ids:
        scenario = client.clone_scenario(scenario_id)
        yield scenario['id']


def get_final_volumes(client, scenario_id):

    attribute = None
    scenario = client.get_scenario(scenario_id, include_data=False)
    nodes = client.get_nodes(scenario['network_id'])

    for node in nodes:
        # Fetch the node's data
        resource_scenarios = client.get_resource_data('NODE', node['id'], scenario_id)
        for resource_scenario in resource_scenarios:
            resource_attribute_id = resource_scenario['resource_attr_id']
            resource_attribute = client.get_resource_attribute(resource_attribute_id)
            this_attr_id = resource_attribute['attr_id']

            if attribute is None:
                possible_attr = client.get_attribute_by_id(this_attr_id)
                if possible_attr.name == 'simulated_volume':
                    attribute = possible_attr


            if attribute is None or this_attr_id != attribute['id']:
                continue

            attribute_name = attribute['name']
            assert attribute_name == 'simulated_volume'

            dataset = resource_scenario['dataset']

            if dataset['type'].lower() != 'dataframe':
                continue  # Skip non-datasets

            df = pandas.read_json(dataset['value'])

            yield node, {c: v for c, v in zip(df.columns, df.iloc[-1, :])}


def apply_final_volumes_as_initial_volumes(client, source_scenario_id, target_scenario_ids):

    scenario = client.get_scenario(source_scenario_id, include_data=False)

    #Find the correct initial volume
    attribute = None
    network_attributes = client.get_all_network_attributes(scenario['network_id'])
    for network_attr in network_attributes:
        if network_attr.name == 'initial_volume':
            attribute = network_attr
            break

    network_id_map = {}
    node_data = []
    for source_node, new_volumes in get_final_volumes(client, source_scenario_id):

        for target_scenario_id, (column, new_initial_volume) in zip(target_scenario_ids, new_volumes.items()):

            # Cache the network_ids to prevent repeat calls to get_source (which is expensive)
            if target_scenario_id in network_id_map:
                target_network_id = network_id_map[target_scenario_id]
            else:
                target_network_id = client.get_scenario(target_scenario_id, include_data=False)['network_id']
                network_id_map[target_scenario_id] = target_network_id

            # Find the equivalent target_node
            target_node = client.get_node_by_name(target_network_id, source_node['name'])

            # Fetch the target node's data
            resource_scenarios = client.get_resource_data('NODE', target_node['id'], target_scenario_id)
            for resource_scenario in resource_scenarios:
                resource_attribute_id = resource_scenario['resource_attr_id']
                resource_attribute = client.get_resource_attribute(resource_attribute_id)

                if resource_attribute['attr_id'] != attribute['id']:
                    continue  # Skip the wrong attribute data

                dataset = resource_scenario['dataset']
                # Update the volume
                dataset['value'] = new_initial_volume

                node_data.append({
                    'node_id': target_node['id'],
                    'resource_attribute_id': resource_attribute['id'],
                    'dataset': dataset,
                    'scenario_id': target_scenario_id
                })

    # Now update the database with the new data
    for data in node_data:
        client.add_data_to_attribute(data['scenario_id'], data['resource_attribute_id'], data['dataset'])


def progress_start_end_dates(client, scenario_id):

    network_id = client.get_scenario(scenario_id, include_data=False)['network_id']

    resource_scenarios = client.get_resource_data('NETWORK', network_id, scenario_id)

    timestepper_data = {}

    attributes = ['timestepper.start', 'timestepper.end']

    for resource_scenario in resource_scenarios:
        resource_attribute_id = resource_scenario['resource_attr_id']
        resource_attribute = client.get_resource_attribute(resource_attribute_id)

        attribute = client.get_attribute_by_id(resource_attribute['attr_id'])
        attribute_name = attribute['name']

        if attribute_name not in attributes:
            continue

        dataset = resource_scenario['dataset']

        timestepper_data[attribute_name] = {
            'resource_attribute_id': resource_attribute['id'],
            'dataset': dataset,
        }

    current_start = timestepper_data['timestepper.start']['dataset']['value']
    current_end = timestepper_data['timestepper.end']['dataset']['value']

    current_start = pandas.to_datetime(current_start)
    current_end = pandas.to_datetime(current_end)

    new_start = current_end + pandas.to_timedelta('1D')
    new_end = current_end + (current_end - current_start) + pandas.to_timedelta('1D')

    timestepper_data['timestepper.start']['dataset']['value'] = new_start.to_pydatetime().date().isoformat()
    timestepper_data['timestepper.end']['dataset']['value'] = new_end.to_pydatetime().date().isoformat()

    # for data in timestepper_data.values():
    #     current = data['dataset']['value']
    #     new = pandas.to_datetime(current) + pandas.to_timedelta(timedelta)
    #     new = new.to_pydatetime().date().isoformat()
    #     data['dataset']['value'] = new

    # Now update the database with the new data
    for data in timestepper_data.values():
        client.add_data_to_attribute(scenario_id, data['resource_attribute_id'], data['dataset'])

"""
  Compatibility patches: these update the Pywr data output of
  get_pywr_data to replace deprecated syntax with that of current
  Pywr versions.
"""
def unnest_parameter_key(param_data, key="pandas_kwargs"):
    """
        Relocates all keys inside parameters' <key> arg
        to the top level of that parameter and removes the
        original <key>.
    """
    if key in param_data:
        for k, v in param_data[key].items():
            param_data[k] = v
        del param_data[key]

    return param_data

def add_interp_kwargs(param_data):
    """
        Replaces the deprecated `kind` key of interpolatedvolume
        parameters with the nested `interp_kwargs` key.
    """
    ptype = "interpolatedvolume"
    new_key = "interp_kwargs"
    if param_data["type"].lower().startswith(ptype) and "kind" in param_data:
        param_data[new_key] = {"kind": param_data["kind"]}
        del param_data["kind"]

    return param_data

def file_to_s3(elem_data, s3prefix):
    """
      Transforms local url references to point to s3 storage
    """
    if "url" not in elem_data:
        return
    url = elem_data["url"]
    path, filename = os.path.split(url)
    s3url = os.path.join(s3prefix, filename)
    elem_data["url"] = s3url


def retrieve_url(url, urldir):
    import shutil
    from urllib.request import urlopen

    if not os.path.exists(urldir):
        try:
            os.makedirs(urldir)
        except OSError as err:
            raise OSError(f"Unable to create URL retrieval directory at {urldir}: {err}")
    elif not os.path.isdir(urldir):
        raise OSError(f"Destination '{urldir}' is not a directory")

    filename = os.path.basename(url)
    filedest = os.path.join(urldir, filename)
    log.info(f"Retrieving {url} to {filedest} ...")

    with urlopen(url) as resp, open(filedest, "wb") as fp:
        shutil.copyfileobj(resp, fp)

    log.info(f"Retrieved {filedest} ({os.stat(filedest).st_size} bytes)")
    return filedest


def retrieve_s3(s3path, datadir):
    try:
        import s3fs
    except ImportError:
        log.error("Retrieval from S3 url requires the s3fs module")
        raise

    u = urlparse(s3path)

    datadir = "data"
    filepath = f"{u.netloc}{u.path}"
    filedest = os.path.join(datadir, filepath)

    if not os.path.exists(datadir):
        try:
            os.makedirs(datadir)
        except OSError as err:
            raise OSError(f"Unable to create S3 retrieval directory at {datadir}: {err}")

    fs = s3fs.S3FileSystem(anon=True)
    log.info(f"Retrieving {s3path} to {filedest} ...")
    fs.get(filepath, filedest)
    log.info(f"Retrieved {filedest} ({os.stat(filedest).st_size} bytes)")

    return filedest
