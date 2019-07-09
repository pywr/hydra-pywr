import pandas


def get_final_volumes(client, network_id, scenario_id):

    attribute = client.get_attribute_by_name_and_dimension('simulated_volume')
    nodes = client.get_nodes(network_id)

    for node in nodes:
        # Fetch the node's data
        resource_scenarios = client.get_resource_data('NODE', node['id'], scenario_id)
        for resource_scenario in resource_scenarios:
            resource_attribute_id = resource_scenario['resource_attr_id']
            resource_attribute = client.get_resource_attribute(resource_attribute_id)

            if resource_attribute['attr_id'] != attribute['id']:
                continue

            attribute_name = attribute['name']
            assert attribute_name == 'simulated_volume'

            dataset = resource_scenario['dataset']

            if dataset['type'].lower() != 'dataframe':
                continue  # Skip non-datasets

            df = pandas.read_json(dataset['value'])
            yield node, df.iloc[-1, 0]


def apply_final_volumes_as_initial_volumes(client, network_id, scenario_id, source_network_id=None,
                                           source_scenario_id=None):

    attribute = client.get_attribute_by_name_and_dimension('initial_volume')

    if source_network_id is None:
        source_network_id = network_id

    if source_scenario_id is None:
        source_scenario_id = scenario_id

    node_data = {}
    for node, new_volume in get_final_volumes(client, source_network_id, source_scenario_id):

        # Fetch the node's data
        resource_scenarios = client.get_resource_data('NODE', node['id'], scenario_id)
        for resource_scenario in resource_scenarios:
            resource_attribute_id = resource_scenario['resource_attr_id']
            resource_attribute = client.get_resource_attribute(resource_attribute_id)

            if resource_attribute['attr_id'] != attribute['id']:
                continue  # Skip the wrong attribute data

            dataset = resource_scenario['dataset']
            # Update the volume
            dataset['value'] = new_volume

            node_data[node['name']] = {
                'node_id': node['id'],
                'resource_attribute_id': resource_attribute['id'],
                'dataset': dataset,
            }

    # Now update the database with the new data
    for node_name, data in node_data.items():
        client.add_data_to_attribute(scenario_id, data['resource_attribute_id'], data['dataset'])


def progress_start_end_dates(client, network_id, scenario_id):

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
