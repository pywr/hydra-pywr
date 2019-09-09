import pandas
import logging
logger

def get_final_volumes(client, scenario_id):

    attribute = client.get_attribute_by_name_and_dimension('simulated_volume')
    scenario = client.get_scenario(scenario_id, include_data=False)
    nodes = client.get_nodes(scenario['network_id'])

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

            yield node, {c: v for c, v in zip(df.columns, df.iloc[-1, :])}


def apply_final_volumes_as_initial_volumes(client, source_scenario_id):

    attribute = client.get_attribute_by_name_and_dimension('initial_volume')

    network_id_map = {}
    node_data = []
    for source_node, new_volumes in get_final_volumes(client, source_scenario_id):

        for column, new_initial_volume in new_volumes.items():
            # Naming convention for ensemble names in this setup contains the scenario_id
            _, target_scenario_id = column.split(':')
            target_scenario_id = int(target_scenario_id.strip())

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
