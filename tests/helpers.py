import json
import hydra_base


def assert_dataset(hydra_data, key, value, decode_from_json=False):
    """ Raise an assertion error if the value is not found as a resource scenario's dataset. """

    for scenario in hydra_data['scenarios']:
        for rs in scenario['resourcescenarios']:
            hydra_value = rs['dataset']['value']
            if decode_from_json:
                # The data in hydra is stored as a JSON encoded string.
                # Load it back to Python types for comparison with Pywr data.
                hydra_value = json.loads(hydra_value)
            if hydra_value == value:
                return
    raise AssertionError('Dataset "{}" value not found in the hydra resource scenario data.'.format(key))


def assert_hydra_pywr(hydra_data, pywr_data):
    """ Compare Hydra and Pywr data for a model and raise an `AssertionError` if they are different.

    This is a convenience function for testing purposes. It compares Hydra data in the
    format suitable to send to an add network request with the Pywr JSON data.

    """

    assert len(hydra_data['nodes']) == len(pywr_data['nodes'])
    assert len(hydra_data['links']) == len(pywr_data['edges'])

    # Check coordinates
    for pywr_node in pywr_data['nodes']:
        for hydra_node in hydra_data['nodes']:
            if pywr_node['name'] == hydra_node['name']:
                try:
                    pywr_coordinate = pywr_node['position']['geographic']
                except KeyError:
                    pass
                else:
                    assert pywr_coordinate == [hydra_node['x'], hydra_node['y']]

    # Ensure that the time-stepper information exists.
    timestepper = pywr_data['timestepper']
    for key, value in timestepper.items():
        assert_dataset(hydra_data, key, value)

    metadata = pywr_data['metadata']
    for key, value in metadata.items():
        if key in ('title', 'description'):
            continue
        assert_dataset(hydra_data, key, value)

    # Ensure that the pywr parameters exist as a hydra dataset
    for component in ('parameters', 'recorders'):
        if component in pywr_data:
            for component_name, component_value in pywr_data[component].items():
                assert_dataset(hydra_data, component_name, component_value, decode_from_json=True)


def assert_identical_pywr_node_data(node1, node2):
    assert sorted(node1.keys()) == sorted(node2.keys())
    for node_key in node1.keys():
        if node_key == 'type':
            # Types can be upper or lower case
            assert node1[node_key].lower() == node2[node_key].lower()
        else:
            assert node1[node_key] == node2[node_key]


def assert_identical_pywr_data(data1, data2):
    """ Assert two Pywr JSON data dictionaries are identical. """
    assert sorted(data1.keys()) == sorted(data2.keys())

    for key in data1.keys():
        if key == 'nodes':
            # The ordering of these lists does not matter to Pywr
            for node1, node2 in zip(sorted(data1[key], key=lambda n: n['name']), sorted(data2[key], key=lambda n: n['name'])):
                assert_identical_pywr_node_data(node1, node2)
        elif key == 'edges':
            # The ordering of these lists does not matter to Pywr
            assert sorted(data1[key]) == sorted(data2[key])
        else:
            assert data1[key] == data2[key]
