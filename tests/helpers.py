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
