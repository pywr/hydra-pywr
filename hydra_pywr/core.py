import json


class BasePywrHydra:
    _node_attribute_component_affix = '__'
    _node_attribute_component_delimiter = ':'

    ignore_json_encoding_data_types = ['descriptor', 'scalar']

    def __init__(self):
        # Default internal variables
        self.next_resource_attribute_id = -1

    def _make_dataset_resource_scenario(self, name, value, data_type, resource_attribute_id,
                                        unit_id=None, encode_to_json=False, metadata={}):
        """ A helper method to make a dataset, resource attribute and resource scenario. """

        if data_type.lower() in self.ignore_json_encoding_data_types:
            encode_to_json = False

        metadata['json_encoded'] = encode_to_json

        # Create a dataset representing the value
        dataset = {
            'name': name,
            'value': json.dumps(value) if encode_to_json is True else value,
            "hidden": "N",
            "type": data_type,
            "unit_id": unit_id,
            "metadata": json.dumps(metadata)
        }

        # Create a resource scenario linking the dataset to the scenario
        resource_scenario = {
            'resource_attr_id': resource_attribute_id,
            'dataset': dataset
        }

        # Finally return resource attribute and resource scenario
        return resource_scenario

    def _make_dataset_resource_attribute_and_scenario(self, name, value, data_type,
                                                      attribute_id, unit_id=None, **kwargs):
        """ A helper method to make a dataset, resource attribute and resource scenario. """
        resource_attribute_id = self.next_resource_attribute_id
        self.next_resource_attribute_id -= 1

        resource_scenario = self._make_dataset_resource_scenario(name,
                                                                 value,
                                                                 data_type,
                                                                 resource_attribute_id,
                                                                 unit_id=unit_id,
                                                                 **kwargs)

        # Create a resource attribute linking the resource scenario to the node
        resource_attribute = {
            'id': resource_attribute_id,
            'attr_id': attribute_id,
            'attr_is_var': 'N'
        }

        # Finally return resource attribute and resource scenario
        return resource_attribute, resource_scenario

    @classmethod
    def is_component_a_node_attribute(cls, component_name, node_name=None):
        """Test whether a component's name should be inferred as a node level attribute in Hydra. """

        if node_name is None:
            # This should probably be done with regex
            if cls._node_attribute_component_delimiter not in component_name:
                return False

            prefix, _ = component_name.split(cls._node_attribute_component_delimiter, 1)
            return prefix.startswith(cls._node_attribute_component_affix) and \
                prefix.endswith(cls._node_attribute_component_affix)
        else:
            # Test that it is exactly true
            prefix = '{affix}{name}{affix}'.format(affix=cls._node_attribute_component_affix, name=node_name)
            return component_name.startswith(prefix)

    @classmethod
    def make_node_attribute_component_name(cls, node_name, attribute_name):
        """Return the component name to use in Pywr for node level attribute. """
        prefix = '{affix}{name}{affix}'.format(affix=cls._node_attribute_component_affix, name=node_name)
        return cls._node_attribute_component_delimiter.join((prefix, attribute_name))
