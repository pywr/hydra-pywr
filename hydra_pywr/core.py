import json
from hydra_base.lib.HydraTypes.Types import Scalar, Array, Descriptor, Dataframe
from hydra_pywr_common.datatypes import PywrParameterPatternReference, PywrParameter
from pywr.schema.fields import ParameterField, ParameterReferenceField, ParameterValuesField, NodeField
from marshmallow.fields import Number, Integer, List
from .nodes import DataFrameField

# TODO move this to template.py
def data_type_from_field(field):
    """ Return the appropriate Hydra DataType for a given node's attribute. """

    if isinstance(field, (ParameterReferenceField, ParameterField)):
        data_type = PywrParameter.tag
    elif isinstance(field, NodeField):
        data_type = Descriptor.tag
    elif isinstance(field, ParameterValuesField):
        # TODO support this data type properly.
        data_type = Scalar.tag
    elif isinstance(field, (Number, Integer)):
        data_type = Scalar.tag
    elif isinstance(field, List):
        data_type = Array.tag
    elif isinstance(field, DataFrameField):
        data_type = Dataframe.tag
    else:
        raise ValueError('No data type found for field: {}'.format(field))

    return data_type.lower()


class BasePywrHydra:
    _node_attribute_component_affix = '__'
    _node_attribute_component_delimiter = ':'

    def __init__(self):
        # Default internal variables
        self.next_resource_attribute_id = -1

    def _make_dataset_resource_scenario(self, name, value, data_type, resource_attribute_id,
                                        encode_to_json=False, unit_id=None):
        """ A helper method to make a dataset, resource attribute and resource scenario. """

        # Create a dataset representing the value
        dataset = {
            'name': name,
            'value': json.dumps(value) if encode_to_json else value,
            "hidden": "N",
            "type": data_type,
            "unit_id": unit_id,
            "metadata": json.dumps({'json_encoded': encode_to_json})
        }

        # Create a resource scenario linking the dataset to the scenario
        resource_scenario = {
            'resource_attr_id': resource_attribute_id,
            'dataset': dataset
        }

        # Finally return resource attribute and resource scenario
        return resource_scenario

    def _make_dataset_resource_attribute_and_scenario(self, name, value, data_type, attribute_id, **kwargs):
        """ A helper method to make a dataset, resource attribute and resource scenario. """

        resource_attribute_id = self.next_resource_attribute_id
        self.next_resource_attribute_id -= 1

        resource_scenario = self._make_dataset_resource_scenario(name, value, data_type, resource_attribute_id, **kwargs)

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
