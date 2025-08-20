import pandas as pd
import logging
import json
from pywrparser.utils import parse_reference_key
import os
import re
from . import ResultsProcessor

log = logging.getLogger(__name__)

class HydraResultsProcessor(ResultsProcessor):
    """Process and store results in MongoDB.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_resample_freq = None
        self.hydra_network = kwargs['hydra_network']
        self.hydra_client = kwargs['hydra_client']
        self.hydra_template = kwargs.get('hydra_template', None)
        self.hydra_attributes = kwargs['hydra_attributes']
        
        self.attr_name_map = self.make_attr_name_map()
        self.attr_unit_map = self.make_attr_unit_map()

    def save(self):
       # Ensure all the results from previous run are removed.
        self._delete_resource_scenarios()

        # Convert the scenario from JSONObject to normal dict
        # This is required to ensure that the complete nested structure of dicts
        # is properly converted to JSONObjects by the client.
        scenario = self._copy_scenario()

        # First add any new attributes required
        attribute_names = []
        for recorder in self.df_recorders:
            attribute_names.append(self._get_attribute_name_from_recorder(recorder, is_dataframe=True))
        for recorder in self.non_df_recorders:
            attribute_names.append(self._get_attribute_name_from_recorder(recorder))

        attribute_names = set(attribute_names)
        attributes = []
        for attribute_name in attribute_names:
            attributes.append({
                'name': attribute_name,
                'description': '',
                'project_id': self.hydra_network.project_id,
                'dimension_id' : self.attr_dimension_map.get(attribute_name)
            })

        # The response attributes have ids now.
        response_attributes = []
        try:
            response_attributes = self.hydra_client.add_attributes(attrs=attributes)
        except Exception as e:
            if hasattr(e, 'message') and 'permission denied' in e.message.lower():
                for a in attributes:
                    a['project_id'] = None
                    a['network_id'] = self.hydra_network['id']
                response_attributes = self.hydra_client.add_attributes(attrs=attributes)


        # Update the attribute mapping
        self.hydra_attributes.update({attr.id: attr for attr in response_attributes})

        for resource_scenario in self.generate_array_recorder_resource_scenarios():
            scenario['resourcescenarios'].append(resource_scenario)

        for i in range(0, len(scenario['resourcescenarios']), 100):
                chunk = scenario['resourcescenarios'][i:i+100]
                log.info('Saving %s datasets', len(chunk))
                self.hydra_client.bulk_update_resourcedata(
                    scenario_ids=[scenario['id']],
                    resource_scenarios=chunk
                )

        self.flush()

        log.info("Results stored to: %s", self.results_location)

        self.save_results_to_s3()

        self.zip_results()

    def _copy_scenario(self):
        # Now construct a scenario object
        scenario = self.hydra_network.scenarios[0]
        new_scenario = {k: v for k, v in scenario.items() if k != 'resourcescenarios'}

        new_scenario['resourcescenarios'] = []
        return new_scenario

    def _delete_resource_scenarios(self):
        scenario = self.hydra_network.scenarios[0]

        ra_is_var_map = {ra['id']: ra['attr_is_var'] for ra in self.hydra_client.get_resource_attributes(ref_key="network", ref_id=self.hydra_network['id'])}
        ra_to_delete = []

        # Compile a list of resource attributes to delete
        for resource_scenario in scenario['resourcescenarios']:
            ra_id = resource_scenario['resource_attr_id']
            ra_is_var = ra_is_var_map.get(ra_id, 'N')
            if ra_is_var == 'Y':
                ra_to_delete.append(ra_id)

        # Now delete them all
        self.hydra_client.delete_resource_scenarios(scenario_id=scenario['id'], resource_attr_ids=ra_to_delete, quiet=True)

    def generate_array_recorder_resource_scenarios(self):
        """ Generate resource scenario data from NumpyArrayXXX recorders. """
        if self.df_recorders is None:
            log.warning('No array recorders defined, results not saved to Hydra.')
            return None

        #get a mapping from recorder names to resource attribute IDs
        self.df_recorder_ra_id_map = self.add_resource_attributes(self.df_recorders, is_dataframe=True)
        self.non_df_recorder_ra_id_map = self.add_resource_attributes(self.non_df_recorders, is_dataframe=False)
    
        for recorder in self.df_recorders:
            recorder_data = self.process_df_recorder(recorder)

            if recorder_data is None:
                continue

            resource_scenario = self._make_recorder_resource_scenario(recorder,
                                                                      recorder_data['data'],
                                                                      self.df_recorder_ra_id_map[recorder.name],
                                                                      recorder_data['data_type'],
                                                                      is_timeseries=recorder_data['is_timeseries'],
                                                                      is_dataframe=True)

            if resource_scenario is None:
                continue

            yield resource_scenario

        for recorder in self.non_df_recorders:
            recorder_data = self.process_non_df_recorder(recorder)

            if recorder_data is None:
                continue

            resource_scenario = self._make_recorder_resource_scenario(recorder,
                                                                      recorder_data['data'],
                                                                      self.non_df_recorder_ra_id_map[recorder.name+'_value'],
                                                                      recorder_data['data_type'],
                                                                      is_timeseries=False,
                                                                      is_dataframe=False)

            if resource_scenario is None:
                continue

            yield resource_scenario

    def make_dataset_resource_scenario(self, name, value, data_type, resource_attribute_id,
                                        unit_id=None, encode_to_json=False, metadata={}):
        """ A helper method to make a dataset, resource attribute and resource scenario. """
        import json

        if data_type.lower() in ("descriptor", "scalar"):
            encode_to_json = False

        metadata['json_encoded'] = encode_to_json

        datasetvalue = json.dumps(value) if encode_to_json is True else value

        if data_type == 'scalar':
            try:
                float(datasetvalue)
            except:
                datasetvalue = -999

        # Create a dataset representing the value
        dataset = {
            'name': name,
            'value': datasetvalue,
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

    def _make_recorder_resource_scenario(self, recorder, value, resource_attribute_id, data_type, is_timeseries=False, is_dataframe=False):
        # Get the attribute and its ID
        attribute_name = self._get_attribute_name_from_recorder(recorder, is_dataframe=is_dataframe)

        attribute = self._get_attribute_from_name(attribute_name)

        unit_id = self.attr_unit_map.get(attribute.id)

        metadata = {}
        if attribute_name.find('simulated_') == 0:
            metadata['yAxisLabel'] = attribute_name.split('_')[1]
            if is_timeseries is True:
                metadata['xAxisLabel'] = 'Time'

        resource_scenario = self.make_dataset_resource_scenario(attribute_name,
                                                                 value,
                                                                 data_type,
                                                                 resource_attribute_id,
                                                                 unit_id=unit_id,
                                                                 encode_to_json=False,
                                                                 metadata=metadata)
        return resource_scenario

    def add_resource_attributes(self, recorders, is_dataframe):
        """
            Identify new resource attribtues which need adding to the network, and add them prior to adding the data
            Return a mapping from the recorder name to the new Resource Attr ID.
        """

        resource_attributes_to_add = []
        self.recorder_ra_map = {}
        self.recorder_ra_id_map={}

        for recorder in recorders:

            resource_attribute_id=None
            resource_type = 'NETWORK'
            resource_id = self.hydra_network['id']
            attribute_name = self._get_attribute_name_from_recorder(
                recorder,
                is_dataframe=is_dataframe
            )
            recorder_name = recorder.name
            if attribute_name.endswith('value'):
                recorder_name = recorder.name + '_value'

            attribute = self._get_attribute_from_name(attribute_name)

            try:
                recorder_node = self._get_pywr_node_from_recorder(recorder)
            except AttributeError:
                recorder_node=None

            if recorder_node is None:
                for network_ra in self.hydra_network['attributes']:
                    if network_ra['attr_id'] == attribute['id']:
                        resource_attribute_id = network_ra['id']
            else:
                resource_attribute_id = None

                try:
                    resource_attribute_id = self._get_resource_attribute_id(recorder_node.name,
                                                                            attribute_name)
                except ValueError:
                    log.info("Unable to find resource attribute for node {} and attribute {}. Trying parent node.".format(recorder_node.name, attribute_name))

                if resource_attribute_id is None:

                    try:
                        if hasattr(recorder_node, 'parent') and recorder_node.parent is not None:
                             resource_attribute_id = self._get_resource_attribute_id(recorder_node.parent.name,attribute_name)
                        else:
                            log.info("Node {} does not have a parent, and the attribute {} is not defined for it.".format(recorder_node.name, attribute_name))
                    except ValueError:
                        log.info("Unable to find resource attribute for node {} and attribute {}. Trying parent node.".format(recorder_node.name, attribute_name))

                if resource_attribute_id is None:

                    hydra_node = self._get_hydra_node_from_recorder(recorder, pywr_node=recorder_node)
                    if hydra_node is not None:
                        resource_id = hydra_node['id']
                        resource_type = 'NODE'

            if resource_attribute_id is not None:
                self.recorder_ra_id_map[recorder_name] = resource_attribute_id
                continue
            else:

                # Try to get the resource attribute
                resource_attributes_to_add.append(dict(resource_type=resource_type,
                                                                resource_id=resource_id,
                                                                ref_key=resource_type,
                                                                ref_id=resource_id,
                                                                attr_id=attribute['id'],
                                                                attr_is_var='Y',
                                                                error_on_duplicate='N'))
                self.recorder_ra_map[(resource_type, resource_id, attribute['id'])] = recorder_name

        if len(resource_attributes_to_add) > 0:
            log.info('Adding %s new resource attributes', len(resource_attributes_to_add))
            self._add_resource_attributes(resource_attributes_to_add)
        return self.recorder_ra_id_map

    def _add_resource_attributes(self, resource_attributes_to_add):
        from itertools import islice

        def chunked_iterable(iterable, size):
            iterator = iter(iterable)
            while chunk := list(islice(iterator, size)):
                yield chunk

        for chunk in chunked_iterable(resource_attributes_to_add, 100):
            returned_new_ids = self.hydra_client.add_resource_attributes(resource_attributes=chunk)

            #based on the data returned by add_resource_attributes, we need to reverse the map
            #to make lookups easier, so the key is the (resource_id, attr_id) tuple and the value is the new resource attribute ID
            new_ids = {}
            for ra_id, metadata in returned_new_ids.items():
                new_ids[tuple(metadata)] = ra_id

            for i, new_ra in enumerate(chunk):
                key = (new_ra["resource_id"], new_ra["attr_id"])
                if key not in new_ids:
                    continue
                new_ra['id'] = new_ids[key]
                if new_ra['resource_type'] == 'NETWORK':
                    # We need to set the network ID for the resource attribute
                    new_ra['network_id'] = new_ra['resource_id']
                if new_ra['resource_type'] == 'NODE':
                    # We need to set the node ID for the resource attribute
                    new_ra['node_id'] = new_ra['resource_id']
                recorder_name = self.recorder_ra_map[
                    (new_ra['ref_key'], new_ra.get('node_id', new_ra.get('network_id')), new_ra['attr_id'])
                ]
                self.recorder_ra_id_map[recorder_name] = new_ra['id']

    def make_attr_unit_map(self):
        """
            Create a mapping between an attribute ID and its unit, as defined
            in the template
        """
        if self.hydra_template is None:
            log.info("Cannot make unit map. Template is Null")
            return
        attr_unit_map = {}
        for templatetype in self.hydra_template.templatetypes:
            for typeattr in templatetype.typeattrs:
                attr_unit_map[typeattr.attr_id] = typeattr.unit_id

        return attr_unit_map

    def make_attr_name_map(self):
        """
            Create a mapping between an attribute's name and itself, as defined
            in the template
        """
        attr_name_map = {}
        for templatetype in self.hydra_template.templatetypes:
            for typeattr in templatetype.typeattrs:
                attr = self.hydra_attributes[typeattr.attr_id]
                attr_name_map[attr.name] = attr
                #populate the dimension mapping
                self.attr_dimension_map[attr.name] = attr.dimension_id

        return attr_name_map


    def _get_attribute_name_from_recorder(self, recorder, is_dataframe=False):
        scalar_suffix = "value"
        simulated_prefix = "simulated"

        if recorder.name is None:
            attribute_name = recorder.__class__.__name__
        else:
            if ':' in recorder.name:
                attribute_name = recorder.name.rsplit(':')[-1]
            elif '.' in recorder.name:
                attribute_name = recorder.name.split('.')[0]
            else:
                attribute_name = recorder.name

        if not attribute_name.startswith(simulated_prefix):
            attribute_name = f'{simulated_prefix}_{attribute_name}'

        if is_dataframe is False and not attribute_name.endswith(scalar_suffix):
            attribute_name = f"{attribute_name}_{scalar_suffix}"

        return attribute_name

    def _get_resource_attribute_id(self, node_name, attribute_name):

        attribute = self._get_attribute_from_name(attribute_name)
        attribute_id = attribute['id']

        node = self.node_lookup.get(node_name)
        if node is not None:
            resource_attributes = node['attributes']
        else:
            raise ValueError('Node name "{}" not found in network data.'.format(node_name))
        node_attribute = self.node_attr_lookup[node.name].get(attribute_id)
        if node_attribute is not None:
            return node_attribute['id']
        else:
            raise ValueError('No resource attribute for node "{}" and attribute "{}" found.'.format(node_name, attribute))

    def _get_attribute_from_name(self, name):
        dimension_id = self.attr_dimension_map.get(name)

        for attribute_id, attribute in self.hydra_attributes.items():
            if attribute['name'].lower() == name.lower() and attribute.get('dimension_id') == dimension_id:
                return attribute
        raise ValueError('No attribute with name "{}" found.'.format(name))
