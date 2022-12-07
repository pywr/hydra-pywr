from .exporter import PywrHydraExporter
import copy
import pandas
from pywr.model import Model
from pywr.nodes import Node, AggregatedNode
from pywr._core import AbstractStorage
#from pywr_dcopf.core import Generator, Load, Line, Battery
from pywr.parameters import Parameter, DeficitParameter
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayStorageRecorder, NumpyArrayLevelRecorder, \
    NumpyArrayParameterRecorder
from pywr.recorders.progress import ProgressRecorder
from .template import PYWR_ARRAY_RECORDER_ATTRIBUTES
from hydra_pywr_common.types.network import PywrNetwork
from hydra_pywr_common.lib.writers import PywrJsonWriter

import os
import logging
log = logging.getLogger(__name__)


class PywrHydraRunner(PywrHydraExporter):
    """ An extension to `PywrHydraExporter` that adds methods for running a Pywr model. """
    def __init__(self, *args, **kwargs):
        self.output_resample_freq = kwargs.pop('output_resample_freq', None)
        super(PywrHydraRunner, self).__init__(*args, **kwargs)
        self.model = None
        self.pywr_data = None
        self._df_recorders = None
        self._non_df_recorders = None


        self.attr_dimension_map = {}

        self.attr_name_map = self.make_attr_name_map()

        self.make_attr_unit_map()

    def _copy_scenario(self):
        # Now construct a scenario object
        scenario = self.data.scenarios[0]
        new_scenario = {k: v for k, v in scenario.items() if k != 'resourcescenarios'}

        new_scenario['resourcescenarios'] = []
        return new_scenario

    def _delete_resource_scenarios(self):
        scenario = self.data.scenarios[0]

        ra_is_var_map = {ra['id']: ra['attr_is_var'] for ra in self._get_all_resource_attributes()}
        ra_to_delete = []

        # Compile a list of resource attributes to delete
        for resource_scenario in scenario['resourcescenarios']:
            ra_id = resource_scenario['resource_attr_id']
            ra_is_var = ra_is_var_map.get(ra_id, 'N')
            if ra_is_var == 'Y':
                ra_to_delete.append(ra_id)

        # Now delete them all
        self.client.delete_resource_scenarios(scenario['id'], ra_to_delete, quiet=True)

    def export_pywr_data(self):
        """
        Export the pywr data to json
        """
        data = self.get_pywr_data()
        pnet = PywrNetwork(data)
        writer = PywrJsonWriter(pnet)
        self.pywr_data = writer.as_dict()

    def load_pywr_model(self, solver=None):
        """ Run Pywr model from the exported data using the model in self.pywr_data """
        log.info("Running pywr model from memory")
        self.model = Model.load(self.pywr_data, solver=solver)

    def load_pywr_model_from_file(self, pywr_model_path, solver=None):
        """ Rum a Pywr model from the exported data using a json file as specified
        in the pywr_model_path """
        log.info("Running pywr model from file %s", pywr_model_path)
        self.model = Model.load(pywr_model_path, solver=solver)

    def run_pywr_model(self, check=True):
        """ Run a Pywr model from the exported data.

        If no model has been loaded (see `load_pywr_model`) then a load is attempted.
        """
        if self.model is None:
            self.load_pywr_model(solver='glpk-edge')

        model = self.model

        # Add a progress recorder to monitor the run.
        ProgressRecorder(model)

        # Add recorders for monitoring the simulated timeseries of nodes
        try:
            self._add_node_flagged_recorders(model)
        except Exception as e:
            log.exception(e)
        # Add recorders for parameters that are flagged
        try:
            self._add_parameter_flagged_recorders(model)
        except Exception as e:
            log.exception(e)

        df_recorders = []
        non_df_recorders = []
        for recorder in model.recorders:
            if hasattr(recorder, 'to_dataframe'):
                df_recorders.append(recorder)
            if hasattr(recorder, 'values'):
                non_df_recorders.append(recorder)

        if check:
            # Check the model
            model.check()

        # Force a setup regardless of whether the model has been run or setup before
        model.setup()

        max_scenarios = os.environ.get('HYDRA_PYWR_MAX_SCENARIOS', None)
        if max_scenarios is not None:
            nscenarios = len(model.scenarios.combinations)
            if nscenarios > max_scenarios:
                raise RuntimeError(f'Number of scenarios ({nscenarios}) exceeds the maximum limit of {max_scenarios}.')

        # Now run the model.
        run_stats = model.run()

        # Save these for later
        self._df_recorders = df_recorders
        self._non_df_recorders = non_df_recorders

    def _get_resource_attribute_id(self, node_name, attribute_name):

        attribute = self._get_attribute_from_name(attribute_name)
        attribute_id = attribute['id']

        for node in self.data['nodes']:

            if node['name'] == node_name:
                resource_attributes = node['attributes']
                break
        else:
            raise ValueError('Node name "{}" not found in network data.'.format(node_name))

        for resource_attribute in resource_attributes:
            if resource_attribute['attr_id'] == attribute_id:
                return resource_attribute['id']
        else:
            raise ValueError('No resource attribute for node "{}" and attribute "{}" found.'.format(node_name, attribute))

    def _get_attribute_from_name(self, name):
        dimension_id = self.attr_dimension_map.get(name)

        for attribute_id, attribute in self.attributes.items():
            if attribute['name'] == name and attribute.get('dimension_id') == dimension_id:
                return attribute
        raise ValueError('No attribute with name "{}" found.'.format(name))

    def _get_node_from_recorder(self, recorder):
        node = None
        if recorder.name is not None:
            if ':' in recorder.name:
                node_name, _ = recorder.name.rsplit(':', 1)
                node_name = node_name.replace('__', '')
                try:
                    node = recorder.model.nodes[node_name]
                except KeyError:
                    pass

        if node is None:
            try:
                node = recorder.node
            except AttributeError:
                try:
                    node = recorder.parameter.node
                except AttributeError:
                    return None
        return node

    def _get_attribute_name_from_recorder(self, recorder, is_dataframe=False):
        """
            Get the name of a hydra attribute from a pywr recorder.
            IF the recorder is 'flow', then return something like 'simulated_flow'.
            If the recorder is not a dataframe (or it outputs both dataframes and
            non-dataframes', then the use the is_dataframes flag. In this case, the
            attribute name has '_value' added on at the end, resulting in
            'simulated_flow_value', which should be a single scalar value.
        """
        non_timeseries_postfix = 'value'

        if recorder.name is None:
            attribute_name = recorder.__class__
        else:
            if ':' in recorder.name:
                _, attribute_name = recorder.name.rsplit(':', 1)
            elif '.' in recorder.name:
                attribute_name = recorder.name.split('.')[0]
            else:
                attribute_name = recorder.name

        if not attribute_name.startswith('simulated'):
            attribute_name = f'simulated_{attribute_name}'

        if is_dataframe is False:
            attribute_name = f"{attribute_name}_{non_timeseries_postfix}"

        return attribute_name

    def _add_node_flagged_recorders(self, model):

        for node in model.nodes:
            try:
                flags = self._node_recorder_flags[node.name]
            except KeyError:
                flags = {'timeseries': True}  # Default to recording timeseries if not defined.

            for flag, to_record in flags.items():
                if not to_record:
                    continue

                if flag == 'timeseries':
                    #if isinstance(node, (Node, Generator, Load, Line)):
                    if isinstance(node, (Node, AggregatedNode)):
                        name = '__{}__:{}'.format(node.name, 'simulated_flow')
                        NumpyArrayNodeRecorder(model, node, name=name)
                    elif isinstance(node, (AbstractStorage)):
                        name = '__{}__:{}'.format(node.name, 'simulated_volume')
                        NumpyArrayStorageRecorder(model, node, name=name)
                    else:
                        import warnings
                        warnings.warn('Unrecognised node subclass "{}" with name "{}" for timeseries recording. Skipping '
                                      'recording this node.'.format(node.__class__.__name__, node.name),
                                      RuntimeWarning)

                elif flag == 'deficit':
                    if isinstance(node, Node):
                        deficit_parameter = DeficitParameter(model, node)
                        name = '__{}__:{}'.format(node.name, 'simulated_deficit')
                        NumpyArrayParameterRecorder(model, deficit_parameter, name=name)
                    else:
                        import warnings
                        warnings.warn('Unrecognised node subclass "{}" with name "{}" for deficit recording. Skipping '
                                      'recording this node.'.format(node.__class__.__name__, node.name),
                                      RuntimeWarning)

    def _add_parameter_flagged_recorders(self, model):

        for parameter_name, flags in self._parameter_recorder_flags.items():

            if parameter_name in model.parameters:
                p = model.parameters[parameter_name]

                if ':' in p.name:
                    recorder_name = p.name.rsplit(':', 1)
                    recorder_name[1] = 'simulated_' + recorder_name[1]
                    recorder_name = ':'.join(recorder_name)
                else:
                    recorder_name = 'simulated_' + p.name
            else:
                log.critical("Parameter %s is not in the model", parameter_name)
                continue

            self._add_flagged_recoder(model, p, recorder_name, flags)

        for node_name, attribute_recorder_flags in self._inline_parameter_recorder_flags.items():
            node = model.nodes[node_name]
            for attribute_name, flags in attribute_recorder_flags.items():
                p = getattr(node, attribute_name)

                if not isinstance(p, Parameter):
                    continue

                recorder_name = f'__{node_name}__:simulated_{attribute_name}'
                self._add_flagged_recoder(model, p, recorder_name, flags)

    def _add_flagged_recoder(self, model, parameter, recorder_name, flags):
        try:
            record_ts = flags['timeseries']
        except KeyError:
            pass
        else:
            if record_ts:
                NumpyArrayParameterRecorder(model, parameter, name=recorder_name)

    def save_pywr_results(self):
        """ Save the outputs from a Pywr model run to Hydra. """
        # Ensure all the results from previous run are removed.
        self._delete_resource_scenarios()

        # Convert the scenario from JSONObject to normal dict
        # This is required to ensure that the complete nested structure (of dicts)
        # is properly converted to JSONObject's by the client.
        scenario = self._copy_scenario()

        # First add any new attributes required
        attribute_names = []

        for recorder in self._df_recorders:
            attribute_names.append(self._get_attribute_name_from_recorder(recorder, is_dataframe=True))
        for recorder in self._non_df_recorders:
            attribute_names.append(self._get_attribute_name_from_recorder(recorder))

        attribute_names = set(attribute_names)

        attributes = []
        for attribute_name in attribute_names:
            attributes.append({
                'name': attribute_name,
                'description': '',
                'dimension_id' : self.attr_dimension_map.get(attribute_name)
            })

        # The response attributes have ids now.
        response_attributes = self.client.add_attributes(attributes)
        # Update the attribute mapping
        self.attributes.update({attr.id: attr for attr in response_attributes})

        for resource_scenario in self.generate_array_recorder_resource_scenarios():
            scenario['resourcescenarios'].append(resource_scenario)

        self.client.update_scenario(scenario)

    def generate_array_recorder_resource_scenarios(self):
        """ Generate resource scenario data from NumpyArrayXXX recorders. """
        if self._df_recorders is None:
            # TODO turn this on when logging is sorted out.
            # logger.info('No array recorders defined not results saved to Hydra.')
            return

        for recorder in self._df_recorders:
            df = recorder.to_dataframe()

            columns = []
            for name in df.columns.names:
                columns.append([f'{name}: {v}' for v in df.columns.get_level_values(name)])

            df.columns = [', '.join(values) for values in zip(*columns)]

            # Resample timeseries if required
            if isinstance(df.index, pandas.DatetimeIndex) and self.output_resample_freq is not None:
                df = df.resample(self.output_resample_freq).mean()
            elif isinstance(df.index, pandas.PeriodIndex):
                df.index = df.index.to_timestamp()

            #clean up column names, as some can look like: ' : 0' so turn it into '0'
            if df.columns[0].strip().find(':') == 0:
                new_col_names = []
                for colname in df.columns:
                    new_col_names.append(colname.split(':')[1].strip())
                df.columns = new_col_names

            # Convert to JSON for saving in hydra
            value = df.to_json(date_format='iso', date_unit='s')

            #Use this later so we can create sensible labels and metadata
            #for when the data is back in hydra
            is_timeseries = False
            if isinstance(df.index, pandas.DatetimeIndex):
                is_timeseries = True

            resource_scenario = self._make_recorder_resource_scenario(recorder,
                                                                      value,
                                                                      'dataframe',
                                                                      is_timeseries=is_timeseries,
                                                                      is_dataframe=True)

            if resource_scenario is None:
                continue

            yield resource_scenario

        if self._non_df_recorders is None:
            # TODO turn this on when logging is sorted out.
            # logger.info('No array recorders defined not results saved to Hydra.')
            return

        #TODO merge this and the above as this is a duplicate
        for recorder in self._non_df_recorders:
            foundValidValue = False
            try:
                value = list(recorder.values())
                data_type = 'array'

                if len(value) == 1:
                    value = value[0]
                    data_type = 'scalar'
                foundValidValue = True
            except NotImplementedError:
                continue

            if foundValidValue is False:
                try:
                    value = recorder.value()
                    data_type = 'scalar'
                except NotImplementedError:
                    continue

            resource_scenario = self._make_recorder_resource_scenario(recorder,
                                                                      value,
                                                                      data_type,
                                                                      is_dataframe=False)

            if resource_scenario is None:
                continue

            yield resource_scenario

    def _make_recorder_resource_scenario(self, recorder, value, data_type, is_timeseries=False, is_dataframe=False):
        # Get the attribute and its ID
        attribute_name = self._get_attribute_name_from_recorder(recorder, is_dataframe=is_dataframe)

        attribute = self._get_attribute_from_name(attribute_name)

        # Now we need to ensure there is a resource attribute for all nodes and recorder attributes

        recorder_node = self._get_node_from_recorder(recorder)


        if recorder_node is None:
            # Try to get the resource attribute
            resource_attribute = self.client.add_resource_attribute(
                'NETWORK',
                self.network_id,
                attribute['id'],
                is_var='Y',
                error_on_duplicate=False)
            resource_attribute_id = resource_attribute['id']

        else:
            try:
                resource_attribute_id = self._get_resource_attribute_id(recorder_node.name,
                                                                    attribute_name)
            except ValueError:
                found_ra_id = False
                resource_type = 'NODE'
                for node in self.data['nodes']:
                    if node['name'] == recorder_node.name:
                        resource_id = node['id']
                        break
                    if recorder_node.parent is not None:
                        if node['name'] == recorder_node.parent.name:
                            resource_id = node['id']
                            break

                # Try to get the resource attribute
                resource_attribute = self.client.add_resource_attribute(
                    resource_type,
                    resource_id,
                    attribute['id'],
                    is_var='Y',
                    error_on_duplicate=False)
                resource_attribute_id = resource_attribute['id']

        unit_id = self.attr_unit_map.get(attribute.id)

        metadata = {}
        if attribute_name.find('simulated_') == 0:
            metadata['yAxisLabel'] = attribute_name.split('_')[1]
            if is_timeseries is True:
                metadata['xAxisLabel'] = 'Time'

        resource_scenario = self._make_dataset_resource_scenario(recorder.name,
                                                                 value,
                                                                 data_type,
                                                                 resource_attribute_id,
                                                                 unit_id=unit_id,
                                                                 encode_to_json=False,
                                                                 metadata=metadata)
        return resource_scenario

    def make_attr_unit_map(self):
        """
            Create a mapping between an attribute ID and its unit, as defined
            in the template
        """
        if self.template is None:
            log.info("Cannot make unit map. Template is Null")
            return

        for templatetype in self.template.templatetypes:
            for typeattr in templatetype.typeattrs:
                self.attr_unit_map[typeattr.attr_id] = typeattr.unit_id

    def make_attr_name_map(self):
        """
            Create a mapping between an attribute's name and itself, as defined
            in the template
        """
        attr_name_map = {}
        for templatetype in self.template.templatetypes:
            for typeattr in templatetype.typeattrs:
                attr = self.client.get_attribute_by_id(typeattr.attr_id)
                attr_name_map[attr.name] = attr
                #populate the dimensioin mapping
                self.attr_dimension_map[attr.name] = attr.dimension_id

        return attr_name_map
