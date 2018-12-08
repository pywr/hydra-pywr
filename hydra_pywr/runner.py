from .exporter import PywrHydraExporter
import copy
from pywr.model import Model
from pywr.nodes import Node, Storage
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayStorageRecorder, NumpyArrayLevelRecorder, \
    NumpyArrayParameterRecorder
from pywr.recorders.progress import ProgressRecorder
from .template import PYWR_ARRAY_RECORDER_ATTRIBUTES


def add_node_array_recorders(model):
    """ Helper function to add NumpyArrayXXX recorders to a Pywr model. """

    # Add node recorders
    for node in model.nodes:
        if isinstance(node, Node):
            name = '__{}__:{}'.format(node.name, 'simulated_flow')
            NumpyArrayNodeRecorder(model, node, name=name)
        elif isinstance(node, Storage):
            name = '__{}__:{}'.format(node.name, 'simulated_volume')
            NumpyArrayStorageRecorder(model, node, name=name)
        else:
            import warnings
            warnings.warn('Unrecognised node subclass "{}" with name "{}". Skipping '
                          'recording this node.'.format(node.__class__.__name__, node.name),
                          RuntimeWarning)


class PywrHydraRunner(PywrHydraExporter):
    """ An extension to `PywrHydraExporter` that adds methods for running a Pywr model. """
    def __init__(self, *args, **kwargs):
        super(PywrHydraRunner, self).__init__(*args, **kwargs)
        self.model = None
        self._array_recorders = None

    def _copy_scenario(self):
        # Now construct a scenario object
        scenario = self.data.scenarios[0]
        scenario = copy.deepcopy(scenario)
        scenario.resourcescenarios = []
        return scenario

    def load_pywr_model(self):
        """ Create a Pywr model from the exported data. """
        pywr_data, modules = self.get_pywr_data()

        for module_name, module_data in modules.items():
            import pdb; pdb.set_trace()
            # Simply execute each module in this global namespace
            module_contents = module_data['contents']
            exec(module_contents, globals())

        model = Model.load(pywr_data)
        self.model = model

    def run_pywr_model(self):
        """ Run a Pywr model from the exported data.

        If no model has been loaded (see `load_pywr_model`) then a load is attempted.
        """
        if self.model is None:
            self.load_pywr_model()

        model = self.model

        # Add a progress recorder to monitor the run.
        ProgressRecorder(model)

        # Add recorders for monitoring the simulated timeseries of nodes
        add_node_array_recorders(model)

        array_recorders = []
        for recorder in model.recorders:
            if isinstance(recorder, (
                    NumpyArrayNodeRecorder,
                    NumpyArrayStorageRecorder,
                    NumpyArrayLevelRecorder,
                    NumpyArrayParameterRecorder,
            )):
                array_recorders.append(recorder)

        # Check the model
        model.check()

        # Force a setup regardless of whether the model has been run or setup before
        model.setup()

        # Now run the model.
        run_stats = model.run()

        # Save these for later
        self._array_recorders = array_recorders

    def _get_resource_attribute_id(self, node_name, attribute_name):

        attribute = self._get_attribute_from_name(attribute_name)
        attribute_id = attribute['id']

        for node in self.data['nodes']:

            if node['name'] == node_name:
                print(node_name, node['id'])
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

        for attribute_id, attribute in self.attributes.items():
            if attribute['name'] == name:
                return attribute
        raise ValueError('No attribute with name "{}" found.'.format(name))

    def _get_node_from_recorder(self, recorder):
        try:
            node = recorder.node
        except AttributeError:
            node = recorder.parameter.node
        return node

    def _get_attribute_name_from_recorder(self, recorder):
        if recorder.name is None:
            attribute_name = recorder.__class__
        else:
            node = self._get_node_from_recorder(recorder)
            prefix = '__{}__:'.format(node.name)
            suffix = '.{}'.format(node.name)
            if recorder.name.startswith(prefix):
                attribute_name = recorder.name.replace(prefix, '')
            elif recorder.name.endswith(suffix):
                attribute_name = recorder.name.replace(suffix, '')
            else:
                attribute_name = recorder.name
        return attribute_name

    def save_pywr_results(self, client):
        """ Save the outputs from a Pywr model run to Hydra. """
        scenario = self._copy_scenario()

        # First add any new attributes required
        attributes = []
        for recorder in self._array_recorders:
            attributes.append({
                'name': self._get_attribute_name_from_recorder(recorder),
                'description': ''
            })

        # The response attributes have ids now.
        response_attributes = client.add_attributes(attributes)
        # Update the attribute mapping
        self.attributes.update({attr.id: attr for attr in response_attributes})

        for resource_scenario in self.generate_array_recorder_resource_scenarios(client):
            scenario['resourcescenarios'].append(resource_scenario)

        client.update_scenario(scenario)

    def generate_array_recorder_resource_scenarios(self, client):
        """ Generate resource scenario data from NumpyArrayXXX recorders. """
        if self._array_recorders is None:
            # TODO turn this on when logging is sorted out.
            # logger.info('No array recorders defined not results saved to Hydra.')
            return

        for recorder in self._array_recorders:
            df = recorder.to_dataframe()

            # Convert to JSON for saving in hydra
            value = df.to_json(date_format='iso', date_unit='s')

            # Get the attribute and its ID
            attribute_name = self._get_attribute_name_from_recorder(recorder)

            # Now we need to ensure there is a resource attribute for all nodes and recorder attributes

            try:
                recorder_node = self._get_node_from_recorder(recorder)
            except AttributeError:
                continue

            try:
                resource_attribute_id = self._get_resource_attribute_id(recorder_node.name, attribute_name)
            except ValueError:
                for node in self.data['nodes']:
                    if node['name'] == recorder_node.name:
                        node_id = node['id']
                        break
                    if recorder_node.parent is not None:
                        if node['name'] == recorder_node.parent.name:
                            node_id = node['id']
                            break
                else:
                    continue
                attribute = self._get_attribute_from_name(attribute_name)

                # Try to get the resource attribute
                resource_attribute = client.add_resource_attribute('NODE', node_id, attribute['id'], is_var='Y',
                                                                   error_on_duplicate=False)
                resource_attribute_id = resource_attribute['id']

            resource_scenario = self._make_dataset_resource_scenario(recorder.name, value, 'dataframe', resource_attribute_id,
                                                                     encode_to_json=False)

            yield resource_scenario


