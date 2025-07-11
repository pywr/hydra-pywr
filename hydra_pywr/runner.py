import hmac
import pandas
import yaml
import tempfile
import os
import logging
import json
import hashlib
import re
from urllib.parse import urlparse

from pywr.model import Model
from pywr.nodes import Node, Storage
from pywr.parameters import Parameter, DeficitParameter
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayStorageRecorder, NumpyArrayParameterRecorder
from pywr.recorders.progress import ProgressRecorder

from pywrparser.lib import PywrTypeJSONEncoder
from pywrparser.utils import parse_reference_key
from random import randbytes

from .exporter import HydraToPywrNetwork, find_missing_parameters, rewrite_ref_parameters

from pywrparser.types.network import PywrNetwork
from hydra_pywr.nodes import *
from . import utils

log = logging.getLogger(__name__)

domain_solvers = {
    "water": "glpk-edge",
    "energy": "glpk-dcopf"
}

def run_file(filename, domain, output_file):
    pfr = PywrFileRunner(domain)
    pfr.load_pywr_model_from_file(filename)
    pfr.run_pywr_model(output_file)


def run_network_scenario(client, scenario_id, template_id, domain,
                         solver=None, data_dir='/tmp'):

    runner = PywrHydraRunner.from_scenario_id(client, scenario_id,
                                             template_id=template_id,
                                             data_dir=data_dir)
    runner.setup(solver=solver)
    runner.run_pywr_model()
    runner.save_pywr_results()
    log.info(f'Pywr model run success. Network ID: {runner.data.id}, Scenario ID: {scenario_id}')


def save_pywr_file(data, data_dir, network_id=None, scenario_id=None):
    """
    Save pywr json data to the specified directory
    """
    if data_dir is None:
        log.info("No data dir specified. Returning.")
        exit(1)

    title = data['metadata']['title']

    #check if the output folder exists and create it if not
    if not os.path.isdir(data_dir):
        #exist_ok sets unix the '-p' functionality to create the whole path
        os.makedirs(data_dir, exist_ok=True)

    filename = os.path.join(data_dir, f'{title}.json')
    with open(filename, mode='w') as fh:
        json.dump(data, fh, sort_keys=True, indent=2)

    log.info(f'Successfully exported "{filename}". Network ID: {network_id}, Scenario ID: {scenario_id}')

    return filename

class PywrFileRunner():
    def __init__(self, domain="water"):
        self.model = None
        self.domain = domain


    def load_pywr_model_from_file(self, filename, solver=None):
        if self.domain == "energy":
            from pywr_dcopf import core
        try:
            from . import hydra_pywr_custom_module
        except (ModuleNotFoundError, ImportError) as e:
            pass

        pnet, errors, warnings = PywrNetwork.from_file(filename)
        if warnings:
            for component, warns in warnings.items():
                for warn in warns:
                    log.info(warn)

        if errors:
            for component, errs in errors.items():
                for err in errs:
                    log.error(err)
            exit(1)

        pywr_data = pnet.as_dict()

        model = Model.load(pywr_data, solver=domain_solvers[self.domain])
        self.model = model
        return pywr_data


    def run_pywr_model(self, outfile="output.csv"):
        if self.domain == "energy":
            from pywr_dcopf import core

        model = self.model

        # Add a progress recorder to monitor the run.
        ProgressRecorder(model)

        # Force a setup regardless of whether the model has been run or setup before
        model.setup()

        run_stats = model.run()
        log.info(run_stats)

        try:
            df = model.to_dataframe()
            df.to_csv(outfile)
        except ValueError:
            print("Unable to output the model dataframe results.")

class PywrHydraRunner(HydraToPywrNetwork):
    """ An extension of `HydraToPywrNetwork` that adds methods for running a Pywr model. """

    def __init__(self, *args, domain="water", **kwargs):
        self.output_resample_freq = kwargs.pop('output_resample_freq', None)
        super(PywrHydraRunner, self).__init__(*args, **kwargs)
        self.domain = domain
        self.model = None
        self._df_recorders = None
        self._non_df_recorders = None

        self.attr_dimension_map = {}

        self.node_lookup = {}
        self.node_attr_lookup = {}
        for n in self.data['nodes']:
            self.node_lookup[n.name] = n
            self.node_attr_lookup[n.name] = {}
            for a in n['attributes']:
                self.node_attr_lookup[n.name][a.attr_id] = a

        self.attr_name_map = self.make_attr_name_map()

        self.solver = kwargs.get('solver')

        self.make_attr_unit_map()

        self.limit_nodes_recording = False

        tmpdir = tempfile.gettempdir()
        self.results_location = os.path.join(os.getenv("PYWR_RESULTS_LOCATION", tmpdir), str(self.scenario_id))
        os.makedirs(self.results_location, exist_ok=True)
        self.bucket_name = os.getenv("PYWR_RESULTS_S3_BUCKET", 'pywr-results')
        hashkey = hashlib.sha256(randbytes(56)).hexdigest().encode('utf-8')
        self.s3_path = hmac.digest(hashkey, str(self.scenario_id).encode('utf-8'), hashlib.sha256).hex()

        self.resultstores = {}

    def setup(self, solver=None):
        """
            Having exported the model, now update the model by doing such things as
            retrieving any external files referenced in the model
        """
        network_data = self.build_pywr_network()
        pywr_network = PywrNetwork(network_data)
        pywr_network.promote_inline_parameters()
        pywr_network.detach_parameters()

        missing_params = find_missing_parameters(pywr_network)
        if len(missing_params) > 0:
            rewrite_ref_parameters(pywr_network, missing_params)

        url_refs = pywr_network.url_references()
        for url, refs in url_refs.items():
            u = urlparse(url)
            filedest = None
            if u.scheme == "s3":
                filedest = utils.retrieve_s3(url, self.data_dir)
            elif u.scheme.startswith("http"):
                filedest = utils.retrieve_url(url, self.data_dir)
            else:
                #'/file.csv' -> ('', file.csv)
                spliturl = url.strip(os.sep).split(os.sep)
                #If the url is 'file.csv'
                if len(spliturl) == 1:

                    full_path = self.filedict.get(spliturl[0])
                    if full_path is not None:
                        filedest = utils.retrieve_s3(full_path, self.data_dir)
            if filedest is not None:
                for ref in refs:
                    ref.data["url"] = filedest

        if self.data_dir is not None:
            self.modelfile = save_pywr_file(pywr_network.as_dict(), self.data_dir, self.data['id'], self.scenario_id)

        self.load_pywr_model(pywr_network, solver=solver)

    def save_to_s3(self):
        """
            Upload the pywr model to S3
        """
        if self.modelfile is None:
            log.info("No model file to upload to S3")
            return

        import boto3
        s3 = boto3.client('s3')
        s3.upload_file(self.modelfile, Bucket=self.bucket_name, Key=f"{self.s3_path}/{os.path.basename(self.modelfile)}")
        log.info("Model file saved to s3 bucket %s", self.bucket_name)
        self.using_s3 = True

    def zip_results(self):
        import zipfile
        zip_path = os.path.join(self.data_dir, f"results_{self.scenario_id}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(self.results_location):
                if file.endswith('.h5'):
                    abs_path = os.path.join(self.results_location, file)
                    zipf.write(abs_path, arcname=file)

    def get_file(self):
        if self.using_s3:
            #get the file from s3
            import boto3
            s3 = boto3.client('s3')
            #check if the file exists
            try:
                s3.head_object(Bucket=self.bucket_name, Key=f"{self.s3_path}/{os.path.basename(self.modelfile)}")
            except Exception as e:
                log.info("Model file not found in s3 bucket %s", self.bucket_name)
                return None
            #sync to the bucket location
            s3.download_file(self.bucket_name, f"{self.s3_path}/{os.path.basename(self.modelfile)}", self.modelfile)
            log.info("Model file downloaded from s3 bucket %s", self.bucket_name)

        return self.modelfile

    def _copy_scenario(self):
        # Now construct a scenario object
        scenario = self.data.scenarios[0]
        new_scenario = {k: v for k, v in scenario.items() if k != 'resourcescenarios'}

        new_scenario['resourcescenarios'] = []
        return new_scenario

    def _delete_resource_scenarios(self):
        scenario = self.data.scenarios[0]

        ra_is_var_map = {ra['id']: ra['attr_is_var'] for ra in self.hydra.get_resource_attributes(ref_key="network", ref_id=self.network_id)}
        ra_to_delete = []

        # Compile a list of resource attributes to delete
        for resource_scenario in scenario['resourcescenarios']:
            ra_id = resource_scenario['resource_attr_id']
            ra_is_var = ra_is_var_map.get(ra_id, 'N')
            if ra_is_var == 'Y':
                ra_to_delete.append(ra_id)

        # Now delete them all
        self.hydra.delete_resource_scenarios(scenario_id=scenario['id'], resource_attr_ids=ra_to_delete, quiet=True)


    def load_pywr_model(self, pywr_network, solver=None):
        """ Create a Pywr model from the exported data. """

        if self.domain == "energy":
            from pywr_dcopf import core

        try:
            from . import hydra_pywr_custom_module
        except (ModuleNotFoundError, ImportError) as e:
            pass

        if solver is None:
            solver = domain_solvers[self.domain]

        #data = self.build_pywr_network()
        #pnet = PywrNetwork(data)
        pywr_data = pywr_network.as_json()

        model = Model.loads(pywr_data, solver=solver)

        tmp = tempfile.gettempdir()

        file_location = os.path.join(tmp, f"pywrmodel_n_{self.data['id']}_s_{self.data['scenarios'][0]['id']}.json")

        with open(file_location, 'w') as f:
            json.dump(pywr_network.as_dict(), f, sort_keys=True, indent=4)
            log.info("File written to %s", file_location)

        self.model = model

        return pywr_data

    def get_nodes_to_record(self):
        """
        Get the nodes to record in the network.
        """

        record_nodes_attr = list(filter(lambda x:x['name'] == 'record_nodes', self.attributes.values()))
        if len(record_nodes_attr) > 0:
            node_recorder_attribute = record_nodes_attr[0]
        else:
            return []

        for network_ra in self.data['attributes']:
            if network_ra['attr_id'] == node_recorder_attribute['id']:
                rs = list(filter(lambda x:x.resource_attr_id==network_ra['id'],
                                 self.data.scenarios[0].resourcescenarios))
                value = []
                if len(rs) > 0:
                    try:
                        value = json.loads(rs[0]['dataset']['value'])
                        self.limit_nodes_recording = True
                        #in case, the original value was, for example, "'[]'"
                        if isinstance(value, str):
                            value = json.loads(value)
                    except:
                        log.critical(f"Unable to read which nodes to record. Value should be an array of node names or IDS. The value is: {rs[0]['dataset']['value']}")
                        return []
                return value


    def run_pywr_model(self, domain="water"):
        """
            Run a Pywr model from the exported data.
            If no model has been loaded (see `load_pywr_model`) then a load is attempted.
        """

        if domain == "energy":
            from pywr_dcopf import core

        if self.model is None:
            self.load_pywr_model(solver=domain_solvers[domain])

        model = self.model

        # Add a progress recorder to monitor the run.
        ProgressRecorder(model)

        # Add recorders for monitoring the simulated timeseries of nodes
        self._add_node_flagged_recorders(model)
        # Add recorders for parameters that are flagged
        self._add_parameter_flagged_recorders(model)

        df_recorders = []
        non_df_recorders = []
        for recorder in model.recorders:
            if hasattr(recorder, 'to_dataframe'):
                df_recorders.append(recorder)

            if hasattr(recorder, "value") or hasattr(recorder, "values"):
                non_df_recorders.append(recorder)




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

        log.info(run_stats)

    def save_results_to_s3(self):
        """
            upload the h5 file results to s3
        """
        resultfiles = list(self.resultstores.keys())
        for f in os.listdir(self.results_location):
            if f not in resultfiles:
                continue
            log.info("Saving %s to bucket %s s3", f, self.bucket_name)
            import boto3
            s3 = boto3.client('s3')
            s3.upload_file(os.path.join(self.results_location, f), Bucket=self.bucket_name, Key=f"{self.s3_path}/{f}")
            log.info("%s saved to s3 bucket %s", f, self.bucket_name)

    def get_do_config(self):
        do_config_prefix = "do_"
        no_config_errtxt = f"No DO settings found on network {self.data['name']}"

        for attr in self.data["attributes"]:
            if attr["name"].lower().startswith(do_config_prefix):
                do_config_attr = attr
                break
        else:
            raise RuntimeError(no_config_errtxt)

        rs = [rs for rs in self.data['scenarios'][0]['resourcescenarios'] if rs.resource_attr_id == do_config_attr.id]
        if len(rs) == 0:
            raise RuntimeError(no_config_errtxt)
        else:
            return yaml.safe_load(rs[0].dataset.value)

    def get_moea_config(self):
        config_prefix = "moea"
        no_config_errtxt = f"No MOEA settings found on network {self.data['name']}"

        for attr in self.data["attributes"]:
            if attr["name"].lower().startswith(config_prefix):
                config_prefix = attr
                break
        else:
            raise RuntimeError(no_config_errtxt)

        rs = [rs for rs in self.data['scenarios'][0]['resourcescenarios'] if rs.resource_attr_id == config_prefix.id]
        if len(rs) == 0:
            raise RuntimeError(no_config_errtxt)
        else:
            return yaml.safe_load(rs[0].dataset.value)

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

        for attribute_id, attribute in self.attributes.items():
            if attribute['name'].lower() == name.lower() and attribute.get('dimension_id') == dimension_id:
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

    def _add_node_flagged_recorders(self, model):

        nodes_to_record = self.get_nodes_to_record()

        if self.domain == "energy":
            from pywr_dcopf.core import Generator, Load, Line, Battery
            node_classes = (Node, Generator, Load, Line, Battery)
        else:
            node_classes = (Node,)

        for node in model.nodes:
            if self.limit_nodes_recording is True and node.name not in nodes_to_record:
                continue

            try:
                flags = self._node_recorder_flags[node.name]
            except KeyError:
                flags = {'timeseries': True}  # Default to recording timeseries if not defined.

            for flag, to_record in flags.items():
                if not to_record:
                    continue

                if flag == 'timeseries':
                    #if isinstance(node, (Node, Generator, Load, Line)):
                    if isinstance(node, node_classes):
                        name = '__{}__:{}'.format(node.name, 'simulated_flow')
                        NumpyArrayNodeRecorder(model, node, name=name)
                    elif isinstance(node, (Storage)):
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
            p = model.parameters[parameter_name]
            if ':' in p.name:
                recorder_name = p.name.rsplit(':', 1)
                recorder_name[1] = 'simulated_' + recorder_name[1]
                recorder_name = ':'.join(recorder_name)
            else:
                recorder_name = 'simulated_' + p.name

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
        # This is required to ensure that the complete nested structure of dicts
        # is properly converted to JSONObjects by the client.
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
                'project_id': self.data.project_id,
                'dimension_id' : self.attr_dimension_map.get(attribute_name)
            })

        # The response attributes have ids now.
        try:
            response_attributes = self.hydra.add_attributes(attrs=attributes)
        except Exception as e:
            if hasattr(e, 'message') and 'permission denied' in e.message.lower():
                for a in attributes:
                    a['project_id'] = None
                    a['network_id'] = self.data['id']
                response_attributes = self.hydra.add_attributes(attrs=attributes)


        # Update the attribute mapping
        self.attributes.update({attr.id: attr for attr in response_attributes})

        for resource_scenario in self.generate_array_recorder_resource_scenarios():
            scenario['resourcescenarios'].append(resource_scenario)

        for i in range(0, len(scenario['resourcescenarios']), 100):
                chunk = scenario['resourcescenarios'][i:i+100]
                log.info('Saving %s datasets', len(chunk))
                self.hydra.bulk_update_resourcedata(
                    scenario_ids=[scenario['id']],
                    resource_scenarios=chunk
                )

        #flush the results to the h5 file
        for resultstore in self.resultstores.values():
            resultstore.close()

        log.info("Results stored to: %s", self.results_location)

        self.save_results_to_s3()

        self.zip_results()

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
            resource_id = self.data['id']
            attribute_name = self._get_attribute_name_from_recorder(
                recorder,
                is_dataframe=is_dataframe
            )
            recorder_name = recorder.name
            if attribute_name.endswith('value'):
                recorder_name = recorder.name + '_value'

            attribute = self._get_attribute_from_name(attribute_name)

            try:
                recorder_node = self._get_node_from_recorder(recorder)
            except AttributeError:
                recorder_node=None

            if recorder_node is None:
                for network_ra in self.data['attributes']:
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
                    if recorder_node is None:
                        recorder_node_name = recorder.name.split('__:')[0].replace('__', '')
                    else:
                        recorder_node_name = recorder_node.name

                    for node in self.data['nodes']:
                        if node['name'] == recorder_node_name:
                            resource_id = node['id']
                            resource_type = 'NODE'

                            break
                        if recorder_node.parent is not None:
                            if node['name'] == recorder_node.parent.name:
                                resource_id = node['id']
                                resource_type = 'NODE'
                                break

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
            returned_new_ids = self.hydra.add_resource_attributes(resource_attributes=chunk)

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
    def generate_array_recorder_resource_scenarios(self):
        """ Generate resource scenario data from NumpyArrayXXX recorders. """
        if self._df_recorders is None:
            log.warning('No array recorders defined, results not saved to Hydra.')
            return

        #get a mapping from recorder names to resource attribute IDs
        df_recorder_ra_id_map = self.add_resource_attributes(self._df_recorders, is_dataframe=True)
        non_df_recorder_ra_id_map = self.add_resource_attributes(self._non_df_recorders, is_dataframe=False)

        for recorder in self._df_recorders:
            try:
                df = recorder.to_dataframe()
            except NotImplementedError:
                continue

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

            if "__:" in recorder.name:
                try:
                    nodename, attrname = parse_reference_key(recorder.name)
                except Exception as e:
                    log.critical(f"Unable to process result for recorder recorder {recorder.name}: Unable to parse the name.")
                    continue
            else:
                nodename = "network"
                attrname = recorder.name

            filename = f'{attrname}.h5'
            resultstore = self.resultstores.get(filename)
            if resultstore is None:
                resultstore = pandas.HDFStore(os.path.join(self.results_location, filename), mode='w')
                self.resultstores[filename] = resultstore

            noderef = re.sub(r'^[^a-zA-Z_]+|[^a-zA-Z0-9_]', '', nodename)
            resultstore.put(f"{noderef}", df)
            resultstore[f"{noderef}"].attrs['pandas_type'] = 'frame'

            # Convert to JSON for saving in hydra
            value = json.dumps({
                "data":
                {
                    "url": f"s3://{self.bucket_name}/{self.s3_path}/{attrname}.h5",
                    "group": f"{noderef}"
                }
            })

            #Use this later so we can create sensible labels and metadata
            #for when the data is back in hydra
            is_timeseries=False
            if isinstance(df.index, pandas.DatetimeIndex):
                is_timeseries=True

            resource_scenario = self._make_recorder_resource_scenario(recorder,
                                                                      value,
                                                                      df_recorder_ra_id_map[recorder.name],
                                                                      'dataframe',
                                                                      is_timeseries=is_timeseries,
                                                                      is_dataframe=True)

            if resource_scenario is None:
                continue

            yield resource_scenario

        if self._non_df_recorders is None:
            log.warning('No array recorders defined, results not saved to Hydra.')
            return

        for recorder in self._non_df_recorders:
            try:
                recorder.values()
            except NotImplementedError:
                continue

            try:
                data_type = "array"
                value = list(recorder.values())
                if len(value) == 1:
                    value = value[0]
                    data_type = "scalar"
                value = json.dumps(value)
            except (NotImplementedError,  TypeError):
                continue
            else:
                try:
                    if hasattr(recorder, 'value'):
                        value = recorder.value()
                        data_type = "scalar"
                except NotImplementedError:
                    continue

            if non_df_recorder_ra_id_map.get(recorder.name+'_value') is None:
                continue

            resource_scenario = self._make_recorder_resource_scenario(recorder,
                                                                      value,
                                                                      non_df_recorder_ra_id_map[recorder.name+'_value'],
                                                                      data_type,
                                                                      is_dataframe=False)

            if resource_scenario is None:
                continue

            yield resource_scenario

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

        resource_scenario = self._make_dataset_resource_scenario(attribute_name,
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
                attr = self.attributes[typeattr.attr_id]
                attr_name_map[attr.name] = attr
                #populate the dimension mapping
                self.attr_dimension_map[attr.name] = attr.dimension_id

        return attr_name_map

    def _make_dataset_resource_scenario(self, name, value, data_type, resource_attribute_id,
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




