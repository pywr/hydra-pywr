import hmac
import pandas
import yaml
import tempfile
import os
import logging
import json
import hashlib
import re
import datetime
from urllib.parse import urlparse
from . import resultsprocessor

from pywr.model import Model
from pywr.nodes import Node, Storage
from pywr.parameters import Parameter, DeficitParameter
from pywr.recorders import NumpyArrayNodeRecorder, NumpyArrayStorageRecorder, NumpyArrayParameterRecorder
from pywr.recorders.progress import ProgressRecorder

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
    runner.save_results()
    log.info(f'Pywr model run success. Network ID: {runner.data.id}, Scenario ID: {scenario_id}')
    return runner

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

        model = Model.load(
            pywr_data, 
            solver=domain_solvers[self.domain] if solver is None else solver)
        
        self.model = model
        return pywr_data


    def run_pywr_model(self, outfile="output.csv"):
        if self.domain == "energy":
            from pywr_dcopf import core
        
        if self.model is None:
            raise RuntimeError("Unable to run pywr model. No Pywr model loaded. Please load a model before running it.")

        # Add a progress recorder to monitor the run.
        ProgressRecorder(self.model)

        # Force a setup regardless of whether the model has been run or setup before
        self.model.setup()

        run_stats = self.model.run()
        log.info(run_stats)

        try:
            df = self.model.to_dataframe()
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
        self._df_recorders = []
        self._non_df_recorders = []
        self.solver = kwargs.get('solver')

        self.limit_nodes_recording = False

        tmpdir = tempfile.gettempdir()
        self.results_location = os.path.join(os.getenv("PYWR_RESULTS_LOCATION", tmpdir), str(self.scenario_id))
        os.makedirs(self.results_location, exist_ok=True)
        self.bucket_name = os.getenv("PYWR_RESULTS_S3_BUCKET", 'pywr-results')
        hashkey = hashlib.sha256(randbytes(56)).hexdigest().encode('utf-8')
        self.s3_path = hmac.digest(hashkey, str(self.scenario_id).encode('utf-8'), hashlib.sha256).hex()

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

    def save_model_to_s3(self):
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
            raise RuntimeError("Unable to run pywr model. No Pywr model loaded. Please load a model before running it.")

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
            max_scenarios = int(max_scenarios)
            if nscenarios > max_scenarios:
                raise RuntimeError(f'Number of scenarios ({nscenarios}) exceeds the maximum limit of {max_scenarios}.')

        # Now run the model.
        run_stats = model.run()

        # Save these for later
        self._df_recorders = df_recorders
        self._non_df_recorders = non_df_recorders

        log.info(run_stats)


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

    def save_results(self):
        """ Save the outputs from a Pywr model run to Hydra. """
        resultsProcessor = resultsprocessor.get_results_processor(
            self.scenario_id,
            df_recorders=self._df_recorders,
            non_df_recorders=self._non_df_recorders,
            hydra_network=self.data,
            hydra_client=self.hydra,
            hydra_template=self.template,
            hydra_attributes=self.attributes,
            data_dir=self.data_dir,
            output_resample_freq=self.output_resample_freq
        )
        resultsProcessor.save()






