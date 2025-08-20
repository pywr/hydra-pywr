import hashlib
import hmac
from random import randbytes
import tempfile
import pandas as pd
import logging
import json
from pywrparser.utils import parse_reference_key
import os
import re
from hydra_client.connection import RemoteJSONConnection

log = logging.getLogger(__name__)

def get_results_processor(scenario_id:int,
                          df_recorders:list,
                          non_df_recorders:list,
                          hydra_network:dict,
                          hydra_client:RemoteJSONConnection,
                          hydra_template:dict,
                          hydra_attributes:dict,
                          data_dir:str,
                          output_resample_freq=None):

    from .mongo import MongoResultsProcessor
    mongoResultsProcessor = MongoResultsProcessor(
        scenario_id=scenario_id,
        df_recorders=df_recorders,
        non_df_recorders=non_df_recorders,
        hydra_network=hydra_network,
        hydra_client=hydra_client,
        hydra_template=hydra_template,
        hydra_attributes=hydra_attributes,
        output_resample_freq=output_resample_freq,
        data_dir=data_dir
    )
    mongoResultsProcessor.connect()
    if mongoResultsProcessor.mongo_client is not None:
        return mongoResultsProcessor
    else:
        from .hydra import HydraResultsProcessor
        hydraResultsProcessor = HydraResultsProcessor(        
            scenario_id=scenario_id,
            df_recorders=df_recorders,
            non_df_recorders=non_df_recorders,
            hydra_network=hydra_network,
            hydra_client=hydra_client,
            hydra_template=hydra_template,
            hydra_attributes=hydra_attributes,
            output_resample_freq=output_resample_freq,
            data_dir=data_dir
        )
        return hydraResultsProcessor

class ResultsProcessor():

    def __init__(self, scenario_id, df_recorders, non_df_recorders, **kwargs):
        self.scenario_id = scenario_id
        self.df_recorders = df_recorders if df_recorders is not None else []
        self.non_df_recorders = non_df_recorders if non_df_recorders is not None else []
        self.resultstores = {}
        self.attr_dimension_map={}
        tmpdir = tempfile.gettempdir()
        self.results_location = os.path.join(os.getenv("PYWR_RESULTS_LOCATION", tmpdir), str(self.scenario_id))
        os.makedirs(self.results_location, exist_ok=True)
        self.bucket_name = os.getenv("PYWR_RESULTS_S3_BUCKET", 'pywr-results')
        hashkey = hashlib.sha256(randbytes(56)).hexdigest().encode('utf-8')
        self.s3_path = hmac.digest(hashkey, str(self.scenario_id).encode('utf-8'), hashlib.sha256).hex()
        self.output_resample_freq = kwargs.get('output_resample_freq')
        self.hydra_network = kwargs['hydra_network']
        self.hydra_client = kwargs['hydra_client']
        self.hydra_template = kwargs['hydra_template']
        self.data_dir = kwargs.get('data_dir', '/tmp')

        self.node_lookup = {}
        self.node_attr_lookup = {}
        for n in self.hydra_network['nodes']:
            self.node_lookup[n.name] = n
            self.node_attr_lookup[n.name] = {}
            for a in n['attributes']:
                self.node_attr_lookup[n.name][a.attr_id] = a
    def save(self):
        """
            Function to be overwritten by the subclass
        """
        pass

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

    def flush(self):
        #flush the results to the h5 file
        for resultstore in self.resultstores.values():
            resultstore.close()

    def zip_results(self):
        import zipfile
        zip_path = os.path.join(self.data_dir, f"results_{self.scenario_id}.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(self.results_location):
                if file.endswith('.h5'):
                    abs_path = os.path.join(self.results_location, file)
                    zipf.write(abs_path, arcname=file)
    
    def process_df_recorder(self, recorder):

        try:
            df = recorder.to_dataframe()
        except NotImplementedError:
            return None

        # If the recorder has no data, skip it
        if df.empty:
            log.warning(f"Recorder {recorder.name} has no data, skipping.")
            return None

        columns = []
        for name in df.columns.names:
            columns.append([f'{name}: {v}' for v in df.columns.get_level_values(name)])

        df.columns = [', '.join(values) for values in zip(*columns)]
        # Resample timeseries if required
        if isinstance(df.index, pd.DatetimeIndex) and self.output_resample_freq is not None:
            df = df.resample(self.output_resample_freq).mean()
        elif isinstance(df.index, pd.PeriodIndex):
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
                return None
        else:
            nodename = "network"
            attrname = recorder.name

        filename = f'{attrname}.h5'
        resultstore = self.resultstores.get(filename)
        if resultstore is None:
            resultstore = pd.HDFStore(os.path.join(self.results_location, filename), mode='w')
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
        if isinstance(df.index, pd.DatetimeIndex):
            is_timeseries=True

        return {
            "data": value,
            "data_type": 'dataframe',
            "is_timeseries": is_timeseries
        }
    def process_df_recorders(self):
        for recorder in self.df_recorders:
            yield self.process_df_recorder(recorder)

    def process_non_df_recorders(self, non_df_recorders):
        if non_df_recorders is None:
            log.warning('No array recorders defined, results not saved to Hydra.')
            return
        for recorder in non_df_recorders:
            yield self.process_non_df_recorder(recorder)

    def process_non_df_recorder(self, recorder):
        try:
            recorder.values()
        except NotImplementedError:
            return None

        try:
            data_type = "array"
            value = list(recorder.values())
            if len(value) == 1:
                value = value[0]
                data_type = "scalar"
            value = json.dumps(value)
        except (NotImplementedError,  TypeError):
            return None
        else:
            try:
                if hasattr(recorder, 'value'):
                    value = recorder.value()
                    data_type = "scalar"
            except NotImplementedError:
                return None

        return {
            "data": value,
            "data_type": data_type
        }

    def _get_pywr_node_from_recorder(self, recorder):

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
        if node is None:
            try:
                nodes = recorder.nodes
                if len(nodes) == 1:
                    return nodes[0]
            except AttributeError:
                try:
                    nodes = recorder.parameter.nodes
                    if len(nodes) == 1:
                        return nodes[0]
                except AttributeError:
                    return None
        return node

    def _get_hydra_node_from_recorder(self, recorder, pywr_node=None):
        
        if pywr_node is None:
            pywr_node = self._get_pywr_node_from_recorder(recorder) 

        if pywr_node is None:
            return None
        
        if pywr_node.name in self.node_lookup:
            return self.node_lookup[pywr_node.name]
        if pywr_node.parent is not None and pywr_node.parent.name in self.node_lookup:
            return self.node_lookup[pywr_node.parent.name]
        
        return None