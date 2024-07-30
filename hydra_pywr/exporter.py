import json
import sys
import os
import re
from collections import defaultdict
from datetime import datetime
from pywrparser.types.network import PywrNetwork
import subprocess

from urllib.parse import urlparse

from pywrparser.types import (
    PywrParameter,
    PywrRecorder,
    PywrTimestepper,
    PywrMetadata,
    PywrTable,
    PywrScenario,
    PywrScenarioCombination,
    PywrNode,
    PywrEdge
)

from pywrparser.types.exceptions import PywrParserException

from .config import CACHE_DIR
from .rules import exec_rules
from .template import PYWR_SPLIT_LINK_TYPES
from . import utils

from hydra_base.lib.objects import JSONObject
from hydra_base.exceptions import ResourceNotFoundError

from pywrparser.lib import PywrTypeJSONEncoder


import logging
log = logging.getLogger(__name__)

PARAMETER_TYPES = (
    "PYWR_PARAMETER",
    "PYWR_DATAFRAME"
)

RECORDER_TYPES = (
    "PYWR_RECORDER",
)


def export_json(client, data_dir, scenario_id, use_cache, json_sort_keys, json_indent):
    """
        A utility function to uxport a Pywr JSON from Hydra.
    """

    exporter = HydraToPywrNetwork.from_scenario_id(client, scenario_id, use_cache=use_cache, data_dir=data_dir)
    network_data = exporter.build_pywr_network()
    network_id = exporter.data.id
    pywr_network = PywrNetwork(network_data)

    pywr_network.promote_inline_parameters()
    pywr_network.detach_parameters()

    url_refs = pywr_network.url_references()

    for url, refs in url_refs.items():
        u = urlparse(url)
        filedest = url
        if u.scheme == "s3":
            filedest = utils.retrieve_s3(url, data_dir)
        elif u.scheme.startswith("http"):
            filedest = utils.retrieve_url(url, data_dir)
        else:
            #'/file.csv' -> ('', file.csv)
            spliturl = url.strip(os.sep).split(os.sep)
            #If the url is 'file.csv'
            if len(spliturl) == 1:

                full_path = exporter.filedict.get(spliturl[0])
                if full_path is not None:
                    filedest = utils.retrieve_s3(full_path, data_dir)
            else:
                log.debug("Not processing file %s", url)

        for ref in refs:
            ref.data["url"] = filedest

    pnet_title = pywr_network.metadata.data["title"]
    outfile = os.path.join(data_dir, f"{pnet_title.replace(' ', '_')}.json")
    with open(outfile, mode='w') as fp:
        json.dump(pywr_network.as_dict(), fp, sort_keys=json_sort_keys, indent=2, cls=PywrTypeJSONEncoder)

    log.info(f"Network: {network_id}, Scenario: {scenario_id} exported to `{outfile}`")

    return outfile

"""
    Hydra => PywrNetwork
"""
class HydraToPywrNetwork():

    exclude_hydra_attrs = (
        "id", "status", "cr_date",
        "network_id", "x", "y",
        "types", "attributes", "layout",
        "network", "description"
    )

    scenario_combinations_attr_name = "scenario_combinations"

    def __init__(self, client, network, network_id, scenario_id, attributes, template, data_dir='.', **kwargs):
        self.hydra = client
        self.data = network
        self.network_id = network_id
        self.scenario_id = scenario_id
        self.attributes = attributes
        self.template = template

        self.data_dir = data_dir

        self.type_id_map = {}
        for tt in self.template.templatetypes:
            self.type_id_map[tt.id] = tt

        self.attr_unit_map = {}
        self.hydra_node_by_id = {}

        self._parameter_recorder_flags = {}
        self._inline_parameter_recorder_flags = defaultdict(dict)
        self._node_recorder_flags = {}

        self.nodes = {}
        self.edges = []
        self.parameters = {}
        self.recorders = {}
        self.tables = {}
        self.scenarios = []
        self.scenario_combinations = None


    @classmethod
    def from_scenario_id(cls, client, scenario_id, template_id=None, index=0, **kwargs):
        if kwargs.get("use_cache") is True:
            scen_cache_file = f"scenario_{scenario_id}.json"
            if not os.path.exists(CACHE_DIR):
                try:
                    os.mkdir(CACHE_DIR)
                except OSError:
                    log.error(f"Unable to create scenario cache at {CACHE_DIR}: defaulting to '/tmp'")
                    cache_dir = "/tmp"
            scen_cache_path = os.path.join(CACHE_DIR, scen_cache_file)
            if os.path.exists(scen_cache_path):
                mod_ts = os.path.getmtime(scen_cache_path)
                mod_dt = datetime.fromtimestamp(int(mod_ts))
                log.info(f"Using cached scenario updated at {mod_dt}")
                with open(scen_cache_path, 'r') as fp:
                    scenario = JSONObject(json.load(fp))
            else:
                    scenario = client.get_scenario(scenario_id=scenario_id,
                                                   include_data=True,
                                                   include_results=False,
                                                   include_metadata=True,
                                                   include_attr=False)

                    with open(scen_cache_path, 'w') as fp:
                        json.dump(scenario, fp)

                    log.info(f"Cached scenario written to '{scen_cache_path}'")

            network_id = scenario.network_id
            net_cache_file = f"network_{scenario.network_id}.json"
            net_cache_path = os.path.join(CACHE_DIR, net_cache_file)
            if os.path.exists(net_cache_path):
                mod_ts = os.path.getmtime(net_cache_path)
                mod_dt = datetime.fromtimestamp(int(mod_ts))
                log.info(f"Using cached network updated at {mod_dt}")
                with open(net_cache_path, 'r') as fp:
                    network = JSONObject(json.load(fp))
            else:
                network = client.get_network(
                            network_id=network_id,
                            include_data=False,
                            include_results=False,
                            template_id=template_id,
                            include_attributes=True)
                with open(net_cache_path, 'w') as fp:
                    json.dump(JSONObject(network), fp)
                log.info(f"Cached network written to '{net_cache_path}'")
        else:
            scenario = client.get_scenario(scenario_id=scenario_id, include_data=True, include_results=False, include_metadata=True, include_attr=False)
            network_id = scenario.network_id
            network = client.get_network(
                        network_id=network_id,
                        include_data=False,
                        include_results=False,
                        template_id=template_id,
                        include_attributes=True)

        network.scenarios = [scenario]
        network.rules = client.get_resource_rules(ref_key='NETWORK', ref_id=network_id)

        attributes = client.get_attributes(network_id=network.id, project_id=network.project_id, include_global=True)
        attributes = {attr.id: attr for attr in attributes}

        log.info(f"Retreiving template {network.types[index].template_id}")
        template = client.get_template(template_id=network.types[index].template_id)

        return cls(client, network, network_id, scenario_id, attributes, template, kwargs.get('data_dir'))


    def write_rules_as_module(self):
        filename = os.path.join(os.path.dirname(__file__), "hydra_pywr_custom_module.py")

        prelude = (
            "from pywr import recorders",
            "from pywr import parameters",
            "from pywr.parameters import *",
            "from pywr.recorders import *",
            "import pandas",
            "import numpy as np",
            "import scipy",
            "from pywr.nodes import *",
            "from pywr.parameters.control_curves import *",
            "from pywr.parameters._thresholds import *",
            "from pywr.parameters._hydropower import *",
            "from pywr.domains.river import *"
        )

        forbidden = ("import", "eval", "exec", "__builtins__")

        audit_handler = """
        def handler(event, args):
            forbidden = ('os.', 'subprocess')
            for forbid in forbidden:
                if event.startswith(forbid):
                    raise PermissionError(f"Use of <{forbid}> forbidden in custom rules.")

        sys.addaudithook(handler)
        """

        log.info("Adding %s rules.", len(self.data.rules))

        with open(filename, 'w') as fp:
            for p in prelude:
                fp.write(f"{p}\n")
            fp.write("\n")
            for rule in self.data.rules:
                for forbid in forbidden:
                    if forbid in rule["value"]:
                        raise PermissionError(f"Use of <{forbid}> forbidden in custom rules.")
                fp.write(rule["value"])
                fp.write("\n\n")

    def get_external_files(self):
        """
            Request the locations of any external files from hydra, using the project appdata column,
            then replace the relevant file names in the 'url' section of parameters and tables with the full path to the file.
            for example replace:
            {
            ...
            "url" : "demand.csv"
            ...
            }
            with
            {
            ...
            "url": "/path/to/demand.csv"
            ...
            }
            where '/path/to' is defined in the project's metadata (the appdata column)
        """
        log.info("Retrieving external files")
        try:
            import s3fs
        except ImportError:
            log.error("Unable to check for external files on S3. Access to S3 requires the s3fs module")
            raise

        #assume credential are in the ~/.aws/credentials file
        fs = s3fs.S3FileSystem()

        self.filedict = {}

        appdata = self.data.get('appdata', {})

        #Files uploaded to the USER_FILE_UPLOAD_DIR are synchronized with the USER_FILE_ROOT_DIR
        #So they are then downloaded from the USER_FILE_ROOT_DIR
        if None not in (appdata.get('data_s3_bucket'), appdata.get('data_uuid')):

            network_data_path = appdata.get('data_uuid')

            bucket_path = f"{data_s3_bucket}/data/projectdata/{network_data_path}"

            try:
                #create a mapping from the files nams in the project_data_path directory
                #to to their full s3 path
                networkfiles = fs.ls(bucket_path)
                for s3filepath in networkfiles:
                    self.filedict[os.path.basename(s3filepath)] = s3filepath
            except (FileNotFoundError, PermissionError):
                log.warning("Unable to access bucket %s. Continuing.", bucket_path)

        #First get the project hierarchy
        project_hierarchy = self.hydra.get_project_hierarchy(project_id=self.data['project_id'])

        #start from the top down. Files with the same name, at a lower level
        #take precedence.
        project_hierarchy.reverse()

        for proj_in_hierarchy in project_hierarchy:
            #Files uploaded to the USER_FILE_UPLOAD_DIR are synchronized with the USER_FILE_ROOT_DIR
            #So they are then downloaded from the USER_FILE_ROOT_DIR
            if proj_in_hierarchy.appdata is None:
                continue
            data_s3_bucket = proj_in_hierarchy.appdata.get('data_s3_bucket')

            if data_s3_bucket is None:
                return

            project_data_path = proj_in_hierarchy.appdata.get('data_uuid')

            if project_data_path is None:
                continue

            bucket_path = f"{data_s3_bucket}/data/projectdata/{project_data_path}"

            try:
                #create a mapping from the files nams in the project_data_path directory
                #to to their full s3 path
                projectfiles = fs.ls(bucket_path)
                for s3filepath in projectfiles:
                    self.filedict[os.path.basename(s3filepath)] = s3filepath
            except (FileNotFoundError, PermissionError):
                log.warning("Unable to access bucket %s. Continuing.", bucket_path)

        log.info("External file mapping created for %s files", len(self.filedict))

        return self.filedict


    def sync_with_s3(self, s3_bucket_name, project_data_path):
        """
            Sync the data folder with the specified s3 bucket
        """
        log.info(f"Syncing with s3 bucket {s3_bucket_name}")

        data_dir = os.path.join(self.data_dir, 'projectdata', project_data_path)

        sync_command = f"aws s3 sync s3://{s3_bucket_name}/data/projectdata/{project_data_path} {data_dir}"

        log.info(sync_command)

        completedprocess = subprocess.run(sync_command, shell=True)
        if completedprocess.returncode == 0:
            log.info(f"Synced s3 bucket {s3_bucket_name}")
        else:
            log.warning(f"error syncing bucket {s3_bucket_name} : {completedprocess.stderr} ")

        return data_dir

    def build_pywr_network(self):
        self.build_pywr_nodes()
        self.edges = self.build_edges()

        self.get_external_files()

        parameters, recorders = self.build_parameters_recorders()
        self.parameters.update(parameters)
        self.recorders.update(recorders)

        self.tables = self.build_tables()
        self.timestepper = self.build_timestepper()
        self.metadata = self.build_metadata()
        self.scenarios = self.build_scenarios()
        self.scenario_combinations = self.build_scenario_combinations()

        if len(self.data.rules) > 0:
            self.write_rules_as_module()

        return self


    def build_pywr_nodes(self):

        for node in self.data["nodes"]:
            pywr_node = {"name": node["name"]}

            self.hydra_node_by_id[node["id"]] = node

            if comment := node.get("description"):
                pywr_node["comment"] = comment

            pywr_node_type = node["types"][0]

            if pywr_node_type:
                log.debug(f"Building node <{node['name']}> as <{pywr_node_type['name']} ({pywr_node_type['id']})>")
                self.build_node_and_references(node, pywr_node_type)


    def build_edges(self):
        edges = []

        for hydra_edge in self.data["links"]:
            src_hydra_node = self.hydra_node_by_id[hydra_edge["node_1_id"]]
            dest_hydra_node = self.hydra_node_by_id[hydra_edge["node_2_id"]]
            # Retrieve nodes from PywrNode store to verify presence
            try:
                # NB Lookup nodes with str key: self.nodes is Dict[str:PywrNode]
                src_node = self.nodes[str(src_hydra_node["name"])]
                dest_node = self.nodes[str(dest_hydra_node["name"])]
            except KeyError:
                # Not in this template...
                continue

            verts = [src_node.name, dest_node.name]

            if hydra_edge["types"][0]["name"].lower() == "slottededge":
                for slot in ("src_slot", "dest_slot"):
                    slot_id = [attr.id for attr in hydra_edge["attributes"] if attr.name == slot][0]
                    slot_ds = self.get_dataset_by_resource_attr_id(slot_id)
                    verts.append(slot_ds.value if slot_ds else None)

            edge = PywrEdge(verts)
            edges.append(edge)

        return edges

    def build_tables(self):
        tables = {}
        table_attr_prefix = "tbl_"
        table_subattrs = ("header", "index_col", "key", "url")
        for attr in self.data["attributes"]:
            ds = self.get_dataset_by_resource_attr_id(attr.id)
            if not ds:
                continue
            if ds["type"].upper().startswith("PYWR_TABLE"):
                # New style Table type: single dictionary value
                value = json.loads(ds["value"])
                table = PywrTable(ds["name"], value)
                tables[table.name] = table
            elif attr.name.lower().startswith(table_attr_prefix):
                # Old style deprecated Table: multiple subattrs w common prefix
                tablename = attr.name[len(table_attr_prefix):]
                for k in table_subattrs:
                    if tablename.endswith(f".{k}"):
                        tablename = tablename.replace(f".{k}", "")
                        try:
                            ds["value"] = float(ds["value"])
                        except ValueError:
                            try:
                                ds["value"] = json.loads(ds["value"])
                            except json.decoder.JSONDecodeError:
                                pass
                        if table := tables.get(tablename):
                            table.data[k] = ds["value"]
                        else:
                            table_data = {k: ds["value"]}
                            if k != "url":  # url key required for valid Table
                                table_data.update({"url": None})
                            tables[tablename] = PywrTable(tablename, table_data)
        return tables

    def build_timestepper(self):
        timestep = {}
        ts_attr_prefix = "timestepper"
        ts_keys = ("start", "end", "timestep")

        for attr in self.data["attributes"]:
            ds = self.get_dataset_by_resource_attr_id(attr.id)
            if ds and ds["type"].upper().startswith("PYWR_TIMESTEPPER"):
                # New style Timestep type: single dictionary value
                value = json.loads(ds["value"])
                return PywrTimestepper(value)
            elif ds:
                # Deprecated multi-attr Timestep, must aggregate
                # all subattrs then create instance
                attr_group, *subs = attr.name.split('.')
                if attr_group != ts_attr_prefix:
                    continue
                ts_key = subs[-1]
                try:
                    value = json.loads(ds["value"])
                except json.decoder.JSONDecodeError:
                    value = ds["value"]
                timestep[ts_key] = value
            else:
                continue

        ts_val = timestep.get("timestep",1)
        try:
            tv = int(float(ts_val))
        except ValueError:
            tv = ts_val
        timestep["timestep"] = tv
        return PywrTimestepper(timestep)


    def build_metadata(self):
        metadata = {
            "title": self.data['name'],
            "description": self.data['description']
        }
        for attr in self.data["attributes"]:
            ds = self.get_dataset_by_resource_attr_id(attr.id)
            if ds and ds["type"].upper().startswith("PYWR_METADATA"):
                # New style Metadata type: single dictionary value
                value = json.loads(ds["value"])
                return PywrMetadata(value)
            elif ds:
                # Deprecated multi-attr Metadata, must aggregate
                # all subattrs then create instance
                attr_group, *subs = attr.name.split('.')
                if attr_group != "metadata":
                    continue
                meta_key = subs[-1]
                try:
                    value = json.loads(ds["value"])
                except json.decoder.JSONDecodeError:
                    value = ds["value"]
                metadata[meta_key] = value
            else:
                continue
        """
          minimum_version is an optional metadata key, but
          Pywr requires it to be a string if present.
        """
        minver = metadata.get("minimum_version")
        if minver and not isinstance(minver, str):
            metadata["minimum_version"] = str(minver)

        return PywrMetadata(metadata)


    def build_scenarios(self):
        try:
            scenarios_dataset = self.get_network_attr(self.scenario_id, self.data["id"], "scenarios")
            scenarios = [ PywrScenario(scenario) for scenario in scenarios_dataset["scenarios"] ]
        except (ResourceNotFoundError, ValueError, KeyError):
            scenarios = []

        return scenarios


    def build_scenario_combinations(self):
        try:
            s_c_dataset = self.get_network_attr(self.scenario_id, self.data["id"], "scenario_combinations")

            scenario_combinations = [ PywrScenarioCombination(sc) for sc in s_c_dataset["scenario_combinations"] ]
        except (ResourceNotFoundError, ValueError, KeyError):
            scenario_combinations = []

        return scenario_combinations

    def _get_attribute(self, attr_id):
        attribute = self.attributes.get(attr_id)
        if attribute is None:
            attribute = self.hydra.get_attribute_by_id(attr_id=attr_id)
            self.attributes[attr_id] = attribute
        return attribute

    def build_parameters_recorders(self):
        parameters = {} # {name: P()}
        recorders = {} # {name: R()}

        for resource_attr in filter(lambda x:x.attr_is_var!='Y', self.data.attributes):

            attribute = self._get_attribute(resource_attr["attr_id"])

            ds = self.get_dataset_by_resource_attr_id(resource_attr.id)

            if not ds:
                # This could raise instead, e.g...
                #raise ValueError(f"No dataset found for attr name {attr.name} with id {attr.id}")
                continue

            #a name might have been set directly in the 'build_node_and_references' function to include the node name
            name = resource_attr.get('name', attribute['name'])

            if not ds["type"].upper().startswith(PARAMETER_TYPES + RECORDER_TYPES):
                continue

            if ds["type"].upper().startswith(PARAMETER_TYPES):
                value = json.loads(ds['value'])
                value = utils.unnest_parameter_key(value, key="pandas_kwargs")
                value = utils.add_interp_kwargs(value)

                if value.get('__recorder__') is not None:
                    self._parameter_recorder_flags[name] = value.pop('__recorder__')
                try:
                    p = PywrParameter(name, value)
                except PywrParserException as e:
                    msg = f"!!Output: An error occurred parsing parameter '{name}'. Value is '{value}'. Error is: '{e.message}' Exiting."
                    logging.critical(msg)
                    print(f"!!Output: {msg}")
                    exit(1)

                assert p.name not in parameters    # Disallow overwriting
                parameters[p.name] = p
            elif ds["type"].upper().startswith(RECORDER_TYPES):
                value = json.loads(ds['value'])
                try:
                    r = PywrRecorder(name, value)
                except:
                    raise ValueError(f"Dataset {ds['name']} is not a valid Recorder")
                recorders[r.name] = r

        return parameters, recorders


    def get_network_attr(self, scenario_id, network_id, attr_key):

        net_attr = self.hydra.get_attribute_by_name_and_dimension(name=attr_key, dimension_id=None)
        ra = self.hydra.get_resource_attributes(ref_key="network", ref_id=network_id)
        ra_id = None
        for r in ra:
            try:
                if r["attr_id"] == net_attr["id"]:
                    ra_id = r["id"]
            except KeyError:
                pass

        if not ra_id:
            raise ValueError(f"Resource attribute for {attr_key} not found in scenario {scenario_id} on network {network_id}")

        data = self.hydra.get_resource_scenario(resource_attr_id=ra_id, scenario_id=scenario_id, get_parent_data=False)
        attr_data = json.loads(data["dataset"]["value"])

        return attr_data # NB: String keys


    def get_dataset_by_resource_attr_id(self, ra_id):

        scenario = self.data.scenarios[0]
        for rs in scenario.resourcescenarios:
            if rs.resource_attr_id == ra_id:
                return rs.dataset

    def _get_resource_scenario(self, resource_attribute_id):

        for scenario in self.data["scenarios"]:
            for resource_scenario in scenario["resourcescenarios"]:
                if resource_scenario["resource_attr_id"] == resource_attribute_id:
                    return resource_scenario

        raise ValueError(f"No resource scenario found for resource attribute id: {resource_attribute_id}")


    def build_node_and_references(self, nodedata, pywr_node_type):

        node_type_attribute_names = [a.attr.name for a in self.type_id_map[pywr_node_type['id']].typeattrs]

        for resource_attribute in filter(lambda x:x.attr_is_var!='Y', nodedata["attributes"]):

            attribute = self._get_attribute(resource_attribute["attr_id"])

            try:
                resource_scenario = self._get_resource_scenario(resource_attribute["id"])
            except ValueError:
                continue  # No data associated with this attribute.

            # Allow export of probable recorders
            if resource_attribute["attr_is_var"] == 'Y' and "recorder" not in attribute["name"].lower():
                continue

            attribute_name = attribute["name"]
            dataset = resource_scenario["dataset"]
            dataset_type = dataset["type"]
            value = dataset["value"]

            try:
                typedval = json.loads(value)

                if isinstance(typedval, dict) and typedval.get('__recorder__') is not None:
                    self._parameter_recorder_flags[f"__{nodedata['name']}__:{attribute_name}"] = typedval.pop('__recorder__')

                #If this is a basic hydra dataframe, transform it into a pywr
                #dataframe so the model can read it
                if dataset_type.lower() == 'dataframe' and attribute_name not in ('weather', 'bathymetry', 'release_values'):
                    typedval = {
                        'type': 'dataframeparameter',
                        'data': typedval,
                    }

                if isinstance(typedval, dict):
                    typedval = utils.unnest_parameter_key(typedval, key="pandas_kwargs")
                    typedval = utils.add_interp_kwargs(typedval)
            except json.decoder.JSONDecodeError as e:
                typedval = value

            #If the attribute name is defined on the node type, put it on the node
            if attribute_name in node_type_attribute_names or attribute_name in ('weather', 'bathymetry', 'release_values'):
                nodedata[attribute_name] = typedval
            else:
                #Otherwise put it in the global paramter list, with a name that reflects the original source
                #e.g. "__node name__:attribute_name"
                resource_attribute['name'] = f"__{nodedata['name']}__:{attribute_name}"
                self.data.attributes.append(resource_attribute)

        nodedata["type"] = pywr_node_type['name']
        node_attr_data = {a:v for a,v in nodedata.items() if a not in self.exclude_hydra_attrs}
        position = {"geographic": [ nodedata.get("x",0), nodedata.get("y",0) ]}
        node_attr_data["position"] = position

        if comment := nodedata.get("description"):
            node_attr_data["comment"] = comment

        node = PywrNode(node_attr_data)
        self.nodes[node.name] = node

