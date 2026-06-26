import json
import os
from datetime import datetime

from .config import CACHE_DIR
from . import utils

from hydra_base.lib.objects import JSONObject

import logging
log = logging.getLogger(__name__)


NODE_TYPE_MAP = {
    'catchment': 'Catchment',
    'river': 'River',
    'rivergauge': 'RiverGauge',
    'river_gauge': 'RiverGauge',
    'riversplit': None,  # No v2 equivalent; nodes will be skipped
    'riversplitwithgauge': 'RiverSplitWithGauge',
    'river_split': 'RiverSplitWithGauge',
    'reservoir': 'Reservoir',
    'storage': 'Storage',
    'link': 'Link',
    'input': 'Input',
    'output': 'Output',
    'wtw': 'WaterTreatmentWorks',
    'water_treatment_works': 'WaterTreatmentWorks',
    'turbine': 'Turbine',
    'loss_link': 'LossLink',
    'losslink': 'LossLink',
    'delay': 'Delay',
    'piecewise_link': 'PiecewiseLink',
    'piecewiselink': 'PiecewiseLink',
}

PARAMETER_TYPE_MAP = {
    'constant': 'Constant',
    'constantparameter': 'Constant',
    'monthly_profile': 'MonthlyProfile',
    'monthlyprofile': 'MonthlyProfile',
    'monthlyprofileparameter': 'MonthlyProfile',
    'daily_profile': 'DailyProfile',
    'dailyprofile': 'DailyProfile',
    'dailyprofileparameter': 'DailyProfile',
    'weekly_profile': 'WeeklyProfile',
    'weeklyprofile': 'WeeklyProfile',
    'weeklyprofileparameter': 'WeeklyProfile',
    'aggregated': 'Aggregated',
    'aggregatedparameter': 'Aggregated',
    'control_curve': 'ControlCurve',
    'controlcurve': 'ControlCurve',
    'controlcurveparameter': 'ControlCurve',
    'controlcurveinterpolatedparameter': 'ControlCurveInterpolated',
    'interpolated': 'Interpolated',
    'interpolatedparameter': 'Interpolated',
    'threshold': 'Threshold',
    'parameterthresholdparameter': 'Threshold',
    'storagethresholdparameter': 'Threshold',
    'python': 'Python',
    'pythonparameter': 'Python',
    # dataframe/interpolatedvolume have no direct v2 equivalent in this schema version
}

PARAMETER_TYPES = (
    "PYWR_PARAMETER",
    "PYWR_DATAFRAME",
)

RECORDER_TYPES = (
    "PYWR_RECORDER",
)

EXCLUDE_NODE_KEYS = (
    "id", "status", "cr_date",
    "network_id", "x", "y",
    "types", "attributes", "layout",
    "network", "description",
)

# Fields not recognised by v2 schema for specific node types (silently drop)
_NODE_TYPE_SKIP_FIELDS = {
    'Reservoir': {'level', 'area'},
    'Storage': {'level', 'area', 'evaporation', 'rainfall'},
    'Catchment': {'min_flow', 'max_flow'},
    'River': {'min_flow', 'max_flow', 'cost', 'flow'},
    'RiverGauge': {'min_flow', 'max_flow', 'cost', 'flow', 'loss_factor'},
    'RiverSplitWithGauge': {'min_flow', 'max_flow', 'cost', 'flow', 'loss_factor'},
    'LossLink': {'min_flow', 'max_flow', 'cost'},
    'Turbine': {'storage_node', 'generation_capacity'},
}

# Fields that require {"data": metric} wrapping on Reservoir
_DATA_WRAPPED_RESERVOIR_FIELDS = {'evaporation', 'rainfall'}

# Storage-like node types that use Absolute/Proportional initial_volume
_STORAGE_NODE_TYPES = {'Reservoir', 'Storage'}

# Node fields that must be raw scalars (f64) rather than Metric objects
_NODE_SCALAR_FIELDS = {
    'Turbine': {'turbine_elevation', 'min_head', 'efficiency', 'water_density',
                'flow_unit_conversion', 'energy_unit_conversion'},
}

# Hydra attribute names → v2 field names (per node type)
_NODE_FIELD_RENAMES = {
    'Turbine': {'density': 'water_density', 'min_operating_elevation': 'min_head'},
}


def _wrap_metric(value):
    """Wrap a scalar or parameter-reference string as a v2 metric object."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return {"type": "Literal", "value": value}
    if isinstance(value, str):
        return {"type": "Parameter", "name": value}
    return value  # dicts, lists pass through unchanged


def _map_node_type(raw_type):
    return NODE_TYPE_MAP.get(raw_type.lower())


def _map_parameter_type(raw_type):
    return PARAMETER_TYPE_MAP.get(raw_type.lower())


_PROFILE_TYPES = {"MonthlyProfile", "DailyProfile", "WeeklyProfile"}
_CONTROL_CURVE_TYPES = {"ControlCurve", "ControlCurveInterpolated", "ControlCurveIndex"}

# v1 optimization / recorder fields that have no v2 equivalent
_PARAM_SKIP_KEYS = {"is_variable", "lower_bounds", "upper_bounds", "is_double_variable",
                    "double_lower_bounds", "double_upper_bounds", "variable_name",
                    "comment", "epsilon"}


def _to_v2_parameter(name, value_dict):
    """Convert a v1 parameter dict to a v2 array-element dict, or None if type is unknown."""
    raw_type = value_dict.get("type", "")
    v2_type = _map_parameter_type(raw_type)

    if v2_type is None:
        log.warning("Unknown parameter type '%s' for parameter '%s'; skipping", raw_type, name)
        return None

    p = {"meta": {"name": name}, "type": v2_type}

    for key, val in value_dict.items():
        if key == "type" or key in _PARAM_SKIP_KEYS:
            continue

        if v2_type == "Constant" and key == "value":
            p[key] = _wrap_metric(val)

        elif v2_type in _PROFILE_TYPES and key == "values":
            if isinstance(val, list):
                p[key] = {"type": "Literal", "values": [float(v) for v in val]}
            else:
                p[key] = val

        elif v2_type == "Aggregated" and key == "parameters":
            p["metrics"] = [_wrap_metric(v) if isinstance(v, str) else v for v in val]

        elif v2_type == "Aggregated" and key == "agg_func":
            if isinstance(val, str):
                p[key] = {"type": val.capitalize()}
            else:
                p[key] = val

        elif v2_type == "Interpolated":
            # v1: x (x-points), values (y-points), parameter (input metric)
            # v2: x (input metric), xp (x-points as Metrics), fp (y-points as Metrics)
            if key == "x":
                p["xp"] = [_wrap_metric(v) for v in val] if isinstance(val, list) else val
            elif key == "values":
                p["fp"] = [_wrap_metric(v) for v in val] if isinstance(val, list) else val
            elif key == "parameter":
                p["x"] = _wrap_metric(val) if not isinstance(val, dict) else val
            else:
                p[key] = val

        elif v2_type in _CONTROL_CURVE_TYPES:
            if key == "storage_node":
                # v1 storage_node string → v2 storage_metric Node reference
                p["storage_metric"] = {"type": "Node", "name": val}
            elif key == "control_curves":
                p[key] = [_wrap_metric(v) for v in val] if isinstance(val, list) else val
            elif key == "values" and v2_type != "ControlCurveIndex":
                # ControlCurve/ControlCurveInterpolated values are Metrics; ControlCurveIndex has no values
                p[key] = [_wrap_metric(v) for v in val] if isinstance(val, list) else val
            elif key == "values" and v2_type == "ControlCurveIndex":
                pass  # ControlCurveIndex has no values field; skip
            else:
                p[key] = val

        else:
            p[key] = val

    return p


class HydraToPywrV2Network:

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
            if tt.status and tt.status.upper() == 'X':
                continue
            self.type_id_map[tt.id] = tt
            if tt.parent_id is not None:
                self.type_id_map[tt.parent_id] = tt

        self.ra_dataset_map = {}
        self.hydra_node_by_id = {}
        self._processed_node_names = set()
        self._promoted_parameters = {}  # name -> v2 param dict

    @classmethod
    def from_scenario_id(cls, client, scenario_id, template_id=None, index=0, **kwargs):
        if kwargs.get("use_cache") is True:
            if not os.path.exists(CACHE_DIR):
                try:
                    os.mkdir(CACHE_DIR)
                except OSError:
                    log.error("Unable to create scenario cache at %s", CACHE_DIR)

            scen_cache_path = os.path.join(CACHE_DIR, f"scenario_{scenario_id}.json")
            if os.path.exists(scen_cache_path):
                mod_dt = datetime.fromtimestamp(int(os.path.getmtime(scen_cache_path)))
                log.info("Using cached scenario updated at %s", mod_dt)
                with open(scen_cache_path) as fp:
                    scenario = JSONObject(json.load(fp))
            else:
                scenario = client.get_scenario(scenario_id=scenario_id,
                                               include_data=True,
                                               include_results=False,
                                               include_metadata=False,
                                               include_attr=False)
                with open(scen_cache_path, 'w') as fp:
                    json.dump(scenario, fp)
                log.info("Cached scenario written to '%s'", scen_cache_path)

            network_id = scenario.network_id
            net_cache_path = os.path.join(CACHE_DIR, f"network_{network_id}.json")
            if os.path.exists(net_cache_path):
                mod_dt = datetime.fromtimestamp(int(os.path.getmtime(net_cache_path)))
                log.info("Using cached network updated at %s", mod_dt)
                with open(net_cache_path) as fp:
                    network = JSONObject(json.load(fp))
            else:
                network = client.get_network(network_id=network_id,
                                             include_data=False,
                                             include_results=False,
                                             template_id=template_id,
                                             include_attributes=True)
                with open(net_cache_path, 'w') as fp:
                    json.dump(JSONObject(network), fp)
                log.info("Cached network written to '%s'", net_cache_path)
        else:
            scenario = client.get_scenario(scenario_id=scenario_id,
                                           include_data=True,
                                           include_results=False,
                                           include_metadata=False,
                                           include_attr=False)
            network_id = scenario.network_id
            network = client.get_network(network_id=network_id,
                                         include_data=False,
                                         include_results=False,
                                         template_id=template_id,
                                         include_attributes=True)

        network.scenarios = [scenario]

        attributes = client.get_attributes(network_id=network.id,
                                           project_id=network.project_id,
                                           include_hierarchy=True,
                                           include_global=True)
        attributes = {attr.id: attr for attr in attributes}

        log.info("Retrieving template %s", network.types[index].template_id)
        template = client.get_template(template_id=network.types[index].template_id)

        return cls(client, network, network_id, scenario_id, attributes, template,
                   kwargs.pop('data_dir', '.'), **kwargs)

    def build(self):
        metadata = self._build_metadata()
        timestepper = self._build_timestepper()
        scenarios = self._build_scenarios()

        # Nodes must be built before edges (populates hydra_node_by_id and _processed_node_names)
        nodes = self._build_nodes()
        edges = self._build_edges()
        parameters = self._build_parameters()

        # Merge in parameters promoted from nodes during _build_nodes
        existing_names = {p["meta"]["name"] for p in parameters}
        for name, p in self._promoted_parameters.items():
            if name not in existing_names:
                parameters.append(p)

        output = {
            "metadata": metadata,
            "timestepper": timestepper,
            "network": {
                "nodes": nodes,
                "edges": edges,
                "parameters": parameters,
                "metric_sets": [{"name": "all", "filters": {"all_nodes": True}}],
                "outputs": [{"name": "all", "type": "Memory", "metric_set": "all"}],
            },
        }
        if scenarios:
            output["scenarios"] = scenarios
        return output

    # ── Metadata ────────────────────────────────────────────────────────────

    def _build_metadata(self):
        metadata = {
            "title": self.data['name'].replace(' ', '_').replace('/', '_'),
            "description": self.data.get('description') or '',
        }
        for attr in self.data["attributes"]:
            ds = self.get_dataset_by_resource_attr_id(attr.id)
            if not ds:
                continue
            if ds["type"].upper().startswith("PYWR_METADATA"):
                metadata.update(json.loads(ds["value"]))
                break
            attr_group, *subs = attr.name.split('.')
            if attr_group != "metadata" or not subs:
                continue
            try:
                value = json.loads(ds["value"])
            except (json.decoder.JSONDecodeError, TypeError):
                value = ds["value"]
            metadata[subs[-1]] = value

        minver = metadata.get("minimum_version")
        if minver is not None and not isinstance(minver, str):
            metadata["minimum_version"] = str(minver)

        return metadata

    # ── Timestepper ─────────────────────────────────────────────────────────

    def _build_timestepper(self):
        ts_data = {}
        for attr in self.data["attributes"]:
            ds = self.get_dataset_by_resource_attr_id(attr.id)
            if not ds:
                continue
            if ds["type"].upper().startswith("PYWR_TIMESTEPPER"):
                ts_data = json.loads(ds["value"])
                break
            attr_group, *subs = attr.name.split('.')
            if attr_group != "timestepper" or not subs:
                continue
            try:
                value = json.loads(ds["value"])
            except (json.decoder.JSONDecodeError, TypeError):
                value = ds["value"]
            ts_data[subs[-1]] = value

        raw_ts = ts_data.get("timestep", 1)
        if isinstance(raw_ts, str) and raw_ts.upper() == 'M':
            timestep = {"type": "Frequency", "freq": "1MS"}
        else:
            try:
                timestep = {"type": "Days", "days": int(float(raw_ts))}
            except (TypeError, ValueError):
                log.warning("Could not parse timestep value '%s'; defaulting to 1 day", raw_ts)
                timestep = {"type": "Days", "days": 1}

        return {
            "start": ts_data.get("start", "2000-01-01"),
            "end": ts_data.get("end", "2000-12-31"),
            "timestep": timestep,
        }

    # ── Scenarios ────────────────────────────────────────────────────────────

    def _build_scenarios(self):
        try:
            data = self._get_network_attr("scenarios")
            return data.get("scenarios", [])
        except Exception:
            log.warning("Unable to build scenarios")
            return []

    # ── Nodes ────────────────────────────────────────────────────────────────

    def _build_nodes(self):
        log.info("Building v2 nodes")
        nodes = []
        for node in self.data["nodes"]:
            self.hydra_node_by_id[node["id"]] = node

            if not node.get("types"):
                log.warning("Node '%s' has no type; skipping", node["name"])
                continue

            pywr_node_type = node["types"][0]
            v2_type = _map_node_type(pywr_node_type["name"])
            if v2_type is None:
                log.warning("Unknown node type '%s' for node '%s'; skipping",
                            pywr_node_type["name"], node["name"])
                continue

            meta = {
                "name": node["name"],
                "position": {"schematic": [node.get("x", 0), node.get("y", 0)]},
            }
            if comment := node.get("description"):
                meta["comment"] = comment

            v2_node = {"meta": meta, "type": v2_type}
            self._process_node_attributes(node, pywr_node_type, v2_node)

            nodes.append(v2_node)
            self._processed_node_names.add(node["name"])

        return nodes

    def _process_node_attributes(self, nodedata, pywr_node_type, v2_node):
        if pywr_node_type["id"] not in self.type_id_map:
            self.type_id_map[pywr_node_type["id"]] = self.hydra.get_templatetype(
                {"type_id": pywr_node_type["id"]}
            )

        node_type_attr_names = {a.attr.name for a in self.type_id_map[pywr_node_type["id"]].typeattrs}

        for resource_attr in filter(lambda x: x.attr_is_var != 'Y', nodedata["attributes"]):
            attribute = self._get_attribute(resource_attr["attr_id"])

            try:
                resource_scenario = self._get_resource_scenario(resource_attr["id"])
            except ValueError:
                continue

            attribute_name = attribute["name"]
            dataset = resource_scenario["dataset"]

            if dataset["type"].upper().startswith(RECORDER_TYPES):
                log.debug("Skipping recorder '%s': recorders are not supported in v2", attribute_name)
                continue

            try:
                typedval = json.loads(dataset["value"])
                if isinstance(typedval, dict):
                    typedval = utils.unnest_parameter_key(typedval, key="pandas_kwargs")
                    typedval = utils.add_interp_kwargs(typedval)
            except (json.decoder.JSONDecodeError, TypeError):
                typedval = dataset["value"]

            v2_type = v2_node["type"]

            # Apply per-node-type field renames (Hydra name → v2 name)
            renames = _NODE_FIELD_RENAMES.get(v2_type, {})
            attribute_name = renames.get(attribute_name, attribute_name)

            # Drop fields the v2 schema doesn't know about for this node type
            skip_fields = _NODE_TYPE_SKIP_FIELDS.get(v2_type, set())
            if attribute_name in skip_fields:
                log.debug("Skipping field '%s': not valid for v2 node type '%s'", attribute_name, v2_type)
                continue

            # initial_volume must be Absolute (not Literal)
            if attribute_name == 'initial_volume' and v2_type in _STORAGE_NODE_TYPES:
                if isinstance(typedval, (int, float)):
                    v2_node['initial_volume'] = {"type": "Absolute", "volume": float(typedval)}
                else:
                    v2_node['initial_volume'] = typedval
                continue

            # initial_volume_pc → Proportional (0-100 scale → 0-1)
            if attribute_name == 'initial_volume_pc' and v2_type in _STORAGE_NODE_TYPES:
                if isinstance(typedval, (int, float)):
                    v2_node['initial_volume'] = {"type": "Proportional", "proportion": typedval / 100.0}
                continue

            # evaporation/rainfall on Reservoir need {"data": metric}
            if attribute_name in _DATA_WRAPPED_RESERVOIR_FIELDS and v2_type == 'Reservoir':
                v2_node[attribute_name] = {"data": _wrap_metric(typedval)}
                continue

            # WTW loss_factor: {"type": "Gross"/"Net", "factor": metric}
            if attribute_name == 'loss_factor' and v2_type == 'WaterTreatmentWorks':
                lf_type = 'Net'
                if isinstance(typedval, dict) and typedval.get('type', '').lower() in ('gross', 'net'):
                    lf_type = typedval['type'].capitalize()
                    factor_val = typedval.get('factor', typedval.get('value', 0.0))
                else:
                    factor_val = typedval
                v2_node['loss_factor'] = {"type": lf_type, "factor": _wrap_metric(factor_val)}
                continue

            # Fields that must be raw scalars (not Metric objects)
            scalar_fields = _NODE_SCALAR_FIELDS.get(v2_type, set())
            if attribute_name in scalar_fields:
                if isinstance(typedval, (int, float)):
                    v2_node[attribute_name] = float(typedval)
                elif isinstance(typedval, dict) and typedval.get('type') == 'Literal':
                    v2_node[attribute_name] = float(typedval.get('value', 0.0))
                else:
                    v2_node[attribute_name] = typedval
                continue

            if attribute_name in node_type_attr_names or attribute_name in ('weather', 'bathymetry', 'release_values'):
                v2_node[attribute_name] = _wrap_metric(typedval)
            else:
                # Promote to parameters list; reference from node via Parameter metric
                param_name = f"__{nodedata['name']}__:{attribute_name}"
                if isinstance(typedval, dict):
                    p = _to_v2_parameter(param_name, typedval)
                    if p:
                        self._promoted_parameters[param_name] = p
                else:
                    self._promoted_parameters[param_name] = {
                        "meta": {"name": param_name},
                        "type": "Constant",
                        "value": _wrap_metric(typedval),
                    }
                v2_node[attribute_name] = {"type": "Parameter", "name": param_name}

    # ── Edges ────────────────────────────────────────────────────────────────

    def _build_edges(self):
        log.info("Building v2 edges")
        edges = []

        for hydra_edge in self.data["links"]:
            src_node = self.hydra_node_by_id.get(hydra_edge["node_1_id"])
            dest_node = self.hydra_node_by_id.get(hydra_edge["node_2_id"])

            if src_node is None or dest_node is None:
                continue

            src_name = src_node["name"]
            dest_name = dest_node["name"]

            if src_name not in self._processed_node_names or dest_name not in self._processed_node_names:
                continue

            edge = {"from_node": src_name, "to_node": dest_name}

            edge_type_name = (hydra_edge["types"][0]["name"].lower()
                              if hydra_edge.get("types") else "")
            if edge_type_name == "slottededge":
                for slot_attr, v2_key in (("src_slot", "from_slot"), ("dest_slot", "to_slot")):
                    matching = [a for a in hydra_edge["attributes"] if a.name == slot_attr]
                    if matching:
                        slot_ds = self.get_dataset_by_resource_attr_id(matching[0].id)
                        slot_val = slot_ds["value"] if slot_ds else None
                        if slot_val:
                            edge[v2_key] = {"type": slot_val}

            edges.append(edge)

        return edges

    # ── Parameters ───────────────────────────────────────────────────────────

    def _build_parameters(self):
        log.info("Building v2 parameters")
        parameters = []

        for resource_attr in filter(lambda x: x.attr_is_var != 'Y', self.data.attributes):
            attribute = self._get_attribute(resource_attr["attr_id"])
            ds = self.get_dataset_by_resource_attr_id(resource_attr.id)

            if not ds:
                continue

            ds_type = ds["type"].upper()

            if ds_type.startswith(RECORDER_TYPES):
                log.debug("Skipping recorder '%s': recorders are not supported in v2",
                          attribute["name"])
                continue

            if not ds_type.startswith(PARAMETER_TYPES):
                continue

            name = resource_attr.get('name', attribute['name'])

            try:
                value = json.loads(ds['value'])
            except (json.decoder.JSONDecodeError, TypeError):
                log.warning("Could not parse parameter value for '%s'; skipping", name)
                continue

            if not isinstance(value, dict):
                log.warning("Unexpected parameter value type for '%s': %s; skipping",
                            name, type(value).__name__)
                continue

            value = utils.unnest_parameter_key(value, key="pandas_kwargs")
            value = utils.add_interp_kwargs(value)

            p = _to_v2_parameter(name, value)
            if p:
                parameters.append(p)

        return parameters

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_attribute(self, attr_id):
        attr = self.attributes.get(attr_id)
        if attr is None:
            attr = self.hydra.get_attribute_by_id(attr_id=attr_id)
            self.attributes[attr_id] = attr
        return attr

    def _get_network_attr(self, attr_key):
        net_attr = self.hydra.get_attribute_by_name_and_dimension(name=attr_key, dimension_id=None)
        ra = self.hydra.get_resource_attributes(ref_key="network", ref_id=self.network_id)
        ra_id = next((r["id"] for r in ra if r.get("attr_id") == net_attr["id"]), None)
        if not ra_id:
            raise ValueError(f"Resource attribute '{attr_key}' not found on network {self.network_id}")
        data = self.hydra.get_resource_scenario(resource_attr_id=ra_id,
                                                scenario_id=self.scenario_id,
                                                get_parent_data=False)
        return json.loads(data["dataset"]["value"])

    def make_ra_dataset_map(self):
        for rs in self.data.scenarios[0].resourcescenarios:
            if rs.resource_attr_id not in self.ra_dataset_map:
                self.ra_dataset_map[rs.resource_attr_id] = rs.dataset

    def get_dataset_by_resource_attr_id(self, ra_id):
        if not self.ra_dataset_map:
            self.make_ra_dataset_map()
        if ra_id in self.ra_dataset_map:
            return self.ra_dataset_map[ra_id]
        for rs in self.data.scenarios[0].resourcescenarios:
            if rs.resource_attr_id == ra_id:
                return rs.dataset
        return None

    def _get_resource_scenario(self, resource_attribute_id):
        for scenario in self.data["scenarios"]:
            for rs in scenario["resourcescenarios"]:
                if rs["resource_attr_id"] == resource_attribute_id:
                    return rs
        raise ValueError(f"No resource scenario for resource attribute id: {resource_attribute_id}")


def export_json_v2(client, data_dir, scenario_id, json_sort_keys=False, json_indent=2):
    exporter = HydraToPywrV2Network.from_scenario_id(client, scenario_id, data_dir=data_dir)
    output = exporter.build()

    title = output["metadata"].get("title", f"network_{exporter.network_id}")
    outfile = os.path.join(data_dir, f"{title}_v2.json")
    with open(outfile, 'w') as fp:
        json.dump(output, fp, sort_keys=json_sort_keys, indent=json_indent)

    try:
        from pywr import ModelSchema
        ModelSchema.from_json_string(json.dumps(output))
        log.info("Schema validation passed")
    except ImportError:
        log.warning(
            "pywr package not installed; skipping schema validation. "
            "Install with: pip install hydra-pywr[v2]"
        )
    except Exception as e:
        log.error("Schema validation failed: %s", e)
        log.error("Output written to '%s' for inspection", outfile)
        raise

    log.info("Network %s, Scenario %s exported to '%s'",
             exporter.network_id, scenario_id, outfile)
    return outfile
