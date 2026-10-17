"""
Helpers for running hydra-pywr against an in-process Hydra (hydra_base + temporary SQLite database).
"""
import json
import os

from hydra_client.connection import JSONConnection
from hydra_base.lib.objects import JSONObject
from sqlalchemy.engine import Row

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'data', 'templates')


def import_template_file(client, filename):
    """Import a Hydra-format template JSON file, returning the template as stored in the database.

    hydra_pywr.template.register_template can no longer be used: the functions it depends on
    (generate_pywr_attributes, generate_pywr_node_templates) have been removed from the package.
    """
    with open(os.path.join(TEMPLATE_DIR, filename)) as fh:
        template_json = fh.read()
    name = json.loads(template_json)['template']['name']
    client.import_template_json(template_json)
    return client.get_template_by_name(name)


class LocalHydraClient(JSONConnection):
    """In-process client that accepts the remote API's calling conventions.

    hydra-pywr is written for the remote JSON API, where a single dict positional argument carries
    named parameters (e.g. add_network({"net": ...})) and some parameter names differ from the
    hydra_base function signatures (net/network, scen/scenario, attrs/attributes). The local
    JSONConnection passes arguments straight through to hydra_base, so translate them here.
    """
    # remote-API parameter name -> hydra_base parameter name, per function
    PARAM_NAMES = {
        'add_network': {'net': 'network'},
        'add_scenario': {'scen': 'scenario'},
        'add_attributes': {'attrs': 'attrs'},
        'get_templatetype': {'type_id': 'type_id'},
    }

    def call(self, func_name, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs and func_name in self.PARAM_NAMES:
            mapping = self.PARAM_NAMES[func_name]
            wrapped = args[0]
            if any(k in wrapped for k in mapping):
                kwargs = {mapping.get(k, k): v for k, v in wrapped.items()}
                args = ()
        return self._rows_to_dicts(super().call(func_name, *args, **kwargs))

    @classmethod
    def _rows_to_dicts(cls, value):
        """Some hydra_base queries return raw SQLAlchemy rows inside the result (e.g. a node's
        'attributes'), which the remote API serialises as dicts. Do the same here."""
        if isinstance(value, Row):
            return JSONObject({k: cls._rows_to_dicts(v) for k, v in value._mapping.items()})
        if isinstance(value, dict):
            for k in list(value.keys()):
                value[k] = cls._rows_to_dicts(value[k])
            return value
        if isinstance(value, (list, tuple)):
            return [cls._rows_to_dicts(v) for v in value]
        return value
