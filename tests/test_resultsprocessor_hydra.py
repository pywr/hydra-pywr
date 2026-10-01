"""
Regression tests for HydraResultsProcessor.add_resource_attributes.

Bug: a shared Attr (e.g. 'simulated_flow') can be used both as a NETWORK-level
resource attribute and as a per-NODE resource attribute across a network's
nodes. add_resource_attributes used to check "does this attr_id have a
network-level resource attribute?" *before* checking whether the recorder was
actually tied to a node -- so as soon as a network ever ended up with a single
stray NETWORK-level resource attribute for that Attr (e.g. because one
recorder couldn't be matched to a node on some earlier run), *every* node's
recorder for that same attribute name would be redirected to that one
network-level resource attribute on every subsequent run, and per-node results
would stop being stored. Deleting the network-level resource attribute made
things work again because it removed the short-circuit.
"""
from types import SimpleNamespace

from hydra_base.lib.objects import JSONObject

from hydra_pywr.resultsprocessor.hydra import HydraResultsProcessor


class FakePywrNode:
    """A stand-in for a pywr Node -- only what _get_pywr_node_from_recorder /
    _get_hydra_node_from_recorder actually touch."""
    def __init__(self, name):
        self.name = name
        self.parent = None


class FakePywrModel:
    def __init__(self, nodes):
        self._nodes = {n.name: n for n in nodes}

    @property
    def nodes(self):
        return self._nodes


class FakeRecorder:
    """A stand-in for a pywr recorder with a '__NodeName__:attr' style name."""
    def __init__(self, name, model):
        self.name = name
        self.model = model


def _make_results_processor(hydra_network, hydra_attributes, attr_dimension_map, node_lookup, node_attr_lookup):
    # Bypass __init__ (which pulls in template/S3/H5 setup unrelated to this logic)
    # and set only the state add_resource_attributes actually depends on.
    rp = HydraResultsProcessor.__new__(HydraResultsProcessor)
    rp.hydra_network = hydra_network
    rp.hydra_client = None
    rp.hydra_attributes = hydra_attributes
    rp.attr_dimension_map = attr_dimension_map
    rp.attr_unit_map = {}
    rp.node_lookup = node_lookup
    rp.node_attr_lookup = node_attr_lookup
    return rp


class TestAddResourceAttributes:
    def test_node_recorder_not_hijacked_by_network_level_attribute(self):
        """
        A network-level resource attribute for 'simulated_flow' must not steal
        the result of a node-specific 'simulated_flow' recorder -- the node's
        own resource attribute should still be used.
        """
        simulated_flow_attr_id = 501

        node_ra = JSONObject({'id': 9001, 'attr_id': simulated_flow_attr_id, 'name': 'simulated_flow'})
        hydra_node = JSONObject({'id': 1, 'name': 'MyNode', 'attributes': [node_ra]})

        # A pre-existing (e.g. erroneously created) NETWORK-level resource
        # attribute for the *same* Attr as the node's.
        network_ra = JSONObject({'id': 9002, 'attr_id': simulated_flow_attr_id, 'name': 'simulated_flow'})
        hydra_network = JSONObject({
            'id': 100,
            'nodes': [hydra_node],
            'attributes': [network_ra],
        })

        hydra_attributes = {
            simulated_flow_attr_id: JSONObject({
                'id': simulated_flow_attr_id, 'name': 'simulated_flow', 'dimension_id': None
            })
        }

        rp = _make_results_processor(
            hydra_network=hydra_network,
            hydra_attributes=hydra_attributes,
            attr_dimension_map={'simulated_flow': None},
            node_lookup={'MyNode': hydra_node},
            node_attr_lookup={'MyNode': {simulated_flow_attr_id: node_ra}},
        )

        pywr_node = FakePywrNode('MyNode')
        model = FakePywrModel([pywr_node])
        recorder = FakeRecorder('__MyNode__:simulated_flow', model)

        result_map = rp.add_resource_attributes([recorder], is_dataframe=True)

        # Must resolve to the node's own resource attribute, not the network's.
        assert result_map[recorder.name] == node_ra['id']
        assert result_map[recorder.name] != network_ra['id']

    def test_genuinely_network_level_recorder_still_uses_network_attribute(self):
        """
        A recorder with no associated pywr node should still resolve to the
        existing network-level resource attribute, as before.
        """
        simulated_something_attr_id = 777

        network_ra = JSONObject({'id': 42, 'attr_id': simulated_something_attr_id, 'name': 'simulated_something'})
        hydra_network = JSONObject({
            'id': 100,
            'nodes': [],
            'attributes': [network_ra],
        })

        hydra_attributes = {
            simulated_something_attr_id: JSONObject({
                'id': simulated_something_attr_id, 'name': 'simulated_something', 'dimension_id': None
            })
        }

        rp = _make_results_processor(
            hydra_network=hydra_network,
            hydra_attributes=hydra_attributes,
            attr_dimension_map={'simulated_something': None},
            node_lookup={},
            node_attr_lookup={},
        )

        # No ':' in the name and no matching pywr node -> _get_pywr_node_from_recorder
        # returns None via the AttributeError fallbacks.
        recorder = SimpleNamespace(name='simulated_something')

        result_map = rp.add_resource_attributes([recorder], is_dataframe=True)

        assert result_map[recorder.name] == network_ra['id']


class TestLossLinkAggregatedCompanionRecorder:
    """
    LossLink (a compound node type used e.g. for 'losslink'-templated nodes)
    creates an internal, parentless AggregatedNode named "<node name>
    Aggregated" as an implementation detail. runner.py's automatic per-node
    recorder logic creates a 'simulated_flow' recorder for it just like any
    other node, but it has no Hydra-side counterpart and no parent to fall
    back to -- so it used to fall through to the NETWORK-level catch-all,
    creating a stray network-level 'simulated_flow' resource attribute on
    every run.
    """

    def test_losslink_aggregated_companion_recorder_is_ignored(self):
        from pywr.model import Model
        from pywr.nodes import LossLink, Input, Output as PywrNodeOutput

        model = Model()
        loss_node = LossLink(model, name="Dargom losses", loss_factor=0.1)
        inp = Input(model, name="Inp", max_flow=10)
        out = PywrNodeOutput(model, name="Out")
        inp.connect(loss_node)
        loss_node.connect(out)

        # Exactly what runner.py's _add_node_flagged_recorders does.
        from pywr.nodes import Node, AggregatedNode
        from pywr.recorders import NumpyArrayNodeRecorder
        recorders = []
        for node in model.nodes:
            if isinstance(node, (Node, AggregatedNode)):
                name = '__{}__:{}'.format(node.name, 'simulated_flow')
                recorders.append(NumpyArrayNodeRecorder(model, node, name=name))

        recorder_names = sorted(r.name for r in recorders)
        assert recorder_names == [
            '__Dargom losses Aggregated__:simulated_flow',
            '__Dargom losses__:simulated_flow',
            '__Inp__:simulated_flow',
            '__Out__:simulated_flow',
        ]

        simulated_flow_attr_id = 501
        node_ra = JSONObject({'id': 9001, 'attr_id': simulated_flow_attr_id, 'name': 'simulated_flow'})
        hydra_node = JSONObject({'id': 1, 'name': 'Dargom losses', 'attributes': [node_ra]})
        hydra_network = JSONObject({
            'id': 100,
            'nodes': [hydra_node],
            'attributes': [],  # no network-level resource attribute (yet)
        })
        hydra_attributes = {
            simulated_flow_attr_id: JSONObject({
                'id': simulated_flow_attr_id, 'name': 'simulated_flow', 'dimension_id': None
            })
        }

        rp = _make_results_processor(
            hydra_network=hydra_network,
            hydra_attributes=hydra_attributes,
            attr_dimension_map={'simulated_flow': None},
            node_lookup={'Dargom losses': hydra_node},
            node_attr_lookup={'Dargom losses': {simulated_flow_attr_id: node_ra}},
        )

        # Capture what would be sent to Hydra to create new resource attributes,
        # instead of actually calling out to a (nonexistent, in this test) client.
        added_resource_attributes = []
        rp._add_resource_attributes = lambda ras: added_resource_attributes.extend(ras)

        # Only give it the two recorders belonging to the loss node itself.
        loss_recorders = [r for r in recorders if 'Dargom losses' in r.name]
        result_map = rp.add_resource_attributes(loss_recorders, is_dataframe=True)

        # The main node's recorder resolves to the real node resource attribute.
        assert result_map['__Dargom losses__:simulated_flow'] == node_ra['id']

        # The orphaned "Aggregated" companion recorder must be ignored entirely --
        # no entry in the result map, and (critically) no attempt to create a
        # NETWORK-level resource attribute for it.
        assert '__Dargom losses Aggregated__:simulated_flow' not in result_map
        assert added_resource_attributes == []
