import unittest
import json
from Lib.utilities import get_dict, is_in_dict
from Exporter.PywrJsonWriter import get_recotds
from HydraLib.PluginLib import JSONPlugin
from Importer.PywrJsonReader import  get_pywr_json_from_file, import_net

class Con_pars():
    def __init__(self):
        self.session_id = None
        self.server_url = None

class HydraConnector(JSONPlugin):
    def __init__(self):
        self.connect(Con_pars())

def test_domain(pywr_mode, hydra_model):
    pass

class Pywr_to_hydra_importer(unittest.TestCase):
    def setUp(self):
        filename = r"demand_saving2_with_variables.json"
        self.pywr_model, json_list = get_pywr_json_from_file(filename)
        connector = HydraConnector()
        hydra_network, nodes_types, links_types = import_net(filename, connector.connection)
        self.hydra_model = get_dict(hydra_network)

    def test_recorders(self):
        pywr_recorders = {}
        hydra_recorders = {}
        if ("recorders" in self.pywr_model):
            pywr_recorders = get_dict(self.pywr_model['recorders'])
        if ("recorders" in self.hydra_model):
            hydra_recorders = get_recotds(self.hydra_model['recorders'])
        assert len(hydra_recorders) == len(pywr_recorders)
        for key in pywr_recorders:
            # firsttest is the recorder is added
            print "Testing record: ", key
            assert key in hydra_recorders;
            "record was not added!"
            # for each recorder it tests all the asscoiated attributes
            for key_ in pywr_recorders[key]:
                assert key_ in hydra_recorders[key];
                "record attribute was not added!"
                assert pywr_recorders[key][key_] == hydra_recorders[key][key_]

    def test_general_paremeters(self):
        pass
    def test_nodes(self):
        pass

    def test_edges(self):
        pass

    def test_solver(self):
        pass



def main():
    unittest.main()

if __name__ == '__main__':
    main()


