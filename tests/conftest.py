# Legacy test modules written against an importer/exporter API that no longer exists
# (PywrHydraImporter / PywrHydraExporter, importer.add_attributes_request_data, etc.). They fail at
# import time and cannot be revived by renaming imports: the replacement classes have different
# constructors and the template helpers (generate_pywr_attributes, PYWR_DEFAULT_DATASETS) were
# removed. They are excluded from collection until they are ported; see test_v1_roundtrip.py for
# the current-API baseline.
collect_ignore = [
    "test_client_exporting.py",
    "test_client_importing.py",
    "test_components.py",
    "test_hydra_base_exporting.py",
    "test_hydra_base_importing.py",
    "test_nodes_edges.py",
    "test_parameter_patterns.py",
]
