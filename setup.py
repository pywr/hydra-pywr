#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='hydra-pywr',
    version='0.1',
    description='Hydra plugins and library for import and exporting Pywr models.',
    packages=find_packages(),
    package_data={
        # Include all template config files
        'hydra_pywr': ['template_configs/*.json'],
    },
    include_package_data=True,
    entry_points='''
    [console_scripts]
    hydra-pywr=hydra_pywr.cli:start_cli
    ''',
)
