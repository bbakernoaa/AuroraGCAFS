"""
Legacy setup.py for compatibility with older tools.
Modern installation should use pyproject.toml with pip install -e .
"""
from setuptools import setup

# This is a simple forwarding setup.py for backwards compatibility.
# Actual configuration is in pyproject.toml
setup(
    # Package metadata and configuration is now in pyproject.toml
    # This file remains for compatibility with tools that don't yet
    # support pyproject.toml
)
