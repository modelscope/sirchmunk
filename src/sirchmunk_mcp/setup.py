# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Setup script for Sirchmunk MCP Server.

For backwards compatibility with older pip versions.
Modern installations should use pyproject.toml.
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(),
        include_package_data=True,
    )
