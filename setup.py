"""Setuptools configuration supplement.

Extends the declarative config in pyproject.toml with dynamic package
discovery so that the ``web/`` source directory is bundled as
``sirchmunk._web`` inside the installed package.  This allows
``sirchmunk web init`` to locate the web source for non-editable
(pip install) installations.
"""

from setuptools import find_packages, setup

packages = find_packages(where="src", include=["sirchmunk*"])
packages.append("sirchmunk._web")

setup(
    packages=packages,
    package_dir={
        "": "src",
        "sirchmunk._web": "web",
    },
)
