# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import setuptools
import inspect
import sys
import os
import re

# Define requirements directly
REQUIREMENTS = [
    "qiskit==1.4.1",
    "qiskit-aer==0.16.3",
    "qiskit-ibm-runtime==0.36.1",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "python-dotenv>=1.0.0",
    "pylatexenc",
    "ibm-cloud-sdk-core==3.23.0",
    "ibm-platform-services==0.59.1",
    "pydantic-settings",
    "requests==2.32.3",
    "dotenv",
    "statsmodels",
    "myst-parser==3.0.1",
]

if not hasattr(setuptools, "find_namespace_packages") or not inspect.ismethod(
    setuptools.find_namespace_packages
):
    print(
        "Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
        "Upgrade it to version >='40.1.0' and repeat install.".format(setuptools.__version__)
    )
    sys.exit(1)

# Get version info
VERSION = "0.1.0"  # Default version
VERSION_PATH = os.path.join(os.path.dirname(__file__), "qiskit_qward", "VERSION.txt")
if os.path.isfile(VERSION_PATH):
    with open(VERSION_PATH, "r") as version_file:
        VERSION = version_file.read().strip()

# Read long description from README.
README = ""
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
if os.path.isfile(README_PATH):
    with open(README_PATH) as readme_file:
        README = re.sub(
            "<!--- long-description-skip-begin -->.*<!--- long-description-skip-end -->",
            "",
            readme_file.read(),
            flags=re.S | re.M,
        )

setuptools.setup(
    name="qiskit-qward",
    version=VERSION,
    description="A framework for analyzing and validating quantum code execution quality on quantum processing units (QPUs)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/xthecapx/qiskit-qward",
    author="Thecap",
    author_email="xthecapx@gmail.com",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="qiskit quantum validation analysis metrics quality",
    packages=setuptools.find_packages(include=["qiskit_qward", "qiskit_qward.*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    package_data={"qiskit_qward": ["VERSION.txt"]},
    python_requires=">=3.8",
    project_urls={
        "Bug Tracker": "https://github.com/xthecapx/qiskit-qward/issues",
        "Documentation": "https://xthecapx.github.io/qiskit-qward/",
        "Source Code": "https://github.com/xthecapx/qiskit-qward",
    },
    zip_safe=False,
)
