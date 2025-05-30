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


def parse_requirements(filename):
    """Parse a requirements file, handling reference directives."""
    requirements = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                # For reference to another requirements file
                ref_file = line[3:].strip()
                if os.path.exists(ref_file):
                    requirements.extend(parse_requirements(ref_file))
            else:
                requirements.append(line)
    return requirements


# Parse requirements directly from qward requirements file for more reliable results
REQUIREMENTS = parse_requirements("requirements.qward.txt")

if not hasattr(setuptools, "find_namespace_packages") or not inspect.ismethod(
    setuptools.find_namespace_packages
):
    print(
        "Your setuptools version:'{}' does not support PEP 420 (find_namespace_packages). "
        "Upgrade it to version >='40.1.0' and repeat install.".format(setuptools.__version__)
    )
    sys.exit(1)

VERSION_PATH = os.path.join(os.path.dirname(__file__), "qward", "VERSION.txt")
with open(VERSION_PATH, "r", encoding="utf-8") as version_file:
    VERSION = version_file.read().strip()

# Read long description from README.
README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md")
with open(README_PATH, "r", encoding="utf-8") as readme_file:
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
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum validation analysis metrics quality",
    packages=setuptools.find_packages(include=["qward", "qward.*"]),
    install_requires=REQUIREMENTS,
    include_package_data=True,
    python_requires=">=3.10",
    project_urls={
        "Bug Tracker": "https://github.com/xthecapx/qiskit-qward/issues",
        "Documentation": "https://xthecapx.github.io/qiskit-qward/",
        "Source Code": "https://github.com/xthecapx/qiskit-qward",
    },
    zip_safe=False,
)
