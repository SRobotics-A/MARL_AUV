"""Installation script for the 'MARL_mav_carry_ext' python package.

Isaac Sim 5.x ships with Python 3.11, so this setup script:
- Uses the stdlib `tomllib` to read extension.toml (no external `toml` dependency).
- Updates python_requires / classifiers to reflect Isaac Sim 5.1.0 / Python 3.11.
"""

import os
import tomllib

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))

# Read the extension.toml file (stdlib in Python 3.11)
with open(os.path.join(EXTENSION_PATH, "config", "extension.toml"), "rb") as f:
    EXTENSION_TOML_DATA = tomllib.load(f)

# Minimum runtime dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add runtime dependencies for your extension here.
    "psutil",
]

# Installation operation
setup(
    name="MARL_mav_carry_ext",
    packages=["MARL_mav_carry_ext"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    # Isaac Sim 5.x uses Python 3.11
    python_requires=">=3.11,<3.12",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
