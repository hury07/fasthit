import os
from setuptools import setup, find_packages

### get dependence package list
_filedir = os.path.dirname(os.path.abspath(__file__))
requirementPath = _filedir + "/requirements.txt"
requirments = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        requirments = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="fast-hit",
    version="0.0.0",
    description=(
        "fast-hit: a fully-automated, algorithm-driven biological sequence design platform"
        "enabled by automated, high-throughput experiment infrastructure."
    ),
    url="https://github.com/hury07/fast-hit",
    author="Ruyun Hu",
    author_email="hury07@hotmail.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=requirments,
    extras_require={
        "fast": [
            "numba",
            "memcache",
        ]
    },
    include_package_data=True,
    package_data={
        "landscapes": [
            "landscapes/data/rosetta/*",
            "landscapes/data/tf_binding/*",
            "landscapes/data/gb1/*",
        ]
    },
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Synthetic Biology",
    ],
)
