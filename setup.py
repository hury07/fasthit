import os
import setuptools

### get dependence package list
_filedir = os.path.dirname(os.path.abspath(__file__))
requirementPath = _filedir + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
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
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=install_requires,
    include_package_data=True,
    package_data={
        "": [
            #"landscapes/data/rosetta/*",
            #"landscapes/data/tf_binding/*",
            "landscapes/data/gb1/*",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
