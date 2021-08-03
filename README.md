Fully-Automated Sequence design plaTform (HIgh-Throughput)

# Installations
```console
git clone https://github.com/hury07/fasthit.git
```
if you need encoder submodule tape,
```console
cd fasthit/encoders/tape
git submodule init
git submodule update

```
use .yaml file to set conda environment configuration
```console
conda env create -f environments.yaml
```
Install in development mode
```console
pip install -e .
```
with extra dependencies
```console
pip install -e .[extras]
```