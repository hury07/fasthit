# fast-hit
**F**ully-**A**utomated **S**equence design pla**T**form (**HI**gh-**T**hroughput)

## Requirements
- Python>=3.7
- pytorch==1.8.0
- gpytorch==1.4.2
## Installation
1. Clone this repository and `cd` into it.
    ```console
    git clone https://github.com/hury07/fasthit.git
    cd fasthit
    ```
2. (Recommend) Set conda environment use file `environment.yaml`.
    ```console
    conda env create -f environment.yaml
    conda activate fast-hit
    ```
3. Install fast-hit using `pip`.
    ```console
    pip install .
    ```
    Or install in developer mode.
    ```console
    pip install -e .
    ```
## Install additional dependencies
### Landscapes
1. ViennaRNA
    ```console
    conda install -c bioconda viennarna
    ```
2. PyRosetta
    ```console
    conda install -c https://levinthal:paradox@conda.graylab.jhu.edu pyrosetta
    ```
### Encoders
#### Pretrained protein sequence encoders
1. ESM
    ```console
    pip install fair-esm==0.4.0
    conda install -c conda-forge -c bioconda hhsuite
    ```
    Extra hhsuite databases are required for MSA-Transformer
2. ProtTrans
    ```console
    pip install transformers
    ```
3. TAPE
    ```console
    cd fasthit/encoders/tape
    git submodule init
    git submodule update
    pip install .
    ```
    Extra hhsuite databases are required for Rosetta
### Explorers
1. DI-engine (reinforcement learning engine)
    ```console
    pip install DI-engine==0.1.1
    ```