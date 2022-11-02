# fast-hit
**F**ully-**A**utomated **S**equence design pla**T**form (**HI**gh-**T**hroughput)
## Examples
Demos. are given in folder `examples`.
## Requirements
- Python=3.7
- pytorch=1.9.0
- gpytorch=1.5.0
## Installation
1. Clone this repository and `cd` into it.
    ```console
    git clone https://github.com/hury07/fasthit.git
    cd fasthit
    ```
2. (Recommend) Set conda environment use file `environment.yaml`.
    ```console
    conda env create -f environment.yaml
    conda activate fasthit
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
    pip install sentencepiece
    ```
3. TAPE
    ```console
    cd fasthit/encoders/tape
    git submodule init
    git submodule update
    pip install .
    ```
    Extra hhsuite databases are required for Rosetta
