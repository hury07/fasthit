# fast-hit
**F**ully-**A**utomated **S**equence design pla**T**form (**HI**gh-**T**hroughput)

## Installation
1. Clone this repository and `cd` into it.
    ```console
    git clone https://github.com/hury07/fasthit.git
    cd fasthit
    ```
2. (Recommend) Set conda environment use `environment.yaml` file provided
    ```console
    conda env create -f environment.yaml
    conda activate fast-hit
    ```
3. Install fast-hit using `pip`.
    ```console
    pip install .
    ```
4. You can also install in development mode.
    ```console
    pip install -e .
    ```
## Additional dependencies
### Landscapes
1. ViennaRNA
    ```console
    conda install -c bioconda viennarna
    ```
2. PyRosetta
    ```console
    conda install -c https://levinthal:paradox@conda.graylab.jhu.edu pyrosetta
    ```
### Pretrained protein sequence encoders
1. ESM
    ```console
    pip install .[esm]
    conda install -c conda-forge -c bioconda hhsuite
    ```
2. ProtTrans
    ```console
    pip install .[ProtTrans]
    ```
3. TAPE
    ```console
    cd fasthit/encoders/tape
    git submodule init
    git submodule update
    pip install .
    ```
    Or install submodule in development mode
    ```console
    pip install -e .
    ```