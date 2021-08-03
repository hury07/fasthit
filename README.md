# fast-hit
**F**ully-**A**utomated **S**equence design pla**T**form (**HI**gh-**T**hroughput)

## Installation
1. Clone this repository and `cd` into it.
    ```console
    git clone https://github.com/hury07/fasthit.git
    cd fasthit
    ```
2. Set conda environment configuration use `environment.yaml` file provided
    ```console
    conda env create -f environment.yaml
    ```
3. Install fasthit using `pip`.
    ```console
    pip install .
    ```
4. Get and install submodules if needed.
    ```console
    cd fasthit/encoders/tape
    git submodule init
    git submodule update
    pip install .
    ```
5. If you want to install extra dependencies, `cd` into repository.
    ```console
    pip install .[extras]
    ```
6. You can also install in development mode by
    ```console
    pip install -e .[extras]
    ```