# fast-hit
**F**ully-**A**utomated **S**equence design pla**T**form (**HI**gh-**T**hroughput)

## Installation
1. Clone this repository and `cd` into it.
    ```console
    git clone https://github.com/hury07/fasthit.git
    cd fasthit
    ```
2. (Optional) Set conda environment use `environment.yaml` file provided
    ```console
    conda env create -f environment.yaml
    ```
3. Install fast-hit using `pip`.
    ```console
    pip install .
    ```
4. You can also install in development mode.
    ```console
    pip install -e .
    ```
5. (Optional) Get and install submodules if needed.
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
6. (Optional) If you want to install extra dependencies, `cd` into repository.
    ```console
    pip install .[extras]
    ```