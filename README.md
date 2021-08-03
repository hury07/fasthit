> FAST-HIT: **F**ully-**A**utomated **S**equence design pla**T**form (**HI**gh-**T**hroughput)

# Installations
## Downloads
1. Download fasthit
    ```console
    git clone https://github.com/hury07/fasthit.git
    ```
2. Get submodules
    If you need submodule tape for encoders,
    ```console
    cd fasthit/encoders/tape
    git submodule init
    git submodule update
    ```
## Install
1. Install fasthit
    ```console
    cd fasthit
    pip install .
    ```
2. If you want to install extra dependencies,
    ```console
    pip install .[extras]
    ```
3. You can also install in development mode by
    ```console
    pip install -e .[extras]
    ```
4. Optionally you can set conda environment configuration use environment.yaml file provided
    ```console
    conda env create -f environment.yaml
    ```