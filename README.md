This repository contains code for loading the multi-object 3D datasets introduced for the [ObSuRF project](https://stelzner.github.io/obsurf/): CLEVR3D and Multishapenet.

## Download Links
The datasets will be released shortly.

## Usage
The file `loading/dataset.py` contains a PyTorch 1.7 dataset class which may be used to load the datasets. It utilizes the `imageio` package for reading images.

The notebook `loading/data_check.ipynb` loads the data using this class and illustrates it via `matploblib` and `plotly`.
