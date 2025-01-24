# Utilizing Synthetic Point Cloud Generation for Semantic Segmentation of Utility Poles

A workflow for generating synthetic datasets for use in semantically segmenting utility poles.

<!-- TODO Citation -->

## Setup

This tool was written and tested using _Python 3.10_ and _Python 3.12_.

Create a python environment and install all dependencies listed in the requirements.txt. You install these dependencies via `pip install -U -r requirements.txt`.

Look into config.yaml for any settings you may want to change before using this project. You may additionally look into utils/config.py to see each setting's default value as well as other settings you may want to use.

## Usage

- To quickly see the utility pole generator in action, take a look at **demo_pole_gen.ipynb**.
- To perform a full training and testing process using a PointNet model, run **train.py**. See utils/config.py and the config.yaml for settings to use.
- See **demo_segmentation.ipynb** For an interactive demonstration of your PointNet model trained with **train.py**.

## References

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://doi.org/10.48550/arXiv.1612.00593)

- [3DP-Point-Cloud-Segmentation](https://github.com/sepideh-shamsizadeh/3DP-Point-Cloud-Segmentation)
