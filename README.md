# Utilizing Synthetic Point Cloud Generation for Semantic Segmentation of Utility Poles

A workflow for generating synthetic datasets for use in semantically segmenting utility poles.

## Setup

This tool was written and tested using _Python 3.10_ and _Python 3.12_.

Create a python environment and install all dependencies listed in the requirements.txt. You can do this with one command via `pip install -U -r requirements.txt`.

Create a .env file in this project's root directory containing an entry SEGMENTS_AI_API_KEY={Your Segments.ai API key}. This is needed to fetch and preprocess the ground truth data.

Look into config.yaml for any settings you may want to change before using this project.

## Usage

- To quickly see the utility pole generator in action, take a look at _pole_gen_demo.ipynb_.
- To perform a full training and testing process using a PointNet model, run _train.py_. Take a look at utils/config.py and the config.yaml.
- To perform the same as above in a Jupyter notebook environment, you may look at _notebook.ipynb_, however take note that multiprocessing was not implemented there and training your AI model might be slower!

## References

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://doi.org/10.48550/arXiv.1612.00593)

- [3DP-Point-Cloud-Segmentation](https://github.com/sepideh-shamsizadeh/3DP-Point-Cloud-Segmentation/tree/main)
