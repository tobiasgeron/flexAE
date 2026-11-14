# flexAE

[![CI](https://github.com/tobiasgeron/flexAE/actions/workflows/main.yml/badge.svg)](https://github.com/tobiasgeron/flexAE/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/tobiasgeron/flexAE/graph/badge.svg)](https://codecov.io/gh/tobiasgeron/flexAE)

flexAE is an autoencoder framework that seamlessly integrates multiple components. These include a classifier component that acts directly in the latent space, a separate input head for observational uncertainties, a scatter component, and a toggle to make it a variational autoencoder. The most common use case for flexAE is classification in a high-dimensional space while accounting for observational uncertainties. It was originally developed in Géron et al. (2026) to classify unresolved binary systems in [47 Tucanae](https://en.wikipedia.org/wiki/47_Tucanae) using photometric data from Rubin DP1, but it can be applied to many other classification problems.

If you find this repository useful, I would love to hear about how you use flexAE, so please let me know at tobias.geron@utoronto.ca. Also feel free to reach out if you have any questions. If you end up using this for a publication, please cite Géron et al. (2026). 


### Minimal example

flexAE can be setup and trained fairly quickly. In the example below, I will assume that we have prepared the training dataset as a PyTorch `TensorDataset`.

```
from flexAE import *

# Initialize model with architecture
model = flexAE(input_dim = 4)

# Train for 100 epochs
model.train_model(train_dataset, epochs = 100)
```

Do check out the tutorials for more elaborate examples! There are many features you can customize. And you can always run `help(flexAE)` to read the docstring. 



### How to install

The simplest way to use this code is to just download the `flexAE.py` file, copy-paste in your project and import it directly. However, this can be prone to dependency conflicts. 

Alternatively, I recommend [git cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this entire repository and navigating into it by running these commands in the terminal:

```
git clone https://github.com/tobiasgeron/flexAE.git
cd flexAE
```

and [creating a separate conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for flexAE and activating it:

```
conda create --name venv_flexAE python=3.14
conda activate venv_flexAE
```

Then run this command to install the correct dependencies:

```
pip install -r requirements.txt
```

This code was last tested with python 3.14. The main dependencies are numpy, matplotlib, pandas, scikit-learn, seaborn, torch, tqdm. 
