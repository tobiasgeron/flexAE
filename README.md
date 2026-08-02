# flexAE

[![CI](https://github.com/tobiasgeron/flexAE/actions/workflows/main.yml/badge.svg)](https://github.com/tobiasgeron/flexAE/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/tobiasgeron/flexAE/graph/badge.svg)](https://codecov.io/gh/tobiasgeron/flexAE)

Work in progress...

### How to install

The simplest way to use this code is to just download the `flexAE.py` file, copy-paste in your project and import it directly. However, this can be prone to dependency conflicts. 

Alternatively, I recommend [git cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this entire repository and navigating into it by running these commands in the terminal:

```
git clone https://github.com/tobiasgeron/flexAE.git
cd flexAE
```

and [creating a separate conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for flexAE and activating it:

```
conda create -name venv_flexAE python=3.14
conda activate venv_flexAE
```

Then run this command to install the correct dependencies:

```
pip install -r requirements.txt
```

This code was last tested with python 3.14. The main dependencies are numpy, matplotlib, pandas, scikit-learn, seaborn, torch, tqdm. 
