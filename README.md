# flexAE

Work in progress...

### How to install

You could just download the `flexAE.py` package and import it directly. Alternatively, I recommend [git cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this entire repository and [creating a separate conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for flexAE:

```
conda create -name venv_flexAE python=3.14
```

Then activate and navigate into the cloned repository, and run this command in the terminal to install the correct dependencies:

```
pip install -r requirements.txt
```

This code was last tested with python 3.14. The main dependencies are numpy, matplotlib, pandas, scikit-learn, seaborn, torch, tqdm. 