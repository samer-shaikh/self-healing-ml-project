# Project Setup Commands

This file contains all terminal commands used to set up the project.

## start with the cookiecutter 
- cookiecutter -c v1 https://github.com/drivendataorg/cookiecutter-data-science

## Create the environment using Conda

- Create a new virtual environment with Python 3.11
``` bash
conda create --name self_healing python=3.11
```

- Activate the environment
``` bash
conda activate self_healing
```
## Initialize Git

- Initialize repository
``` bash
git init
```

- Add all files
``` bash
git add .
```

- First commit
``` bash
git commit -m "Initial project setup"
```

- Connect to GitHub
``` bash
git remote add origin <your-repo-url>
```

- Push code
``` bash
git push -u origin main
```

## Initialize Dvc

- Initialize repository
``` bash
dvc init
```



## MLflow 
``` bash
pip install mlflow
```



