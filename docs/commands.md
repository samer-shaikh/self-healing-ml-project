# Project Setup Commands

This file contains all terminal commands used to set up the project.

## start with the cookiecutter 
- cookiecutter -c v1 https://github.com/drivendataorg/cookiecutter-data-science

## Create the environment using Conda

- Create a new virtual environment with Python 3.11
conda create --name self_healing python=3.11

- Activate the environment
conda activate self_healing

## Initialize Git

- Initialize repository
git init

- Add all files
git add .

- First commit
git commit -m "Initial project setup"

- (Optional) Connect to GitHub
git remote add origin <your-repo-url>

- Push code
git push -u origin main