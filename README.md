# Learning-ML
This repository is related to ML projects, whose purpose is to learn ML while realizing the concepts.

## Managing dependencies across environment 🏞️

First create a virtual enviornment for you project:
1. Type `Ctrl+Shift+P`
2. Type `Python: Create Environment` and select the command.
3. The command will present a list of enviornment types, Venv or Conda. Select Venv for our case.

This will create a `.venv` folder in your project. This will have your installed packages.

To activate enviornment: `source .venv/bin/activate` and to deactivate enviornment: `deactivate`

To get all the dependencies for your project, generate a requirements.txt file using `pip freeze > requirements.txt`. You can later use this file to install all the dependencies of your project in another folder using `pip install -r requirements.txt`.

// check
// update git blog 
// about how to remove file from staging area