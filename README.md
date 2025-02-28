# Learning-ML
This repository is related to ML projects, whose purpose is to learn ML while realizing the concepts.

## Tricks

- Matplot Visualization in VS Code

    VS Code has an interactive window that supports plot visualization. Select your code and right-click, then choose "Run Selection/Line in Python Interactive Window".

## Always remember virtualenv 🏞️

Create a virtual environment using command line:

```bash
sudo apt install python3-venv
python3 -m venv ml_env
source ml_env/bin/activate
deactivate
```

Or, Create a virtual enviornment using VS Code 👨🏽‍💻:

1. Type `Ctrl+Shift+P`
2. Type `Python: Create Environment` and select the command.
3. The command will present a list of enviornment types, Venv or Conda. Select Venv for our case.

This will create a `.venv` folder in your project. This will have your installed packages. It is easy and more safe, if you ever wants to deleter your development environment in frustration 😤. 

Make sure you select the correct **python interpreter** 🤌 in your IDE. To change it in VS Code:
- Press `Ctrl+Shift+P`
- Type "Python: Select Interpreter" and select it from the list. 📃
- Go the your virtuan environment bin folder and select python file. e.g. 'ml_env/bin/python3.10'

To get all the dependencies for your project, generate a requirements.txt file using `pip freeze > requirements.txt`. You can later use this file to install all the dependencies of your project in another folder using `pip install -r requirements.txt`.

# Best practices:

- Keep separate virtual environments for tensorflow and torch.
