
## Setup and Execution
We will use PIPENV to manage environment

Certainly! Here's a user guide to help you get started with `pipenv` for managing dependencies in your Python project with multiple modules:

Step 1: Install pipenv
If you haven't already, you need to install `pipenv`. Open a command prompt or terminal and run the following command:

```shell
pip install pipenv
```

Step 2: Set Up a New Project
Navigate to the root directory of your project using the command prompt or terminal. Run the following command to initialize a new project with `pipenv`:

```shell
pipenv --python 3.8
```

Replace `3.8` with your desired Python version.

> Optional
> Step 3: Install Dependencies (optional)
> Create a `requirements.txt` file or use an existing one that lists your project's dependencies and their versions. 
> For example:
> 
> ```
> requests==2.25.1
> numpy==1.21.0
> ```
> 
> Run the following command to install the dependencies (optional):
> 
> ```shell
> pipenv install -r requirements.txt
> ```

This command creates a virtual environment, if one doesn't exist already, and installs the specified dependencies inside it.

Step 4: Activate the Virtual Environment
To activate the virtual environment, use the following command:

```shell
pipenv shell
```

This command will spawn a new shell with the virtual environment activated. From here, all subsequent commands will run within the virtual environment.

Step 5: Managing Dependencies
To install a new package and add it to your project's dependencies, use the following command:

```shell
pipenv install package_name
```

To uninstall a package and remove it from your project's dependencies, use:

```shell
pipenv uninstall package_name
```

You can also specify specific versions or ranges of versions when installing or uninstalling packages.

Step 6: Running Scripts or Modules
To run a script or module within the virtual environment, use the following command:

```shell
pipenv run python path/to/script.py
```

Replace `path/to/script.py` with the path to your script or module.

Step 7: Exiting the Virtual Environment
To exit the virtual environment and return to your system's default environment, use the following command:

```shell
exit
```

These steps should guide you in setting up and using `pipenv` to manage dependencies in your project with multiple modules. Remember to keep your `Pipfile` and `Pipfile.lock` files (created by `pipenv`) under version control to ensure consistent dependency management across different environments.
