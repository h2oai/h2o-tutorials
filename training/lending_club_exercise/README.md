# Hands On H2O Session

## Introduction

This session will cover a hands on training of H2O in Flow, Python, and R.  We will go through a complete machine learning workflow in R and Python.  Attendees do not need to follow the workshop for both R and Python, they can choose which language they are most comfortable with and stick with that.  We will also show Flow, H2O's web-based UI and how it can be used in conjunction with R and Python.

## Requirements for the Hands On Training

* Make sure you have either Python or R installed.
* Make sure you have H2O 3.10.4.2 installed for Python or R (depending on your preferred language).  Instructions on installing H2O can be found below.


## H2O Installation Instructions

Installation in R and Python requires packages to be installed.  For more details about the installation see the [H2O downloads page](http://h2o-release.s3.amazonaws.com/h2o/rel-ueno/2/index.html).

### Python Users

Prerequisites for using H2O and Python is: 

* JDK version 6+
* Python 2.7 or 3.3 installed

#### Installation Instructions:

1. In terminal, install dependencies

  ```bash
  pip install requests
  pip install tabulate
  pip install scikit-learn
  pip install matplotlib

  # If you wish to do the demo with an ipython notebook, make sure ipython is installed
  pip install ipython
  ```
  **Troubleshooting Tip**: If you see that you are missing dependencies for ipython, try the following: 
  ```
  pip install --upgrade setuptools pip
  pip uninstall ipython
  pip install ipython
  ```

2. Uninstall h2o module for Python (if already installed)

  ```bash
  pip uninstall h2o
  ```

3. Install latest version of h2o Python module (version 3.10.4.2)

  ```bash
  pip install http://h2o-release.s3.amazonaws.com/h2o/rel-ueno/2/Python/h2o-3.10.4.2-py2.py3-none-any.whl
  ```
  **Troubleshooting Tip**: If this installation times out, try the following:
  ```bash
  pip install https://h2o-release.s3.amazonaws.com/h2o/rel-ueno/2/Python/h2o-3.10.4.2-py2.py3-none-any.whl
  ```

4.  Check H2O was properly installed.  In a new terminal, type `python` to open a python shell.

  ```python
  # Import H2O pacakge
  import h2o
  # Initialize h2o cluster
  h2o.init()
  ```

5. If installation was successful, you can stop the h2o cluster with the following command in python:
  
  ```python
  h2o.shutdown()
  ```

6. If you want to use an IPython  notebook for the training, please check to make sure you are able to open an ipython notebook.  In terminal, run the following command: 
 ```bash
 ipython notebook
 ```

### R Users


Prerequisites for using H2O and Python is: 

* JDK version 6+
* R installed

#### Installation Instructions:


1. From R, uninstall h2o package for R (if already installed)

  ```
  if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
  if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
  ```
2. Install dependencies in R.

  ```
  if (! ("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
  if (! ("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
  if (! ("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
  if (! ("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
  if (! ("jsonlite" %in% rownames(installed.packages()))) { install.packages("jsonlite") }
  if (! ("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
  if (! ("utils" %in% rownames(installed.packages()))) { install.packages("utils") }
  ```

3. Install the H2O Package (version 3.10.4.2).
  
  ```
  install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/rel-ueno/2/R")))
  ```
4.  Check H2O was properly installed.

  ```
  # Load H2O library
  library(h2o)
  # Initialize h2o cluster
  h2o.init()
  ```

5. If installation was successful, you can stop the h2o cluster with the following command in R:
  
  ```
  h2o.shutdown()
  ```


