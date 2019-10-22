# H2O Tutorials

This document contains tutorials and training materials for H2O-3.  If you find any problems with the tutorial code, please open an issue in this repository.

For general H2O questions, please post those to [Stack Overflow using the "h2o" tag](http://stackoverflow.com/questions/tagged/h2o) or join the [H2O Stream Google Group](https://groups.google.com/forum/#!forum/h2ostream) for questions that don't fit into the Stack Overflow format.

## Finding tutorial material in Github

There are a number of tutorials on all sorts of topics in this repo.  To help you get started, here are some of the most useful topics in both R and Python.


### R Tutorials

- [Intro to H2O in R](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/intro-to-h2o.R)
- [H2O Grid Search & Model Selection in R](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/grid-search-model-selection.R)
- [H2O Deep Learning in R](http://htmlpreview.github.io/?https://github.com/ledell/sldm4-h2o/blob/master/sldm4-deeplearning-h2o.html)
- [H2O Stacked Ensembles in R](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)
- [H2O AutoML in R](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-world-2017/automl/README.md)
- [LatinR 2019 H2O Tutorial](https://github.com/ledell/LatinR-2019-h2o-tutorial) (broad overview of all the above topics)


### Python Tutorials

- [Intro to H2O in Python](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/intro-to-h2o.ipynb)
- [H2O Grid Search & Model Selection in Python](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/grid-search-model-selection.ipynb)
- [H2O Stacked Ensembles in Python](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)
- [H2O AutoML in Python](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-world-2017/automl/README.md)


### Most current material

Tutorials in the master branch are intended to work with the lastest stable version of H2O.

| | URL |
| --- | --- |
| Training material | <https://github.com/h2oai/h2o-tutorials/blob/master/SUMMARY.md> |
| Latest stable H2O release | <http://h2o.ai/download> |



### Historical events

Tutorial versions in named branches are snapshotted for specific events.  Scripts should work unchanged for the version of H2O used at that time.

#### H2O World 2017 Training

| | URL |
| --- | --- |
| Training material | <https://github.com/h2oai/h2o-tutorials/tree/master/h2o-world-2017/README.md> |
| Wheeler-2 H2O release | <http://h2o-release.s3.amazonaws.com/h2o/rel-wheeler/2/index.html> |

#### H2O World 2015 Training

| | URL |
| --- | --- |
| Training material | <https://github.com/h2oai/h2o-tutorials/blob/h2o-world-2015-training/SUMMARY.md> |
| Tibshirani-3 H2O release | <http://h2o-release.s3.amazonaws.com/h2o/rel-tibshirani/3/index.html> |


### Requirements:

For most tutorials using Python you can install dependent modules to your environment by running the following commands.

```
# As current user
pip install -r requirements.txt
```

```
# As root user
sudo -E pip install -r requirements.txt
```

**Note:** If you are behind a corporate proxy you may need to set environment variables for `https_proxy` accordingly.

```
# If you are behind a corporate proxy
export https_proxy=https://<user>:<password>@<proxy_server>:<proxy_port>

# As current user
pip install -r requirements.txt
```

```
# If you are behind a corporate proxy
export https_proxy=https://<user>:<password>@<proxy_server>:<proxy_port>

# As root user
sudo -E pip install -r requirements.txt
```
