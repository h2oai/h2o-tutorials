# H2O Tutorials

This document contains tutorials and training materials for H2O-3.  Post questions on StackOverflow using the h2o tag at [http://stackoverflow.com/questions/tagged/h2o](http://stackoverflow.com/questions/tagged/h2o) or join the "H2O Stream" Google Group:

* Web: <https://groups.google.com/forum/#!forum/h2ostream>
* E-mail: <mailto:h2ostream@googlegroups.com>

## Finding tutorial material in Github

### Most current material

Tutorials in the master branch are intended to work with the lastest stable version of H2O.

| | URL |
| --- | --- |
| Training material | <https://github.com/h2oai/h2o-tutorials/blob/master/SUMMARY.md> |
| Latest stable H2O release | <http://h2o.ai/download> |

### Historical events

Tutorial versions in named branches are snapshotted for specific events.  Scripts should work unchanged for the version of H2O used at that time.

#### H2O World 2015 Training

| | URL |
| --- | --- |
| Training material | <https://github.com/h2oai/h2o-tutorials/blob/h2o-world-2015-training/SUMMARY.md> |
| Tibshirani-3 H2O release | <http://h2o-release.s3.amazonaws.com/h2o/rel-tibshirani/3/index.html> |


### R Tutorials

- [Intro to H2O in R](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/intro-to-h2o.R)
- [H2O Grid Search & Model Selection](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/grid-search-model-selection.R)
- [H2O Deep Learning in R](http://htmlpreview.github.io/?https://github.com/ledell/sldm4-h2o/blob/master/sldm4-deeplearning-h2o.html)
- [H2O Stacked Ensembles](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)
- [h2oEnsemble R package](http://learn.h2o.ai/content/tutorials/ensembles-stacking/index.html)


### Python Tutorials

- [Intro to H2O in Python](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/intro-to-h2o.ipynb)
- [H2O Grid Search & Model Selection](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/grid-search-model-selection.ipynb)
- [H2O Stacked Ensembles](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html)

For most tutorials using python you can install dependent modules to your environment by running the following commands.

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
