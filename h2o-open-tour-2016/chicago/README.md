# H2O Open Chicago Training 

The training is broken up into two modules, an introductory machine learning module and a grid search / model selection module.  The slides that accompany this tutorial are available [here](https://github.com/h2oai/h2o-tutorials/raw/master/h2o-open-tour-2016/chicago/h2o-open-chicago-2016-training.pdf).

### R Users

R users can use RStudio or the R console to execute the R scripts.

- Introductory Module: [intro-to-h2o.R](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/intro-to-h2o.R)
- Grid Search Module: [grid-search-model-selection.R](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/grid-search-model-selection.R)

### Python Users

Python users can use a [Jupyter/IPython notebook](http://jupyter.org/) to execute the scripts.  To install Jupyter, we recommend doing the following:

```bash
pip install -U jupyter
```

In this directory, execute the following to start the Jupyter notebook server:

```bash
jupyter notebook
```
To execute the cells in the notebook simply highlight the cell and press Shift + Enter.

- Introductory Module: [intro-to-h2o.ipynb](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/intro-to-h2o.ipynb)
- Grid Search Module: [grid-search-model-selection.ipynb](https://github.com/h2oai/h2o-tutorials/blob/master/h2o-open-tour-2016/chicago/grid-search-model-selection.ipynb)

### Take Home Tutorial (Extra Credit)

To learn how to create multi-algorithm ensembles using the [h2oEnsemble](https://github.com/h2oai/h2o-3/tree/master/h2o-r/ensemble) R package, visit the  ensemble [demos](https://github.com/h2oai/h2o-3/tree/master/h2o-r/ensemble/demos) folder and check out some of the demos.

Suggested:

- Create an ensemble from scratch: [h2o\_ensemble\_documentation\_example.R](https://github.com/h2oai/h2o-3/blob/master/h2o-r/ensemble/demos/h2o_ensemble_documentation_example.R)
- Stacking existing H2O models: [h2o\_stack\_documentation\_example.R](https://github.com/h2oai/h2o-3/blob/master/h2o-r/ensemble/demos/h2o_stack_documentation_example.R)
- Stacking with Random Grids: [higgs\_randomgrid\_stack.R](https://github.com/h2oai/h2o-3/blob/master/h2o-r/ensemble/demos/higgs_randomgrid_stack.R)