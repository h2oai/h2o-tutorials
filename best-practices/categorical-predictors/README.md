# Best Practices - Categorical Predictors

In this section, we will discuss best practices for dealing with categorical predictors in supervised learning problems.

## Contents

* [gbm_drf](gbm_drf.ipynb): describes the effects categorical columns have on algorithms like GBM and Random Forest.  It also walks through two methods for dealing with high cardinality categorical columns:
	* comparing model performance after removing high cardinality columns
	* parameter tuning (specifically tuning `nbins_cats` and `categorical_encoding`)
* [target_encoding](target_encoding.md): describes the process of target encoding with H2O-3 (this is another method for dealing with high cardinality categorical columns)