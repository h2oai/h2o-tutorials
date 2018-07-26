
# GLRM Best Practices

This folder contains a best practices tutorial that goes through common uses of GLRM and how to tune parameters. 

The data used for this tutorial is computer generated employee attributes.   

## What is a Low Rank Model?
* **Given:** Data table *A* with m rows and n columns
* **Find:** Compress representation as numeric tables *X* and *Y*, where # cols in X = # rows in Y = small user-specified k << max(m, n)
* \# cols in *Y* is d = (total dimension of embedded features in *A*) >= n

In the example below, we are seeing the decomposition of A when A consists of only numeric columns.

![](GLRM.png)

* *Y* = archetypal features created from columns of *A*
* *X* = row of *A* in reduced feature space
* GLRM can approximately reconstruct *A* from product *XY*


## GLRM Models Uses

GLRM models have multiple use cases: 

* filling in missing entries
* reduce storage
* remove noise
* understand (visualize, cluster)


## General Information

### Regularizations
structure | regularization_x | regularization_y
----------|------------------|-----------------
small | Quadratic | Quadratic
sparse | L1 | L1
non negative | Non Negative | Non Negative
clustered | Sparse | None

## Loss
data type | loss | L(u, a)
----------|------|--------
real | quadratic | $(u-a)^2$
real | absolute | $|u - a|$
real | huber | $huber(u - a)$
boolean | hinge | $(1 - ua)_+$
boolean | logistic | $log(1 + exp(-au))$
integer | poisson | $exp(u) - au + alog(a) - a$
ordinal | ordinal hinge | $\sum_{a^\prime = 1}^{a-1} (1 - u + a^\prime)_+ + \sum_{a^\prime = a + 1}^d (1 + u - a^\prime)_+$
categorical | one-vs-all | $(1-u_a)_+ + \sum_{a^\prime \neq a} (1 + u_{a^\prime})_+$

## Initialization

The `init` parameter specifies how the X and Y matrices are initially generated.

* **Random:** random arrays from normal distribution
* **PlusPlus:** initialization using teh clusters from k-means++ initialization.
* **SVD:** initialization using the first k right singular values.  Helps with global convergence for matrix factorizations where global convergence not guaranteed (ex Non Negative Matrix Factorization)
