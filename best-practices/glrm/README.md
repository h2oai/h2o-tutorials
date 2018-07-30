GLRM Model
----------

What is a Low Rank Model?
-------------------------

-   **Given:** Data table *A* with m rows and n columns
-   **Find:** Compress representation as numeric tables *X* and *Y*,
    where \# cols in X = \# rows in Y = small user-specified k &lt;&lt;
    max(m, n)
-   \# cols in *Y* is d = (total dimension of embedded features in
    *A*) &gt;= n

In the example below, we are seeing the decomposition of A when A
consists of only numeric columns.

![](./images/GLRM.png)

-   *Y* = archetypal features created from columns of *A*
-   *X* = row of *A* in reduced feature space
-   GLRM can approximately reconstruct *A* from product *XY*

GLRM Models Uses
----------------

GLRM models have multiple use cases:

-   filling in missing entries
-   reduce storage
-   remove noise
-   understand (visualize, cluster)

Understanding the Data
----------------------

We can use GLRM to understand the data. GLRM generates principal
components to interpret the data. Each row of Y is an archetypal feature
formed from the columns of A, and each row of X corresponds to a row of
A projected into this reduced dimension feature space.

For this tutorial, we will use data that contains attributes about a
house and its price in King County. The data is a public dataset from
Kaggle: <https://www.kaggle.com/harlfoxem/housesalesprediction>. We can
use GLRM to reduce the dimensions of the data 2 columns. We can use GLRM to reduce the dimensions of the data to 2 columns in order to visualize the houses data in a 2 dimensional plot.

We will begin by initializing the h2o cluster and importing the csv
file.

``` r
# Initate H2O Cluster
library('h2o')
h2o.init()
h2o.no_progress()

# Import Data
houses_data <- h2o.importFile("/Users/megankurka/Downloads/kc_house_data.csv")
```

    ##           id            date   price bedrooms bathrooms sqft_living
    ## 1 7129300520 20141013T000000  221900        3      1.00        1180
    ## 2 6414100192 20141209T000000  538000        3      2.25        2570
    ## 3 5631500400 20150225T000000  180000        2      1.00         770
    ## 4 2487200875 20141209T000000  604000        4      3.00        1960
    ## 5 1954400510 20150218T000000  510000        3      2.00        1680
    ## 6 7237550310 20140512T000000 1225000        4      4.50        5420
    ##   sqft_lot floors waterfront view condition grade sqft_above sqft_basement
    ## 1     5650      1          0    0         3     7       1180             0
    ## 2     7242      2          0    0         3     7       2170           400
    ## 3    10000      1          0    0         3     6        770             0
    ## 4     5000      1          0    0         5     7       1050           910
    ## 5     8080      1          0    0         3     8       1680             0
    ## 6   101930      1          0    0         3    11       3890          1530
    ##   yr_built yr_renovated zipcode     lat     long sqft_living15 sqft_lot15
    ## 1     1955            0   98178 47.5112 -122.257          1340       5650
    ## 2     1951         1991   98125 47.7210 -122.319          1690       7639
    ## 3     1933            0   98028 47.7379 -122.233          2720       8062
    ## 4     1965            0   98136 47.5208 -122.393          1360       5000
    ## 5     1987            0   98074 47.6168 -122.045          1800       7503
    ## 6     2001            0   98053 47.6561 -122.005          4760     101930

There are 3 columns that are being treated as integer that should be
converted to categorical or enum. The columns `condition`, `grade`, and
`view` are really ordinal columns so we will convert it to categorical.
Later on we can use ordinal loss for these functions. The column
`zipcode` is also being treated as an integer when it is really a
categorical column.

``` r
cat_cols <- c('condition', 'grade', 'zipcode', 'view', 'waterfront')
for(i in cat_cols){
  houses_data[[i]] <- as.factor(houses_data[[i]])
}
```

When we define our first GLRM, we need to know which columns we want to
include in the model. In this case, we will not use `zipcode`, `id`, or
`date` since they have many unique values. We will also want to select
the loss function for GLRM. The GLRM model is trying to minimize this
loss function. It is tightly coupled to:

-   the use case
    -   some loss functions are sensitive to outliers and some are
        robust to outliers
-   the data type
    -   some loss functions are specific to the type of data in the
        column

For this example, we will try Absolute loss for the numeric features.
This loss will be robust to outliers. For the columns, `condition`,
`grade`, and `view`, we will use Ordinal loss.

``` r
glrm_cols <- colnames(houses_data)[!(colnames(houses_data) %in% c("id", "date", "zipcode"))]

losses <- data.frame('index' = which(colnames(houses_data) %in% c("condition", "grade", "view")) - 1,
                     'loss' = rep("Ordinal", 3), stringsAsFactors = FALSE)

rank2_model <- h2o.glrm(training_frame = houses_data, cols = glrm_cols, loss = "Absolute",
                        model_id = "rank2", seed = 1234, k = 2, transform = "STANDARDIZE",
                        loss_by_col_idx = losses$index, loss_by_col = losses$loss)
```

``` r
# Decompose training frame into XY

X <- h2o.getFrame(rank2_model@model$representation_name)
Y <- rank2_model@model$archetypes
```

In this case, the rows of Y represent some house attribute profiles. The
rows of X can be used to cluster houses.

Let’s visualize the first two archetypes of Y.

``` r
# Visualize first two archetypes of Y
archetypes_y <- as.data.frame(t(Y))
archetypes_y$attribute <- rownames(archetypes_y)
```

Plot the archetypes:

``` r
library('plotly')
m <- archetypes_y[archetypes_y$attribute %in% c("sqft_lot", "sqft_lot15", "floors", "bedrooms", "price", "waterfront0", "waterfront1"), ]

a <- list(
  x = m$Arch1,
  y = m$Arch2,
  text = m$attribute,
  xref = "x",
  yref = "y"
)
plot_ly(data = archetypes_y, x = ~Arch1, y = ~Arch2, mode = "markers", text = ~attribute) %>%
  layout(annotations = a)
```

![archetypes\_y](./images/archetypes_y.png)

Attributes that are close together are considered similar. For example,
`sqft_lot` and `sqft_lot15` are very close together. This makes sense
since `sqft_lot` is the sqft of the lot where as `sqft_lot15` is the
sqft of the lot in 2015. Additionally `floors`, `bedrooms`, and `price`
are very close together. The values `waterfront0` and `waterfront1` are
far apart.

Now let’s visualize the first two archetypes of X.

``` r
# Visualize first two archetypes of X
archetypes_x <- as.data.frame(X)
archetypes_x$id <- as.character(as.matrix(houses_data$id))
set.seed(1234)
sample_indices <- sample(c(1:nrow(archetypes_x)), 100)
archetypes_x <- archetypes_x[sample_indices, ]

# Plot
plot_ly(data = archetypes_x, x = ~Arch1, y = ~Arch2, mode = "markers", 
        text = ~paste0("House: ", id))
```

![archetypes\_x](./images/archetypes_x.png)

Houses that are close together are considered similar. If we examine the
two houses next to each other at the bottom of the graph, we can see a
lot of similarities. The two houses are highlighted in red.

``` r
house_ids <-  c(5379802181, 3046200125)
archetypes_x$color <- ifelse(archetypes_x$id %in% house_ids, "A", "B")

plot_ly(data = archetypes_x, x = ~Arch1, y = ~Arch2, mode = "markers", 
        text = ~paste0("House: ", id), marker = list(
    color = factor(archetypes_x$color,labels=c("#d62728","#1f77b4"))  
  ) ) %>%
  layout(showlegend = FALSE)
```

![archetypes\_x\_highlighted](./images/archetypes_x_highlighted.png)

The houses are shown below:

|          id| date            |   price|  bedrooms|  bathrooms|  sqft\_living|  sqft\_lot|  floors| waterfront | view | condition | grade |  sqft\_above|  sqft\_basement|  yr\_built|  yr\_renovated| zipcode |      lat|      long|  sqft\_living15|  sqft\_lot15|
|-----------:|:----------------|-------:|---------:|----------:|-------------:|----------:|-------:|:-----------|:-----|:----------|:------|------------:|---------------:|----------:|--------------:|:--------|--------:|---------:|---------------:|------------:|
|  5379802181| 20141119T000000 |  193000|         2|          1|           680|       8640|       1| 0          | 0    | 4         | 5     |          680|               0|       1951|              0| 98188   |  47.4559|  -122.289|            1320|        13140|
|  3046200125| 20150406T000000 |  202000|         2|          1|           740|       6550|       1| 0          | 0    | 4         | 5     |          740|               0|       1946|              0| 98168   |  47.4807|  -122.332|            1080|         8515|

We can see that they have very similar sizes and price and were also
built during the same 5 year period.

#### Clustering

Now that we’ve examined the archetypal space, we can cluster the houses
based on their archetypes. This is more effective than clustering on the
raw data because the raw data has noise and correlated attributes.

``` r
kmeans_data <- h2o.cbind(houses_data, X)

kmeans_model <- h2o.kmeans(kmeans_data, x = c("Arch1",  "Arch2"), estimate_k = TRUE, k = 100, seed = 1234)
```

We can let H2O choose the best number of clusters by speciying
`estimate_k = TRUE`.

We will add the clusters to our original visualization.

``` r
# Visualize employee clusters
clusters <- h2o.predict(kmeans_model, kmeans_data)
colnames(clusters) <- "cluster"
kmeans_data <- h2o.cbind(kmeans_data, clusters)
```

Let’s visualize the clusters:

``` r
kmeans_df <- as.data.frame(kmeans_data[c("Arch1", "Arch2", "cluster", "id")])[sample_indices, ]
kmeans_df$cluster <- as.factor(kmeans_df$cluster)
plot_ly(data = kmeans_df, x = ~Arch1, y = ~Arch2, mode = "markers", text = ~paste0("House: ", id), color = ~cluster)
```

![clusters](./images/clusters.png) There are two clusters found. If we
examine these clusters, we can see that the top cluster represents more
expensive and larger houses with a higher grade.

``` r
cluster_stats <- h2o.group_by(kmeans_data, "cluster", mean("price"), mean("sqft_lot"), mean("bedrooms"), mean("grade"))
```

|  cluster|  mean\_price|  mean\_sqft\_lot|  mean\_bedrooms|  mean\_grade|
|--------:|------------:|----------------:|---------------:|------------:|
|        0|     392913.4|         10460.50|        2.974798|     4.997285|
|        1|     716569.1|         20678.67|        3.845747|     6.447904|

Filling in Missing Entries
--------------------------

Another use of GLRM would be to fill in any missing entries. We can view
missing entries as NA’s or events that have not yet happened. For
example, if we had data of customers and products, we could place NA’s
where a user has not yet purchased a product. In this way, by using GLRM
to impute missing values, we are also creating a recommender.

For our houses dataset, we do not having any missing values. We will
manually add missing values to our dataset to show how to use GLRM for
imputation.

### Split the Data

Given a training frame A, GLRM will produce a matrix factorization X and
Y. During validation, it approximately reconstructs A as a function of
the matrix product XY. This gives A = f(XY) which has the same number of
dimensions and features as A. Any missing values in A will also be
automatically filled in. Validation occurs by comparing A with the
validation frame and computing the squared error (for numeric entries)
and the number of mismatches (for categorical entries).

The correct way to do validation in GLRM is to take a dataset with no
(or very few) missing entries as your validation frame B. Create a deep
copy of B, then deliberately replace known values with missing values.

``` r
houses_data <- h2o.assign(houses_data, "houses_data.hex")
miss_data <- h2o.assign(houses_data, "miss_data.hex")
h2o.insertMissingValues(data = miss_data, fraction = 0.15, seed = 1234)
```

We can now compare the number of NA’s between the two datasets:

``` r
sum(is.na(houses_data))
```

    ## [1] 0

``` r
sum(is.na(miss_data))
```

    ## [1] 68240

### Determining Best Parameters

The main parameters in the GLRM model are associated with rank,
regularization, loss, and initialization:

#### Rank

The size of the matrices *X* and *Y*. In the diagram, the rank is
denoted as `k`.

![](./images/GLRM.png)

#### Regularizations

| structure    | regularization\_x | regularization\_y |
|--------------|-------------------|-------------------|
| small        | Quadratic         | Quadratic         |
| sparse       | L1                | L1                |
| non negative | Non Negative      | Non Negative      |
| clustered    | Sparse            | None              |

#### Loss

| data type   | loss          | L(u, a)                                                  |
|-------------|---------------|----------------------------------------------------------|
| real        | quadratic     | ![quadratic\_loss](./images/quadratic_loss.gif)          |
| real        | absolute      | ![absolute\_loss](./images/absolute_loss.gif)            |
| real        | huber         | ![huber\_loss](./images/huber_loss.gif)                  |
| boolean     | hinge         | ![hinge\_loss](./images/hinge_loss.gif)                  |
| boolean     | logistic      | ![logistic\_loss](./images/logistic_loss.gif)            |
| integer     | poisson       | ![poisson\_loss](./images/poisson_loss.gif)              |
| ordinal     | ordinal hinge | ![ordinal\_hinge\_loss](./images/ordinal_hinge_loss.gif) |
| categorical | one-vs-all    | ![one\_vs\_all\_loss](./images/one_vs_all_loss.gif)      |

#### Initialization

The `init` parameter specifies how the X and Y matrices are initially
generated.

-   **Random:** random arrays from normal distribution
-   **PlusPlus:** initialization using the clusters from k-means++
    initialization.
-   **SVD:** initialization using the first k right singular values.
    Helps with global convergence for matrix factorizations where global
    convergence not guaranteed (ex Non Negative Matrix Factorization)

In this tutorial, we will start by building a model using the default
parameters. We will set the rank (`k`) of the model to 5 and the
regularization strength to 2. We will tune these parameters later on. We
will also tell the GLRM model to use Ordinal loss for our ordinal
columns: `view`, `condition`, and `grade`.

``` r
glrm_k <- 5
gamma <- 2

losses <- data.frame('index' = which(colnames(houses_data) %in% c("condition", "grade", "view")) - 1,
                     'loss' = rep("Ordinal", 3), stringsAsFactors = FALSE)

base_model <- h2o.glrm(training_frame = miss_data, cols = glrm_cols, validation_frame = houses_data, 
                       model_id = "base_glrm", seed = 1234, 
                       k = glrm_k, gamma_x = gamma, gamma_y = gamma, 
                       regularization_x = "Quadratic", regularization_y = "Quadratic",
                       loss_by_col_idx = losses$index, loss_by_col = losses$loss)
```

Let’s take a look at the validation error.

``` r
print(h2o.performance(base_model, valid = T))
```

    ## H2ODimReductionMetrics: glrm
    ## ** Reported on validation data. **
    ## 
    ## Sum of Squared Error (Numeric):  3.262888e+15
    ## Misclassification Error (Categorical):  57857
    ## Number of Numeric Entries:  302582
    ## Number of Categorical Entries:  86452

We have a very high numeric error. We are also misclassifying more than
50% of the categorical values.  
Let’s improve this by transforming the numerical values in the dataset.
This is advised because some columns may have very large means which may
cause our model to be biased.

``` r
standardized_model <- h2o.glrm(training_frame = miss_data, cols = glrm_cols, validation_frame = houses_data,
                               model_id = "standardized_glrm", seed = 1234,
                               k = glrm_k, gamma_x = gamma, gamma_y = gamma, 
                               regularization_x = "Quadratic", regularization_y = "Quadratic",
                               transform = "STANDARDIZE", impute_original = TRUE,
                               loss_by_col_idx = losses$index, loss_by_col = losses$loss)
```

Since we are transforming the data, we make sure to include
`impute_original = TRUE`. This will reverse the transformation during
prediction.

``` r
print(h2o.performance(standardized_model, valid = T))
```

    ## H2ODimReductionMetrics: glrm
    ## ** Reported on validation data. **
    ## 
    ## Sum of Squared Error (Numeric):  8.935894e+14
    ## Misclassification Error (Categorical):  62968
    ## Number of Numeric Entries:  302582
    ## Number of Categorical Entries:  86452

Here we see a great reduction in the numeric error.

### Effects of Rank

Rank controls the size of our decomposed matrices. If our rank is too
low, we will have high error because we are not including some
archetypes. If our rank is too high, our X and Y matrices are
unnecessarily large and this may cause overfitting. We will examine how
different ranks effect the reconstruction error on the validation data
by training a GLRM model for the ranks 1, 3, 5, …, 21.

``` r
library('plyr')
k_range <- seq(1, 21, 4)

k_models <- llply(k_range, function(k) 
  h2o.glrm(training_frame = miss_data, cols = glrm_cols, validation_frame = houses_data, 
           model_id = paste0("glrm_k_", k), seed = 1234,
           k = k, gamma_x = gamma, gamma_y = gamma, 
           regularization_x = "Quadratic", regularization_y = "Quadratic", 
           transform = "STANDARDIZE", impute_original = TRUE,
           loss_by_col = losses$loss, loss_by_col_idx = losses$index))
```

Below we generate a scree plot of the numeric error for each model.

``` r
# Plot Numeric Error and Rank

k_error <- ldply(k_models, function(ml) 
  data.frame('k' = ml@parameters$k, 
             'numeric_error' = c(h2o.performance(ml, valid = FALSE)@metrics$numerr,
                                 h2o.performance(ml, valid = TRUE)@metrics$numerr),
             'type' = c("training", "validation"))
)

plot_ly(data = k_error, x = ~k, y = ~numeric_error, mode = "markers+lines", color = ~type, text = ~paste0("k: ", k))
```

![k\_error](./images/k_error.png) Based on our scree plot, it looks like
our optimal k is 17. After this value, the improvement plateaus. If our
aim was to lower the dimensionality of the data, we could choose the
elbow of the graph as `k`. In this case, the elbow is round `k = 5`.

The scree plot of the validation error should helps us to determine the
true rank of the data. The true rank of the data should encompass the
signal of the data, not the noise. When we see that the validation error
is increasing as k increases, the low rank matrix is beginning to
capture not just the signal but also the noise.

### Effects of Regularization Strength

Regularization strength is especially important if there are a lot of
missing values in the data to prevent overfitting. Regularization is
added to the loss function so that our new goal is to mimimize:

![loss\_equation](./images/loss_equation.gif)

We will examine how different regularization strengths effect the
reconstruction error on the validation data when `k = 17`. In a future
version of this tutorial, we will run a grid search over both `k` and
`gamma` since the effects of each parameter are dependent of each other.

``` r
gamma_range <- seq(0, 5, 1)

gamma_models <- llply(gamma_range, function(g) 
  h2o.glrm(training_frame = miss_data, cols = glrm_cols, validation_frame = houses_data, 
           model_id = paste0("glrm_gamma_", g), seed = 1234,
           k = 17, gamma_x = g, gamma_y = g, 
           regularization_x = "Quadratic", regularization_y = "Quadratic", 
           transform = "STANDARDIZE", impute_original = TRUE,
           loss_by_col = losses$loss, loss_by_col_idx = losses$index))
```

Below we generate a scree plot of the numeric error for each model.

``` r
# Plot Numeric Error and Regularization Strength/Gamma
gamma_error <- ldply(gamma_models, function(ml) 
  data.frame('gamma' = ifelse(is.null(ml@parameters$gamma_x), 0,  ml@parameters$gamma_x),
             'numeric_error' = c(h2o.performance(ml, valid = FALSE)@metrics$numerr,
                                 h2o.performance(ml, valid = TRUE)@metrics$numerr),
             'type' = c("training", "validation"))
)

plot_ly(data = gamma_error, x = ~gamma, y = ~numeric_error, mode = "markers+lines", 
        color = ~type, text = ~paste0("gamma: ", gamma))
```

![k\_error](./images/gamma_error.png)

The relationship between regularization strength and validation error
shows a different story than what we saw with rank. Some regularization
helps improve the validation error but too much hurts the performance.
It looks like we have the best performance with a regularization
strength around 1.

### Our Final Model

Below we generate the final model and view the performance metrics.

``` r
final_glrm <- gamma_models[[2]]
```

``` r
print(h2o.performance(final_glrm, valid = TRUE))
```

    ## H2ODimReductionMetrics: glrm
    ## ** Reported on validation data. **
    ## 
    ## Sum of Squared Error (Numeric):  1.576372e+14
    ## Misclassification Error (Categorical):  62837
    ## Number of Numeric Entries:  302582
    ## Number of Categorical Entries:  86452

### Matrix Completion and Denoising

Matrix Completion and Denoising occur by multiplying the matrices X and
Y (if the original data was numeric) or using the function `h2o.predict`
(in the event of mixed data types).

In our `miss_data`, we manually created ~68k missing values. We will use
our final GLRM model to impute these missing values.

``` r
# Predict data based on low rank matrices
houses_pred <- h2o.predict(final_glrm, miss_data)

# Add House ID
houses_pred <- h2o.cbind(miss_data$id, houses_pred)
```

Let’s take a look at the number of missing values in the predicted data.

``` r
print(sum(is.na(miss_data[glrm_cols])))
```

    ## [1] 58488

``` r
print(sum(is.na(houses_pred[c(2:ncol(houses_pred))])))
```

    ## [1] 0

Let’s examine the first record which had missing values added to the
features `floors` and `sqft_above`.

|  floors|  sqft\_above|
|-------:|------------:|
|       1|         1180|

|  reconstr\_floors|  reconstr\_sqft\_above|
|-----------------:|----------------------:|
|          1.229246|               1181.704|

The GLRM model is able to correctly estimate the number of floors and
the sqft of the interior above the ground level.

Compressing Data
----------------

If the goal of GLRM is to compress data, then choosing the best rank of
the model may be done slightly differently. As seen before, we want to
minimize the error on a validation dataset, however, now we need to keep
in mind the compression rate.

Compression Rate = Number of Non Zeros is X and Y Matrices/ Number of
Non Zeros in original data

If the decomposed matrices are dense, then the compression rate is
simply =
`(num rows of X + num cols of Y)*k/(num cols of A * num rows of A)`
where A is the original data.

We have a dense decomposition because we have used the quadratic
regularization. Therefore, the compression rate is
`((1470 + 77)*9)/(1470*35) = 27%`. We may choose to compress further by
using an L1 regularization instead of L2.

If we need to achieve a compression rate of 10%, then we would choose
the rank that minimizes the validation error under the restriction that
the compression rate is &lt;= 10%.
