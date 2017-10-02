# Driverless AI

### Version 1.0.1

## Experiment Settings

### Test Data

* used to create test predictions only
* this dataset is not used for model scoring

### Dropped Columns

* columns that will not be used as predictors in the experiment

### Accuracy

| Accuracy | Max Rows | Ensemble Level | Target Transformation | Tune Parameters | Num Individuals | CV Folds | Only First CV Model | Strategy | 
|---|---|---|---|---|---|---|---|---|
| 1 | 100K | 0 | False | False | Default | 3 | True  | None |
| 2 | 500K | 0 | False | False | Default | 3 | True  | None |
| 3 | 1M   | 1 | False | False | Default | 3 | True  | None |
| 4 | 2.5M | 1 | False | False | Default | 3 | True  | None |
| 5 | 5M   | 1 | True  | False | Default | 3 | True  | None |
| 6 | 10M  | 2 | True  | True  | Default | 3 | True  | FS   |
| 7 | 20M  | 2 | True  | True  | 4       | 4 | False | FS   |
| 8 | 20M  | 2 | True  | True  | 4       | 4 | False | FS   |
| 9 | 20M  | 3 | True  | True  | 4       | 4 | False | FS   |
| 10| None | 3 | True  | True  | 8       | 4 | False | FS   |

* **Max Rows:** the maximum number of rows to use in model training.
	*  For classification, stratified random sampling done
	*  For regression, random sampling done
*  **Ensemble Level:** the level of ensembling done
	* 0: single final model
	* 1: 2 4-fold final models ensembled together
	* 2: 5 4-fold final models ensembled together
	* 3: 8 5-fold final models ensembled together
* **Target Transformation:** try target transformations and choose the transformation that has the best score 
	* Possible transformations: identity, log, square, square root, inverse, Anscombe, logit, sigmoid 
* **Tune Parameters:** tune the parameters of the xgboost model
	* Only max_depth tuned - range 3 to 10
	* Max depth chosen by `penalized_score` which is a combination of the model's accuracy and complexity
* **Num Individuals:** number of individuals in the population for the genetic algorithms
	* Each individual is a gene.  The more genes, the more combinations of features are tried.
	* Default is automatically determined.  Typical values are 4 or 8.
* **CV Folds:** the number of cross validation folds done for each model
	* if the problem is a classification problem, stratified folds are created 
* **Only First CV Model:** equivalent to splitting data into a training and testing set
	* Example: setting CV Folds to 3 and Only First CV Model = True means you are splitting the data into 66% training and 33% testing.
* **Strategy:** feature selection strategy
	* None: no feature selection
	* FS: feature selection permutations 
 
### Time

| Time | Epochs |
|---|---|
| 1  | 10  |
| 2  | 20  |
| 3  | 30  |
| 4  | 40  |
| 5  | 50  |
| 6  | 100 |
| 7  | 150 |
| 8  | 200 |
| 9  | 300 |
| 10 | 500 |

### Interpretability

| Interpretability | Strategy |
| --- | --- |
| <= 5 | None |
| > 5  | FS   |