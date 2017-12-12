# NLP with H2O Tutorial

The focus of this tutorial is to provide an introduction to H2O's Word2Vec algorithm. Word2Vec is an algorithm that trains a shallow neural network model to learn vector representations of words. These vector representations are able to capture the meanings of words. During the tutorial, we will use H2O's Word2Vec implementation to understand relationships between words in our text data. We will use the model results to find similar words and synonyms. We will also use it to showcase how to effectively represent text data for machine learning problems where we will highlight the impact this representation can have on accuracy. 

- More information and code examples are available in the [Word2Vec Documentation](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/word2vec.html)

## Supervised Learning with Text Data

For the demo, we use a subset of the [Amazon Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews) dataset.  The goal here is to predict whether or not an Amazon review is positive or negative. 

The tutorial is split up into three parts.  In the first part, we will train a model using non-text predictor variables.  In the second and third part, we will train a model using our text columns.  The text columns in this dataset are the review of the product and the summary of the review.  In order to leverage our text columns, we will train a Word2Vec model to convert text into numeric vectors.

### Initial Model - No Text

In this section, you will see how accurate your model is if you do not use any text columns.  You will: 

- Specify a training frame.
- Specify a test frame.
- Train a GBM model on non-text predictor variables such as: `ProductId`, `UserId`, `Time`, etc.
-  Analyze our initial model - AUC, confusion matrix, variable importance, partial dependency plots

### Second Model - Word Embeddings of Reviews

In this section, you will see how much your model improves if you include the word embeddings from the reviews. You will:

- Tokenize words in the review.
- Train a Word2Vec model (or import the already trained Word2Vec model: <https://s3.amazonaws.com/tomk/h2o-world/megan/w2v.hex>)
- Find synonyms using the Word2Vec model.
- Aggregate word embeddings - one word embedding per review.
- Train a GBM model using our initial predictors plus the word embeddings of the reviews.
- Analyze our second model - AUC, confusion matrix

### Third Model - Word Embeddings of Summaries

In this section, you will see if you can improve your model even more by also adding the word embeddings from the summary of the review. You will:

- Aggregate word embeddings of summaries - one word embedding per summary.
- Train a GBM model now including the word embeddings of the summary.
- Analyze our final model - AUC, confusion matrix, variable importance, partial dependency plot
- Predict on new reviews using our third and final model.


## Resources

- Demo Notebooks: [AmazonReviews.ipynb](./AmazonReviews.ipynb)
- The subset of the Amazon Reviews data used for this demo can be found here: <https://s3.amazonaws.com/tomk/h2o-world/megan/AmazonReviews.csv>
- The word2vec model that was trained on this data can be found here: <https://s3.amazonaws.com/tomk/h2o-world/megan/w2v.hex>
