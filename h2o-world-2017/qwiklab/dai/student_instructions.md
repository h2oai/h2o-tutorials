## Introduction to Driverless AI

### ***Caution:***  Do not navigate away from this page!  Open new browser tabs instead!

H2O Driverless AI is an artificial intelligence (AI) platform that automates some of the most difficult data science and machine learning workflows such as feature engineering, model validation, model tuning, model selection and model deployment. It aims to achieve the highest predictive accuracy, comparable to expert data scientists, but in much shorter time thanks to end-to-end automation. Driverless AI also offers automatic visualizations and machine learning interpretability (MLI). Especially in regulated industries, model transparency and explanation are just as important as predictive performance.

The Driverless AI documentation (including the Driverless AI User Guide and the MLI booklet) is available on <http://docs.h2o.ai>. These docs can also be accessed directly within Driverless AI.

Here are some additional resources that may be referenced by the instructor during the lab:

* [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset)
* [Driverless AI User Guide](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/index.html)
* [List of Driverless AI Feature Engineering Transformations](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/transformations.html)

Here is more information about Driverless AI you might find interesting for later:

* [InfoWorld Review of Driverless AI](https://www.infoworld.com/article/3236048/machine-learning/review-h2oai-automates-machine-learning.html)
* [H2O World 2017 Video - AutoViz by Leland Wilkinson (from H2O.ai)](https://www.youtube.com/watch?v=bas3-Ue2qxc&index=16&list=PLNtMya54qvOHQs2ZmV-pPSW_etMUykE0_)
* [H2O World 2017 Video - Driverless AI Hands-On by Arno Candel (from H2O.ai)](https://www.youtube.com/watch?v=niiibeHJtRo&list=PLNtMya54qvOHQs2ZmV-pPSW_etMUykE0_&index=6)
* [H2O World 2017 Video - MLI Hands-On by Mark Chan, Navdeep Gill, Patrick Hall (from H2O.ai)](https://www.youtube.com/watch?v=axIqeaUhow0&index=32&list=PLNtMya54qvOHQs2ZmV-pPSW_etMUykE0_)
* [H2O World 2017 Video - Drive Away Fraudsters with Driverless AI by Venkatesh Ramanathan (from PayPal)](https://www.youtube.com/watch?v=r9S3xchrzlY&list=PLNtMya54qvOHQs2ZmV-pPSW_etMUykE0_&index=24&t=1416s)
* [Driverless AI Webinars](https://www.gotostage.com/channel/4a90aa11b48f4a5d8823ec924e7bd8cf)
* [H2O4GPU Github Repository - Open Source Algorithms Used Inside Driverless AI](https://github.com/h2oai/h2o4gpu)

### Lab Instructional Steps

#### Step 1:  Find your Driverless AI license key.

If you don't have a Driverless AI license key yet, you can get a trial here:

* <https://www.h2o.ai/try-driverless-ai/>

#### Step 2:  WAIT for the instructor's directions before doing anything.

#### Step 3:  Click the green "Start Lab" button above.

The lab can take a few minutes to start. 

#### Step 4:  After the lab instance boots, copy the "DriverlessAI" URL at the left, and paste it into a new browser tab.

Qwiklab is browser agnostic, so any browser should work. 

#### Step 5:  Wait for Driverless AI to come up.

You may see a "502 Bad Gateway" error message for several minutes while Driverless AI starts.  This is harmless.  Just keep refreshing the page.

#### Step 6:  Scroll down to the bottom of the evaluation agreement and click "I agree to these terms".

#### Step 7:  Enter a username and password.

* username:  h2o
* password:  h2o

#### Step 8:  Click "Enter License". 

Paste your license key into the UI and then click **Save**.

#### Step 9: Import datasets by clicking "Add Dataset".

For this lab, we're going to import the CreditCard training and testing datasets. Select the "data/Kaggle/CreditCard" folder. Select the CreditCard-train.csv file and then click the **Import** button. 

Perform these same steps to import the CreditCard-test.csv file.

#### Step 10.  Visualize the dataset.

In the list of datasets, click on the **CreditCard-train** dataset and then click **Visualize** to start the internal visualization server. The graphs on the top row show you outliers or any information that’s unusual or that you might need to be aware of. The bottom row shows an overview of the data. 

Available graphs vary depending on your data and include:

- **Clumpy Scatterplots**: Clumpy scatterplots are 2D plots with evident clusters. These clusters are regions of high point density separated from other regions of points.
- **Correlated Scatterplots**: Correlated scatterplots are 2D plots with large values of the squared Pearson correlation coefficient.
- **Unusual Scatterplots**: Unusual scatterplots are 2D plots with features not found in other 2D plots of the data. 
- **Spikey Histograms**: Spikey histograms are histograms with huge spikes that often indicate an inordinate number of single values (usually zeros) or highly similar values.
- **Skewed Histograms**: Skewed histograms are histograms with especially large skewness (asymmetry). 
- **Varying Boxplots**: Varying boxplots reveal unusual variability in a feature across the categories of a categorical variable. 
- **Heteroscedastic Boxplots**: Heteroscedastic boxplots reveal unusual variability in a feature across the categories of a categorical variable.
- **Biplots**: A Biplot is an enhanced scatterplot that uses both points and vectors to represent structure simultaneously for rows and columns of a data matrix. Rows are represented as points (scores), and columns are represented as vectors (loadings). 
- **Outliers**: Outliers are variables with anomalous or outlying values.
- **Correlation Graph**: The correlation network graph is constructed from all pairwise squared correlations between variables (features). 
- **Radar Plot**: A Radar Plot is a two-dimensional graph that is used for comparing multiple variables, with variable having its own axis that starts from the center of the graph. 
- **Data Heatmap**: The heatmap is constructed from the transposed data matrix. Rows of the heatmap represent variables, and columns represent cases (instances). 
- **Missing Values Heatmap**: The missing values heatmap is constructed from the transposed data matrix. Rows of the heatmap represent variables and columns represent cases (instances). 

Click the **Help** button in the graph to see a detailed description of each plot. You can also refer to the following section in the Driverless AI User Guide: <http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/viewing-datasets.html>

Click through the graphs to visualize your data. 

#### Step 11. Run an experiment.

a. Click the **Experiments** link at the top of the UI, then click **New Experiment**.

b. Select the CreditCard-train dataset as the training dataset. 

c. Select the CreditCard-test dataset as the test dataset. This will provide you with a better idea of how well your experiment is performing. 

d. Select a Target column. For this experiment, pick the last column, which is the “default payment next month” column. This will predict the chance of someone defaulting next month. Once selected, Driverless AI shows you that this column has two unique values – YES or NO, or 0 or 1. The model will be attempting to predict whether a person will default on a payment next month. So the model will be a credit card default prediction model, binary classification. 

> **Note**: You can leave the remaining fields as their defaults, though you have the option to specify columns to drop (ignore), a fold column, a weight column, and a time column. You can specify a different accuracy or relative time tolerance or interpretability values. Hover over these fields or review the [Driverless AI documentation](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/launching.html#experiment-settings) for more information about these settings. 

e. Driverless AI allows you to select a scorer based on whether this is a classification or regression problem. For this lab, we'll select the Logloss scorer. 

f. Click **Launch Experiment** to start the experiment.

#### Step 12. View experiment dashboard.

You can see a status of what’s happening at the top of UI as the experiment is running. First, Driverless AI determines the backend (so whether GPUs are running). Then it starts parameter tuning. After that, Driverless AI performs feature engineering. Driverless AI uses early stopping and stops when overfitting starts. The number of trees is not fixed. You can click "Finish" at any time. This ends the experiment, then performs the ensembling, and generates the deployment package.

The top left and top right sections show the current experiment settings. The right side also includes a Trace. You can click on the **Trace** button to see what’s happening. Each green bar is a GPU or CPU model, each gray bar is a method call, and the red lines are feature engineering. You can hover over each to see what’s happening. If you watch, you’ll see this running new features and models and new features and models and so on. This is how Driverless AI determines the best features.

The bottom left section shows a chart with performance metrics out of fold. This only shows you validation scores. You can mouse over this section to learn more about this section. 

Finally, the lower middle section shows the variable importance. In this experiment you can see the features are pretty widely ranged. You can see a Pay times Pay, which are two different pay amounts that are multiplied. Driverless AI segments the payments columns into clusters and then computes the distance to cluster one. For example, these clusters could be high income and low income. This helps tell you why a decision is made. This is not a neural net that tells you a weight; this is instead telling you what the feature is.

For Variable Importance, Driverless AI performs frequency encoding and cluster target encoding. So first the data is clustered and then grouped, which basically says that in each cluster, these people belong together. Then Driverless AI computes the mean of some other columns in all of these clusters. All of this is done with cross validation so it provides fair estimates.

#### Step 13. View completed experiment.

After an experiment is complete, the top center of the UI provides you with a number of optional tasks that you can perform:

- You can interpret this model, which we'll do in the next step.
- You can score on another dataset, which tells Driverless AI to give the predictions of credit card default probabilities for each individual person in that dataset. This is provided as a download.
- You can transform another dataset that you already imported. You select your data, and Driverless AI provides the munged one. Information about Driverless AI transformations is available in the Driverless AI User Guide.
- You can download the training dataset - the out-of-fold predictions. With this, you can do more model training with the out-of-fold predictions.
- You can download the test predictions.
- You can download the logs.
- You can download the scoring package. The scoring package includes a scoring module and a scoring service that can be started in TCP or HTTP mode. If you have a Linux system, the scoring package will make a virtual environment that will run the examples included in the package. The examples are random data that is auto-generated with the same schema as this dataset.
- You can download the features and view their importance for the final model.

The bottom right section, where CPU/GPU monitoring was previously now shows a summary of the experiment. 

#### Step 14. Interpret the model

Click on the **Interpret This Model** button.

The Model Interpretation page includes the following charts. More information about these is available by click the tooltip in the UI. Information is also available in the MLI booklet and in the Driverless AI User Guide.

- The Global Interpretable Model Explanation Plot in the upper-left corner shows Driverless AI model predictions and K-LIME model predictions in sorted order by the Driverless AI model predictions.
- The Variable Importance plot in the upper-right corner shows the scaled global variable importance values for the features in the model. Driverless AI assigns local variable importance for every decision that the models makes. 
- The Decision Tree in the lower left corner shows an approximate flow-chart of the Driverless AI model’s decision-making process. This displays the most important variables and interactions in the model.
- The Partial Dependence chart in the lower right corner shows the partial dependence for a selected variable and the ICE values for a specific row. Select a point on the graph to see the specific value at that point. By default, this graph shows the partial dependence values for the top feature. Change this view by selecting a different feature in the feature drop-down. This graph is available for the top five features.

Model interpretation aims to justify every decision that this black box machine learning model makes, including which features contributed to each decision that each model made and, to a certain extent, how much each feature contributed to that decision. This information is then written out in plain English in the Explanations. To view an example, click on any point in the Global Interpretable Model Explanation Plot graph and the click the **Explanations** button at the top. 

### Summary

This concludes our overview of Driverless AI.  We encourage you to obtain a free trial license key so that you can try this out on your own data.  We welcome any feedback!
