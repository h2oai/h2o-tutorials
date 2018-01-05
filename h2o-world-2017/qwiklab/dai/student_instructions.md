## Introduction to Driverless AI

H2O Driverless AI is an artificial intelligence (AI) platform that automates some of the most difficult data science and machine learning workflows such as feature engineering, model validation, model tuning, model selection and model deployment. It aims to achieve the highest predictive accuracy, comparable to expert data scientists, but in much shorter time thanks to end-to-end automation. Driverless AI also offers automatic visualizations and machine learning interpretability (MLI). Especially in regulated industries, model transparency and explanation are just as important as predictive performance.

### ***Caution:***  Do not navigate away from this page!  Open new browser tabs instead!

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

Paste your license key into the UI and then click "Save".

#### Step 9: Import datasets by clicking "Add Dataset".

For this lab, we're going to import the CreditCard training and testing datasets. Select the "data/Kaggle/CreditCard" folder. Select the CreditCard-train.csv file and click the **Import** button. 

Perform these same steps to import the CreditCard-test.csv file.

#### Step 10.  Visualize the dataset.

In the list of datasets, click on the **CreditCard-train** dataset and then click **Visualize** to start the internal visualization server. The graphs on the top row show you outliers or any information that’s unusual or that you might need to be aware of. The bottom row shows an overview of the data. 

Click through the graphs to visualize your data.

#### Step 11. Run an experiment.

a. Click the **Experiments** link at the top of the UI, then click **New Experiment**.

b. Select the CreditCard-train dataset as the training dataset. 

c. Select the CreditCard-test dataset as the test dataset. This will provide you with a better idea of how well your experiment is performing. 

d. Select a Target column. For this experiment, pick the last column, which is the “default payment next month” column. This will predict the chance of someone defaulting next month. Once selected, Driverless AI shows you that this column has two unique values – YES or NO, or 0 or 1. The model will be attempting to predict whether a person will default on a payment next month. So the model will be a credit card default prediction model, binary classification. 

> **Note**: You can leave the remaining fields as their defaults, though you have the option to specify columns to drop (ignore), a fold column, a weight column, and a time column. You can specify a different accuracy or relative time tolerance or interpretability values. Hover over these fields or review the Driverless AI documentation for more information about these settings.

e. Driverless AI allows you to select a scorer based on whether this is a classification or regression problem. For this lab, we'll select the Logloss scorer. 

f. Click **Launch Experiment** to start the experiment.

#### Step 12. View experiment dashboard.

You can see a status of what’s happening at the top of UI as the experiment is running. First, Driverless AI determines the backend (so whether GPUs are running). Then it starts parameter tuning. After that, Driverless AI performs feature engineering. Driverless AI uses early stopping and stops when overfitting starts. The number of trees is not fixed. You can click "Finish" at any time. This ends the experiment, then performs the ensembling, and generates the deployment package.

The top left and top right sections show the current experiemnt settings. The right side also includes a Trace. You can click on the **Trace** button to see what’s happening. Each green bar is a GPU or CPU model, each gray bar is a method call, and the red lines are feature engineering. You can hover over each to see what’s happening. If you watch, you’ll see this running new features and models and new features and models and so on. This is how Driverless AI determines the best features.

The bottom left section shows a chart with performance metrics out of fold. This only shows you validation scores. You can mouse over this section to learn more about this section. 

Finally, the lower middle section shows the variable importance. In this experiment you can see the features are pretty widely ranged. You can see a Pay times Pay, which are two different pay amounts that are multiplied. Driverless AI segments the payments columns into clusters and then computes the distance to cluster one. For example, these clusters could be high income and low income. This helps tell you why a decision is made. This is not a neural net that tells you a weight; this is instead telling you what the feature is.For Variable Importance, Driverless AI performs frequency encoding and cluster target encoding. So first the data is clustered and then grouped, which basically says that in each cluster, these people belong together. Then Driverless AI computes the mean of some other columns in all of these clusters. All of this is done with cross validation so it provides fair estimates.  

#### Step 13. View completed experiment.

After an experiment is complete, the top center of the UI provides you with a number of optional tasks that you can perform:

-	You can interpret this model, which we'll do in the next step.-	You can score on another dataset, which tells Driverless AI to give the predictions of credit card default probabilities for each individual person in that dataset. This is provided as a download.-	You can transform another dataset that you already imported. You select your data, and Driverless AI provides the munged one. Information about Driverless AI transformations is available in the Driverless AI User Guide.-	You can download the training dataset - the out-of-fold predictions. With this, you can do more model training with the out-of-fold predictions. -	You can download the test predictions.-	You can download the transformed training and test sets.-	You can download the logs.-  You can download the scoring package. The scoring package includes a scoring module and a scoring service that can be started in TCP or HTTP mode. If you have a Linux system, the scoring package will make a virtual environment that will run the examples included in the package. The examples are random data that is auto-generated with the same schema as this dataset.

The bottom right section, where CPU/GPU monitoring was previously now shows a summary of the experiment. 

#### Step 14. Interpret the model

Click on the **Interpret This Model** button.

The Model Interpretation page includes the following charts. More information about these is available by hovering over a chart to view the tooltips, in the Driverless AI User Guide, and in the MLI booklet.

- The Global Interpretable Model Explanation Plot in the upper-left corner shows Driverless AI model predictions and K-LIME model predictions in sorted order by the Driverless AI model predictions.
- The Variable Importance plot in the upper-right corner shows the scaled global variable importance values for the features in the model. Driverless AI assigns local variable importance for every decision that the models makes. 
- The Decision Tree in the lower left corner shows an approximate flow-chart of the Driverless AI model’s decision making process. This displays the most important variables and interatctions in the model. 
- The Partial Depencence chart in the lower right corner shows the partial dependence for a selected variable and the ICE values for a specific row. Select a point on the graph to see the specific value at that point. By default, this graph shows the partial dependence values for the top feature. Change this view by selecting a different feature in the feature drop-down. This graph is available for the top five features.

Model interpretation aims to justify every decision that this black box machine learning model makes, including which features contributed to each decision that each model made and, to a certain extent, how much each feature contributed to that decision. This information is then written out in plain English in the Explanations. To view an example, click on any point in the Global Interpretable Model Explanation Plot graph and the click the **Explanations** button at the top. 

#### Summary

This concludes our overview of Driverless AI. We encourage you to obtain a 30-day trial so that you can try this out on your own data, and we welcome any feedback! 

 



