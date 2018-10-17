# Best Practices - Segmentation

This section shows best practices for using unsupervised learning to understand and segment data.

**Goal:** learn about customer segments in retail data.

## The Data
The data is the public Kaggle data: <https://archive.ics.uci.edu/ml/datasets/wholesale+customers>

## Workflow
1. Explore the data - find columns that are correlated.
2. Cluster using K-Means.
3. Understand the clusters by plotting the clusters along different dimensions.
4. Train a Generalized Low Rank model on the data.
5. Plot the GLRM archetypes Y-coordinates to understand which columns are similar.
6. Plot the GLRM archetypes X-coordinates to view the data in a low dimensional (2-D) space.
7. Cluster the low dimensional data with K-Means.
8. Plot the new clusters along different dimensions.

**Caveat:**
Unsupervised clustering helps us understand data (in this example customers), but there may be more direct ways to solve a particular use case.  For example, if we wanted to determine which customers would react positively to a change in the delivery window, instead of clustering we should actually use supervised learning approaches.