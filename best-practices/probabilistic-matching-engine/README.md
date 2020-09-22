# Best Practices - Probabilistic Matching

This section shows best practices for using machine learning to solve a probabilistic matching use case.

**Goal:** show an example of using machine learning to link records and/or deduplicate data using probabilistic matching

## The Data
We use the `recordlinkage` package to load the data we will use during this tutorial: <https://recordlinkage.readthedocs.io/en/latest/notebooks/link_two_dataframes.html>.

## Workflow
1. Create matching dataset - we will need to formulate the data so that there is one record comparing two people 
2. Calculate string distances between the attributes of two people
3. Build machine learning models to predict if two people are a match
4. Use model to predict matches

Note: this workflow assumes that matches were previously done and saved by some process (for example manual matching).  Because of this assumption, we are able to use supervised learning to learn how matches were found in the past.