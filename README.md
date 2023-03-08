# IDS721-Project3
## Introduction
This is the individual project3 for the course IDS721. I plan to build machine learning models in AWS SageMaker to train the model based on a dataset on [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) and test the trained model using SageMaker's Batch Transform functionality, so as to compare the predicted Boston house price with the median of the real house price. Also, I would implement S3 to store the data in the cloud.

## General Plan
1. Download the data from Kaggle
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Connect AWS SageMaker to S3.
5. Train/Test the model on AWS SageMaker.

## Project3 Requirements
* Use a major Big Data system to perform a Data Engineering related task
* Example systems could be: (AWS Athena, AWS Spark/EMR, AWS Sagemaker, Databricks, Snowflake)
