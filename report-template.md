# Module 12 Report Template

## Overview of the Analysis

The purpose of this analysis:
To train and evaluate a model based on loan risk using a dataset of historical lending activity from a peer-to-peer lending services company to identify the creditworthiness of borrowers.

We analyzed a dataset containing customer account information to create a logistic regression model where A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

The dataset contains the following column information in 77507 accounts. 75306 healthty loans, and 2500 high-risk:
  loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks and total_debt.	

The following steps were utilized to create the model and analyze the data:

Creating a logistic logression model steps:

1. segregate the independent variables in data frames X and the dependent variable y.
2. split the dataset into training and testing sets with the help of train_test_split(). 
3. create an instance of LogisticRegression() function for logistic regression.
4. fit our model to the training data with the help of fit() function.
5. Evaluate the model’s performance by calculating the accuracy score of the model,
    generating a confusion matrix, and printing a classification report.
    
We then restructured the data using The RandomOverSampler module from imbalanced-learn.
This was because the data had 75036 healthy loans and only 2500 high risk loans.

We ran the logistic regression model again using the resampled data to fit the model and make predictions.
We then evaluated this model’s performance by calculating the accuracy score of the model,
    generating a confusion matrix, and printing a classification report.


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  
  Test Acc: 0.995
  
 
           precision    recall  f1-score   support

           0       1.00      1.00      1.00     18759
           1       0.87      0.89      0.88       625

    accuracy                           0.99     19384
   macro avg       0.94      0.94      0.94     19384
weighted avg       0.99      0.99      0.99     19384

Precision is a measure of how many True Positives were actually correct. It is defined as the ratio of true positives (TP) to the sum of true and false positives (TP+FP).

     * Precision = TP / (TP + FP) = 1.00 for Healthy, .87 for high risk.

Recall, is the ratio of true positives to the sum of true positives and false negatives (TP+FN). It shows how many of the positive predictions were actually correct.

    * Recall = TP / (TP + FN) = 1.00 for Healthy, .89 for high-risk.

F1 score is the harmonic mean of precision and recall. An F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.

    * F1 Score = 2*(Recall * Precision) / (Recall + Precision) = 1.00 for healthy, .88 for high-risk.

Support is the number of actual occurrences of the class in the dataset. It does not play a role in the calculation of the above metrics but is useful to have a look at, because it gives us a notion of how the given classes are distributed.

    * Support = 18759 Healthy, 625 high-risk.





* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  
  Test Acc: 0.995
  
      precision    recall  f1-score   support

           0       0.99      0.99      0.99     75036
           1       0.99      0.99      0.99     75036

    accuracy                           0.99    150072
   macro avg       0.99      0.99      0.99    150072
weighted avg       0.99      0.99      0.99    150072

    * Precision = TP / (TP + FP) = .99 for Healthy, .99 for high risk.
    * Recall = TP / (TP + FN) = .99 for Healthy, .99 for high-risk.
    * F1 Score = 2*(Recall * Precision) / (Recall + Precision) = 1.00 for healthy, .99 for high-risk.
    * Support = 75036 Healthy,  75036 high-risk.

This model is evenly balanced compared to the Model with Original Data.

  

## Summary

We recommend the bank implement model #2 based on the better scores for predicting un-healthy loans.
This will help us maximize revenue from these loans without granting credit to unworthy customers.
