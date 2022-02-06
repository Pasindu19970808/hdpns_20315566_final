After using pd.read_csv to read the txt file of the test dataset, use the existing "fraudDetectionclass" to "fitandtransform" your test dataset. 
This will return testing_data and testing_labels.
Then use gridsearch_dtree.best_estimator.predict_proba(testing_data) to get the predicted probabilities. 
Apply this result to the roc_auc function with the testing_labels.