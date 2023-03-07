"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression


from regression import BaseRegressor, LogisticRegressor, loadDataset


# Load the dataset
X_train, X_test, y_train, y_test = loadDataset(features=['NSCLC', 'GENDER', 'Penicillin V Potassium 250 MG', 'Penicillin V Potassium 500 MG',
'Computed tomography of chest and abdomen', 'Plain chest X-ray (procedure)', 'Diastolic Blood Pressure',
'Body Mass Index', 'Body Weight', 'Body Height', 'Systolic Blood Pressure',
'Low Density Lipoprotein Cholesterol', 'High Density Lipoprotein Cholesterol', 'Triglycerides',
'Total Cholesterol', 'Documentation of current medications',
'Fluticasone propionate 0.25 MG/ACTUAT / salmeterol 0.05 MG/ACTUAT [Advair]',
'24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet',
'Carbon Dioxide', 'Hemoglobin A1c/Hemoglobin.total in Blood', 'Glucose', 'Potassium', 'Sodium', 'Calcium',
'Urea Nitrogen', 'Creatinine', 'Chloride', 'AGE_DIAGNOSIS'], split_percent=0.7, split_seed=42)

###testing prediction accuracy based on an arbitrary cutoff
def test_prediction():
    
    # Training the logistic regression model
    model = LogisticRegressor(num_feats=X_train.shape[1], learning_rate=0.001, tol=0.00001, max_iter=10000, batch_size=115)
    model.train_model(X_train, y_train, X_test, y_test)
    
    # Predict labels for test set
    X_test_with_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    y_pred = np.round(model.make_prediction(X_test_with_bias))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Checking that accuracy is above a reasonable threshold/significantly better than random assignment
    assert accuracy > 0.65



###testing loss function values against sklearn log-regression implementation (w/ admittedly loose error tolerance...)
def test_loss_function():
    
    # Training the logistic regression model
    model = LogisticRegressor(num_feats=X_train.shape[1], learning_rate=0.001, tol=0.00001, max_iter=10000, batch_size=115)
    model.train_model(X_train, y_train, X_test, y_test)
    
    # Predict labels for test set
    X_test_with_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    y_pred = model.make_prediction(X_test_with_bias)

    # Calculate loss using my implementation and scikit-learn's implementation
    my_loss = model.loss_function(y_test, y_pred)
    sklearn_model = LogisticRegression(max_iter=10000, C=1.0, solver='lbfgs', penalty='l2', random_state=42)
    sklearn_model.fit(X_train, y_train)
    sklearn_loss = log_loss(y_test, sklearn_model.predict_proba(X_test)[:, 1])

    # Comparing my loss to scikit-learn's loss
    assert np.isclose(my_loss, sklearn_loss, rtol=.75, atol=.75)


def test_gradient():

    # Training the logistic regression model
    model = LogisticRegressor(num_feats=X_train.shape[1], learning_rate=0.001, tol=0.00001, max_iter=10000, batch_size=115)
    model.train_model(X_train, y_train, X_test, y_test)

    # Select a random instance from the test set
    i = np.random.randint(X_test.shape[0])

    # Calculate gradient using my implementation
    X_i_with_bias = np.hstack([X_test[i], [1]])
    grad_my = model.calculate_gradient(y_test[i], X_i_with_bias)

    # Calculate gradient using finite differences
    epsilon = 1e-6
    f_x_plus_epsilon = model.loss_function(y_test[i], model.make_prediction(X_i_with_bias + epsilon))
    f_x_minus_epsilon = model.loss_function(y_test[i], model.make_prediction(X_i_with_bias - epsilon))
    grad_fd = (f_x_plus_epsilon - f_x_minus_epsilon) / (2 * epsilon)

    # Comparing my gradient to the finite differences gradient
    assert np.isclose(grad_my, grad_fd, rtol=.25, atol=.25)


def test_training():
        
    # Training the logistic regression model
    model = LogisticRegressor(num_feats=X_train.shape[1], learning_rate=0.001, tol=0.00001, max_iter=10000, batch_size=115)
    model.train_model(X_train, y_train, X_test, y_test)

    # Save initial weights
    initial_weights = model.W.copy()

    # Train model for more iterations
    model.train_model(X_train, y_train, X_test, y_test)

    # Save final weights
    final_weights = model.W.copy()

    # Checking that the weights have updated
    assert not np.allclose(initial_weights, final_weights, rtol=1e-3)
