import numpy as np 
from sklearn.datasets import load_diabetes, load_breast_cancer 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn.metrics import ( mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix )

# QUESTION 1 – Linear Regression Pipeline (Diabetes)

def diabetes_linear_pipeline():
    # STEP 1: Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2: Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # STEP 5: Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # STEP 6: Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # STEP 7: Top 3 features by absolute coefficient value
    coef_abs = np.abs(model.coef_)
    top_3_feature_indices = np.argsort(coef_abs)[-3:][::-1].tolist()

    # COMMENT:
    # Overfitting check:
    # If train R² is much higher than test R², the model may be overfitting.


    # Feature scaling importance:
    # Scaling ensures all features contribute equally.
    # Without scaling, features with larger units dominate coefficient learning.

    return (
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices
    )


# QUESTION 2 – Cross-Validation (Linear Regression)


def diabetes_cross_validation():
    # STEP 1: Load dataset
    data = load_diabetes()
    X = data.data
    y = data.target

    # STEP 2: Standardize entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3: 5-fold cross-validation
    model = LinearRegression()
    cv_scores = cross_val_score(
        model, X_scaled, y, cv=5, scoring='r2'
    )

    # STEP 4: Mean and standard deviation
    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)

    # COMMENT:
    # Standard deviation shows how much model performance varies across folds.
    # Lower std means more stable model.
    # Cross-validation reduces variance risk by testing the model
    # on multiple train-test splits instead of trusting one split.

    return mean_r2, std_r2


# QUESTION 3 – Logistic Regression Pipeline (Cancer)


def cancer_logistic_pipeline():
    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # STEP 4: Train Logistic Regression
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    # STEP 5: Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    # COMMENT:
    # False Negative in medical context:
    # A False Negative means the model predicts "healthy"
    # when the patient actually has cancer.
    # This is dangerous because it can delay treatment.

    return (
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    )


# QUESTION 4 – Logistic Regularization Path

def cancer_logistic_regularization():
    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # STEP 3: Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    C_values = [0.01, 0.1, 1, 10, 100]

    for C in C_values:
        model = LogisticRegression(max_iter=5000, C=C)
        model.fit(X_train_scaled, y_train)

        train_acc = accuracy_score(
            y_train, model.predict(X_train_scaled)
        )
        test_acc = accuracy_score(
            y_test, model.predict(X_test_scaled)
        )

        results[C] = (train_acc, test_acc)

    # COMMENT:
    # Overfitting occurs when C is very large.
    #If C is large, Model Complexity is large, if c is low, there is almost no regularization and thus mmodel becomes almost too simple 

    return results



# QUESTION 5 – Cross-Validation (Logistic Regression)


def cancer_cross_validation():
    # STEP 1: Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # STEP 2: Standardize entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # STEP 3: Cross-validation
    model = LogisticRegression(C=1, max_iter=5000)
    cv_scores = cross_val_score(
        model, X_scaled, y, cv=5, scoring='accuracy'
    )

    # STEP 4: Metrics
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)

    # COMMENT
    # Cross-validation ensures the model is reliable and not dependent
    # on a single lucky or unlucky train-test split.

    return mean_accuracy, std_accuracy
