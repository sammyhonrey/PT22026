
# Model Documentation
## House Price Prediction Using Machine Learning



## 1. Introduction

This document describes the machine learning model used in the House Price Prediction application.

The project uses a pre-trained Random Forest Regressor to estimate California house prices based on housing characteristics.

The model receives numerical and categorical input features, processes them through a saved preprocessing pipeline, and generates a numerical prediction representing the estimated median house value.

The purpose of this document is to explain the model architecture, algorithm, input features, preprocessing methods, prediction workflow, evaluation metrics, capabilities, and limitations.

### Project Information

| Field | Details |
|---|---|
| Project Name | House Price Prediction Using Machine Learning |
| Student | IMANE |
| Student ID | 20243120041 |
| Programming Language | Python |
| Machine Learning Framework | Scikit-learn |
| Algorithm | Random Forest Regressor |
| Model Format | Joblib |
| Application Framework | Gradio |


## 2. Model Overview

The application uses a pre-trained Random Forest Regression model.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to produce a prediction.

Unlike a single decision tree, which makes predictions using one tree structure, Random Forest combines the results of multiple trees.

For regression tasks, the final prediction is generally calculated by averaging the predictions from the individual trees.

The model used in this project contains 30 estimators according to its original model documentation.

### Model Configuration

| Parameter | Value |
|---|---:|
| Algorithm | Random Forest Regressor |
| Number of Estimators | 30 |
| Maximum Features | 8 |
| Training Samples | 16,512 |
| Testing Samples | 4,128 |
| Test Size | 20% |
| Random State | 42 |

The application loads the existing trained model from:

`models/house_price_model.joblib`

It does not train a new model when the application starts.



## 3. Random Forest Regression

### 3.1 What Is Random Forest?

Random Forest is an ensemble learning technique that creates multiple decision trees and combines their predictions.

Each decision tree learns relationships between input variables and the target variable.

For a regression problem, each tree generates a numerical output. The Random Forest combines these outputs to calculate the final prediction.

### 3.2 How It Works

The general process is:

1. Multiple decision trees are used as part of the Random Forest.
2. Each tree learns patterns from training data.
3. Input features are passed through the trees.
4. Each tree generates a numerical prediction.
5. The predictions are combined to produce the final result.

### 3.3 Regression Prediction

The final Random Forest prediction can generally be represented as:

\[
\hat{y} = \frac{1}{N}\sum_{i=1}^{N} T_i(x)
\]

Where:

- \(\hat{y}\) = Final predicted value.
- \(N\) = Number of decision trees.
- \(T_i(x)\) = Prediction from the i-th decision tree.
- \(x\) = Input feature vector.

For this project, the model uses 30 estimators according to its documented configuration.

### 3.4 Why Random Forest Was Used

Random Forest is appropriate for this project because:

- It supports numerical regression problems.
- It can learn nonlinear relationships.
- It combines multiple decision trees.
- It can work with different types of input features after preprocessing.
- It provides a practical approach to estimating housing values.

---

## 4. Dataset and Input Features

The model is associated with the California Housing dataset.

The dataset contains information about housing characteristics in different California districts.

The model receives nine input features.

### 4.1 Numerical Features

| Feature | Description |
|---|---|
| longitude | Geographic longitude of the district |
| latitude | Geographic latitude of the district |
| housing_median_age | Median age of houses in the district |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Total population |
| households | Number of households |
| median_income | Median household income |

### 4.2 Categorical Feature

The ninth input feature is:

**ocean_proximity**

This feature represents the geographical proximity of a district to the ocean.

The supported categories are:

- `<1H OCEAN`
- `INLAND`
- `ISLAND`
- `NEAR BAY`
- `NEAR OCEAN`

### 4.3 Target Variable

The target variable is:

`median_house_value`

It represents the median house value associated with a district in the dataset.

The model predicts a numerical value that is displayed in US dollars.



## 5. Data Preprocessing Pipeline

Data preprocessing is an important part of machine learning because raw input data may contain missing values and categorical information that cannot be directly processed by the model.

This project uses a saved Scikit-learn preprocessing pipeline.

The pipeline is stored in:

`models/preprocessing_pipeline.joblib`

It contains the preprocessing operations fitted during the original model development.

### 5.1 Numerical Data Processing

The eight numerical features are processed using two main techniques.

#### A. Simple Imputer

The Simple Imputer handles missing numerical values.

The pipeline uses:

`strategy="median"`

This means missing values are replaced using the median of the corresponding feature calculated during pipeline fitting.

#### B. Standard Scaler

StandardScaler transforms numerical values using the fitted mean and standard deviation.

The general formula is:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:

- \(x\) = Original feature value.
- \(\mu\) = Mean calculated during fitting.
- \(\sigma\) = Standard deviation calculated during fitting.
- \(z\) = Standardized feature value.

The saved pipeline applies the original fitted scaling parameters.

### 5.2 Categorical Data Processing

The `ocean_proximity` feature is processed using One-Hot Encoding.

One-Hot Encoding converts categorical values into numerical indicator features.

For example, the five supported categories are represented by five separate binary features.

This allows the Random Forest model to process ocean proximity information numerically.

### 5.3 Final Feature Transformation

The original input contains nine features:

- Eight numerical features.
- One categorical feature.

After preprocessing, the data contains thirteen transformed features:

- Eight processed numerical features.
- Five encoded categorical features.

The transformed feature names are:

1. num__longitude
2. num__latitude
3. num__housing_median_age
4. num__total_rooms
5. num__total_bedrooms
6. num__population
7. num__households
8. num__median_income
9. cat__ocean_proximity_<1H OCEAN
10. cat__ocean_proximity_INLAND
11. cat__ocean_proximity_ISLAND
12. cat__ocean_proximity_NEAR BAY
13. cat__ocean_proximity_NEAR OCEAN

The preprocessing pipeline ensures that the model receives data in the expected format.



## 6. Model Architecture

The system consists of two main machine learning components:

1. Preprocessing Pipeline.
2. Random Forest Regressor.

### 6.1 Architecture Diagram

```text
          Raw User Input
                 |
                 v
        Input Data Validation
                 |
                 v
          Pandas DataFrame
                 |
                 v
       Preprocessing Pipeline
                 |
          +------+------+
          |             |
          v             v
    Numerical Data   Categorical Data
          |             |
          v             v
     Imputation     One-Hot Encoding
          |
          v
    Standard Scaling
          |
          +------+------+
                 |
                 v
       13 Processed Features
                 |
                 v
       Random Forest Model
                 |
                 v
        Price Prediction
                 |
                 v
         Result in USD
```

### 6.2 Model Components

| Component | Responsibility |
|---|---|
| Input Validation | Checks whether user inputs are valid |
| Pandas DataFrame | Organizes input features |
| Simple Imputer | Handles missing numerical values |
| Standard Scaler | Standardizes numerical features |
| One-Hot Encoder | Converts categories into numerical features |
| Random Forest Regressor | Predicts the house value |
| Gradio Interface | Displays the prediction |



## 7. Prediction Workflow

The application follows a structured prediction process.

### Step 1: Receive User Input

The user enters housing information through the Gradio interface.

### Step 2: Validate Input

The application checks the input values before prediction.

Examples of validation rules include:

- Total bedrooms must not exceed total rooms.
- Households must not exceed population.
- Room and population values must be positive.
- Median income must be positive.

### Step 3: Create DataFrame

The application organizes the input values into a Pandas DataFrame.

### Step 4: Apply Preprocessing

The saved preprocessing pipeline transforms the input data.

### Step 5: Generate Prediction

The processed data is passed to the Random Forest model.

The prediction is generated using:

`model.predict(processed_data)`

### Step 6: Display Result

The application displays the predicted house price in US dollars.



## 8. Model Evaluation

The project includes a separate evaluation script named `evaluate_model.py`.

The script evaluates the existing pre-trained model using a reconstructed stratified test split.

The dataset is divided into training and testing portions for the evaluation procedure.

### 8.1 Evaluation Configuration

| Parameter | Value |
|---|---:|
| Dataset Records | 20,640 |
| Training Portion | 16,512 |
| Testing Portion | 4,128 |
| Test Size | 20% |
| Random State | 42 |

The split uses stratification based on median income categories.

The original model's exact training indices have not been independently verified. Therefore, these results describe the reconstructed evaluation rather than a confirmed reproduction of the original training experiment.

### 8.2 Mean Absolute Error (MAE)

Mean Absolute Error measures the average absolute difference between actual values and predicted values.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

Where:

- \(y_i\) = Actual value.
- \(\hat{y}_i\) = Predicted value.
- \(n\) = Number of test observations.

A lower MAE indicates smaller average prediction errors.

**Result: $31,678.01**

### 8.3 Root Mean Squared Error (RMSE)

Root Mean Squared Error measures the square root of the average squared prediction errors.

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

RMSE gives greater weight to larger prediction errors.

A lower RMSE generally indicates better prediction performance.

**Result: $47,859.54**

### 8.4 R-Squared Score

R-squared measures how much of the variation in the target variable is explained by the model on the evaluated dataset.

\[
R^2 = 1-\frac{SS_{res}}{SS_{tot}}
\]

Where:

- \(SS_{res}\) = Sum of squared residuals.
- \(SS_{tot}\) = Total sum of squares.

The R-squared score ranges from negative values to 1.

A value closer to 1 indicates a stronger fit on the evaluated data.

**Result: 0.8243**

This is an R-squared score, not classification accuracy.

### 8.5 Summary of Evaluation Results

| Metric | Result |
|---|---:|
| MAE | $31,678.01 |
| RMSE | $47,859.54 |
| R² | 0.8243 |

These metrics provide different perspectives on the model's regression performance.



## 9. Testing and Validation

The project includes an automated testing script named `test_cases.py`.

The purpose of this script is to verify the input validation rules and confirm that valid inputs can produce predictions.

### 9.1 Test Cases

| Test Case | Expected Behavior |
|---|---|
| Valid input | Generate a prediction |
| Bedrooms exceed rooms | Reject the input |
| Households exceed population | Reject the input |
| Zero total rooms | Reject the input |
| Negative median income | Reject the input |
| Different ocean category | Generate a prediction |

### 9.2 Test Results

The current execution produced:

```text
MODEL TEST CASES
=============================================
PASS: Valid input | $440,803.57
PASS: Bedrooms exceed rooms | Rejected correctly
PASS: Households exceed population | Rejected correctly
PASS: Zero total rooms | Rejected correctly
PASS: Negative median income | Rejected correctly
PASS: Different ocean category | $379,556.80
=============================================
PASSED: 6/6
ALL TESTS PASSED
```

All six test cases passed.

This confirms that the tested prediction and validation paths behaved as expected.



## 10. Model Capabilities

The model provides several useful capabilities:

- Predicts estimated house values from housing characteristics.
- Accepts numerical and categorical features.
- Processes missing numerical values through its saved preprocessing pipeline.
- Supports five ocean proximity categories.
- Can be integrated into a web application.
- Produces numerical regression predictions.
- Can be reused without retraining during application startup.



## 11. Model Limitations

The model also has limitations that should be considered.

### 11.1 Historical Dataset

The model is based on historical California housing data.

Its predictions may not represent current property prices.

### 11.2 Geographic Scope

The model is designed for California housing data.

It should not be assumed to produce reliable predictions for other countries or regions.

### 11.3 Prediction Uncertainty

A prediction is an estimate rather than a guaranteed market value.

Actual house prices may differ due to factors not represented in the dataset.

### 11.4 Input Quality

Incorrect or unrealistic input values may affect prediction quality.

The application includes basic validation, but not every possible unrealistic input can be detected.

### 11.5 Evaluation Reproducibility

The evaluation uses a reconstructed stratified split.

The exact original model training indices have not been independently verified.



## 12. Model Files and Dependencies

The application requires two saved model assets.

| File | Description |
|---|---|
| house_price_model.joblib | Pre-trained Random Forest Regressor |
| preprocessing_pipeline.joblib | Saved preprocessing pipeline |

The files are loaded using Joblib.

The project dependencies include:

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Gradio

The dependency versions are listed in `requirements.txt`.

The model files are available from the original Hugging Face repository:

https://huggingface.co/niru-nny/house-price-prediction



## 13. Conclusion

This document explains the machine learning model and its integration into the House Price Prediction application.

The system combines a saved preprocessing pipeline with a pre-trained Random Forest Regressor to generate estimated California house prices.

It demonstrates important machine learning concepts, including regression, ensemble learning, missing-value imputation, feature scaling, categorical encoding, model evaluation, and software testing.

The application also demonstrates how an existing machine learning model can be integrated into an interactive Python application using Gradio.

The successful execution of the six automated test cases confirms that the tested input validation and prediction workflows are functioning as expected.



## 14. References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. Random Forest Documentation: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
3. Scikit-learn Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
4. Pandas Documentation: https://pandas.pydata.org/docs/
5. NumPy Documentation: https://numpy.org/doc/
6. Joblib Documentation: https://joblib.readthedocs.io/
7. Gradio Documentation: https://www.gradio.app/docs
8. Original Hugging Face Model Repository: https://huggingface.co/niru-nny/house-price-prediction



**Document:** Model Documentation  
**Project:** House Price Prediction Using Machine Learning
