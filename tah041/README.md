
# House Price Prediction Using Machine Learning

<p align="center">
  <strong>An Interactive California House Price Prediction Application</strong>
</p>


## 1. Project Information

| Field | Details |
|---|---|
| Project Title | House Price Prediction Using Machine Learning |
| Student Name | IMANE |
| Student ID | 20243120041 |
| Course | Professional Training |
| Course Code | YN3012170034 |
| Project Level | Intermediate |
| Programming Language | Python |
| Machine Learning Algorithm | Random Forest Regressor |
| Application Framework | Gradio |
| Machine Learning Library | Scikit-learn |



## 2. Introduction

Machine Learning is widely used to analyze historical data and make predictions that support real-world decision-making.

The real estate market is one area where machine learning can be useful. House prices depend on several factors, including geographical location, property age, number of rooms, population density, household information, and income levels.

This project, **House Price Prediction Using Machine Learning**, develops an interactive application that estimates California house prices using a pre-trained Random Forest Regression model.

Users can enter housing characteristics through a simple web interface and receive an estimated house price in US dollars.

The application uses an existing pre-trained model and its corresponding preprocessing pipeline. It does not train a new model when the application starts.


## 3. Problem Statement

Estimating house prices can be challenging because property values are influenced by multiple interconnected variables.

Traditional estimation methods may not effectively capture complex relationships between housing characteristics and property values.

This project aims to provide an automated prediction system that:

- Accepts different housing characteristics as input.
- Processes numerical and categorical information.
- Uses a machine learning model to estimate house prices.
- Displays predictions through an interactive interface.
- Validates user inputs before generating predictions.



## 4. Project Objectives

The main objectives are:

1. Understand how machine learning can be applied to real-world housing problems.
2. Work with a real-world California housing dataset.
3. Use an existing pre-trained Random Forest Regression model.
4. Understand data preprocessing techniques.
5. Develop an interactive application using Python and Gradio.
6. Implement input validation and error handling.
7. Evaluate the model using standard regression metrics.
8. Create automated test cases.
9. Organize the project using modular and maintainable Python code.



## 5. Dataset Description

The project uses the California Housing dataset associated with the pre-trained model.

The dataset contains housing-related information collected from different California districts.

### Dataset Features

| Feature | Description |
|---|---|
| longitude | Geographic longitude of the district |
| latitude | Geographic latitude of the district |
| housing_median_age | Median age of houses in the district |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Total population of the district |
| households | Number of households |
| median_income | Median household income |
| ocean_proximity | Category describing proximity to the ocean |
| median_house_value | Target variable representing median house value |

### Dataset Characteristics

- Total records used: 20,640.
- Total columns: 10.
- Target variable: `median_house_value`.
- Numerical and categorical features are included.
- The `total_bedrooms` column contains missing values.

Missing numerical values are handled by the saved preprocessing pipeline.

**Original model repository:**

https://huggingface.co/niru-nny/house-price-prediction


## 6. Machine Learning Algorithm

### Random Forest Regressor

The project uses a pre-trained Random Forest Regressor.

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees.

For regression tasks, individual decision trees produce numerical predictions, and the Random Forest combines their outputs to generate the final prediction.

### Why Random Forest?

Random Forest is suitable for this project because:

- It can model complex relationships between input variables.
- It supports regression problems.
- It combines multiple decision trees.
- It can learn nonlinear patterns.
- It provides a practical approach to estimating housing values.

### Model Configuration

According to the original model documentation:

| Parameter | Value |
|---|---:|
| Algorithm | Random Forest Regressor |
| Number of Estimators | 30 |
| Max Features | 8 |
| Training Samples | 16,512 |
| Testing Samples | 4,128 |
| Test Size | 20% |
| Random State | 42 |

The application loads the existing trained model from a Joblib file.



## 7. Data Preprocessing

The project uses a saved Scikit-learn preprocessing pipeline.

The pipeline handles numerical and categorical features before passing them to the model.

### Numerical Features

Eight numerical features are processed using:

**Simple Imputer**

Missing numerical values are replaced using the median value of the corresponding feature.

**Standard Scaler**

Numerical features are standardized using the scaling parameters stored in the fitted pipeline.

### Categorical Features

The `ocean_proximity` feature contains five categories:

- `<1H OCEAN`
- `INLAND`
- `ISLAND`
- `NEAR BAY`
- `NEAR OCEAN`

The pipeline uses One-Hot Encoding to convert these categories into numerical features.

### Final Feature Transformation

The original nine input features are transformed into thirteen model features:

- Eight processed numerical features.
- Five encoded categorical features.

This transformation ensures that the input data has the expected format required by the pre-trained model.



## 8. System Architecture and Workflow

The application follows a simple prediction workflow.

### Prediction Process

1. The user opens the Gradio application.
2. The user enters housing information.
3. The application validates the input values.
4. The input data is converted into a Pandas DataFrame.
5. The saved preprocessing pipeline transforms the input.
6. The processed data is passed to the Random Forest model.
7. The model generates a house price prediction.
8. The application displays the estimated price in USD.

### Workflow Diagram

```text
          User
           |
           v
    Gradio Interface
           |
           v
      Input Values
           |
           v
     Input Validation
           |
           v
  Preprocessing Pipeline
           |
           v
 Random Forest Regressor
           |
           v
   Predicted House Price
           |
           v
     Result Display
```



## 9. Application Features

### Interactive User Interface

The application uses Gradio to provide a browser-based interface.

Users can enter:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

### Prediction Output

After receiving valid inputs, the application displays an estimated house price in US dollars.

### Input Validation

The application checks several conditions before generating a prediction:

- Total bedrooms cannot exceed total rooms.
- Households cannot exceed population.
- Room and population values must be positive.
- Median income must be positive.

If an invalid value is entered, the application displays an error message instead of processing the invalid input.



## 10. Project Structure

The project is organized into separate files to improve readability and maintainability.

```text
house-rent/
│
├── app.py
├── evaluate_model.py
├── test_cases.py
├── requirements.txt
├── README.md
├── MODEL_DOCUMENTATION.md
├── evaluation_results.json
│
├── models/
│   ├── house_price_model.joblib
│   └── preprocessing_pipeline.joblib
│
└── dataset/
    └── CaliforniaHousing/
```

### File Descriptions

| File | Purpose |
|---|---|
| app.py | Main Gradio application and prediction logic |
| evaluate_model.py | Evaluates the pre-trained model |
| test_cases.py | Runs automated validation and prediction tests |
| requirements.txt | Lists Python dependencies |
| README.md | Project overview and instructions |
| MODEL_DOCUMENTATION.md | Technical model documentation |
| evaluation_results.json | Stores evaluation results |
| house_price_model.joblib | Pre-trained Random Forest model |
| preprocessing_pipeline.joblib | Saved preprocessing pipeline |

Large model files and datasets are excluded from Git tracking through `.gitignore`. They must be downloaded separately before running the application.



## 11. Installation and Environment Setup

### Requirements

- Python 3.13
- pip
- Visual Studio Code or another Python IDE
- Internet connection for installing dependencies and downloading model files

### Step 1: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 2: Activate the Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Dependencies

```bash
pip check
```

Expected output:

```text
No broken requirements found.
```


## 12. Downloading the Pre-trained Model

The application requires two model files:

1. `house_price_model.joblib`
2. `preprocessing_pipeline.joblib`

Both files are available from the original Hugging Face repository:

https://huggingface.co/niru-nny/house-price-prediction

Download both files and place them inside the `models` directory.

The directory should look like this:

```text
models/
├── house_price_model.joblib
└── preprocessing_pipeline.joblib
```

The application cannot generate predictions if these files are missing.


## 13. Running the Application

After installing dependencies and placing the required model files in the correct directory, execute:

```bash
python app.py
```

Gradio will start a local web server and display a local URL in the terminal.

Open that URL in a web browser to access the application.

Users can then enter housing information and receive a price prediction.



## 14. Model Evaluation

The project includes a separate evaluation script named `evaluate_model.py`.

The script evaluates the existing pre-trained model using a reconstructed stratified test split.

### Evaluation Configuration

| Parameter | Value |
|---|---:|
| Dataset Records | 20,640 |
| Training Portion | 16,512 |
| Testing Portion | 4,128 |
| Test Size | 20% |
| Random State | 42 |

The split uses stratification based on median income categories.

### Evaluation Metrics

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between actual and predicted house values.

A lower MAE indicates smaller average prediction errors.

**Root Mean Squared Error (RMSE)**

RMSE measures prediction errors while giving greater weight to larger errors.

A lower RMSE generally indicates better prediction performance.

**R-squared (R²)**

R² measures how much of the variation in the target variable is explained by the model on the evaluated dataset.

A value closer to 1 indicates a stronger fit on the evaluated data.

### Evaluation Results

| Metric | Result |
|---|---:|
| MAE | $31,678.01 |
| RMSE | $47,859.54 |
| R² | 0.8243 |

These results come from a reconstructed stratified test split. The original model's exact training indices have not been independently verified.

The R² score should not be interpreted as classification accuracy.

### Run Evaluation

```bash
python evaluate_model.py
```

The results are saved in:

```text
evaluation_results.json
```



## 15. Software Testing

The project includes an automated testing script named `test_cases.py`.

The purpose of this script is to verify the prediction and input validation logic.

### Test Cases

| Test Case | Expected Result |
|---|---|
| Valid input | Prediction generated |
| Bedrooms exceed rooms | Input rejected |
| Households exceed population | Input rejected |
| Zero total rooms | Input rejected |
| Negative median income | Input rejected |
| Different ocean category | Prediction generated |

### Test Execution

```bash
python test_cases.py
```

### Current Test Results

```text
MODEL TEST CASES

PASS: Valid input | $440,803.57
PASS: Bedrooms exceed rooms | Rejected correctly
PASS: Households exceed population | Rejected correctly
PASS: Zero total rooms | Rejected correctly
PASS: Negative median income | Rejected correctly
PASS: Different ocean category | $379,556.80

PASSED: 6/6
ALL TESTS PASSED
```

All six test cases passed successfully.



## 16. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Gradio | Interactive web application |
| Scikit-learn | Machine learning and preprocessing |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Joblib | Loading saved model files |

---

## 17. Limitations

Although the application provides house price estimates, it has several limitations:

1. The model is based on historical California housing data.
2. Predictions are estimates, not official property valuations.
3. Results may not reflect current real estate market prices.
4. The model is designed for California housing data and may not generalize to other regions.
5. Prediction quality depends on the accuracy and relevance of user-provided inputs.
6. The reconstructed evaluation split does not independently verify the original training indices.

---

## 18. Future Improvements

Several improvements could be implemented in future versions:

- Improve the visual design of the application.
- Add interactive charts to explain housing data.
- Display additional information about input locations.
- Add more detailed prediction explanations.
- Compare predictions from different regression algorithms.
- Improve input validation and user guidance.
- Deploy the application online using Hugging Face Spaces.
- Evaluate the model on additional datasets.

---

## 19. Conclusion

This project demonstrates how machine learning can be applied to a real-world regression problem.

By using a pre-trained Random Forest Regressor and a saved preprocessing pipeline, the application estimates California house prices from user-provided housing characteristics.

The project demonstrates practical skills in Python programming, data preprocessing, model evaluation, software testing, dependency management, and interactive application development.

The successful execution of the application and the passing of all six automated test cases demonstrate that the tested prediction and validation workflows are functioning as expected.

Overall, the project provides practical experience in integrating a machine learning model into an interactive software application.

---

## 20. References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. Gradio Documentation: https://www.gradio.app/docs
3. Pandas Documentation: https://pandas.pydata.org/docs/
4. NumPy Documentation: https://numpy.org/doc/
5. Joblib Documentation: https://joblib.readthedocs.io/
6. Hugging Face Model Repository: https://huggingface.co/niru-nny/house-price-prediction

---

**Project:** House Price Prediction Using Machine Learning  
**Document:** README.md
