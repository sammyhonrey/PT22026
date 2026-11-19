# Weather Prediction System

An interactive machine-learning application that estimates whether it
will rain tomorrow (`RainTomorrow`) from observed weather conditions.
The project uses **Python, PyTorch, and Streamlit**.

> **Important:** This application provides an educational
> machine-learning estimate. It is not an official weather forecast and
> should not be used as the sole basis for safety-critical decisions.

## Features

-   Interactive web interface built with Streamlit.
-   Prediction of `RainTomorrow` (`Yes` / `No`).
-   Estimated probability of rain and the decision threshold used by the
    model.
-   Inputs for temperature, rainfall, evaporation, sunshine, wind,
    humidity, pressure, cloud cover, location, wind directions,
    `RainToday`, and month.
-   Preprocessing based on statistics and category lists saved during
    training.
-   PyTorch model loaded from a `.pt` state-dictionary file.

## Technology stack

-   Python
-   PyTorch
-   Streamlit
-   pandas
-   NumPy

## Project structure

The expected local structure is:

``` text
ald077/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── weather_prediction_model.pt
    ├── model_metadata.json
    └── model_evaluation_results.json
```

The model artifacts must be from the **same training run**. In
particular, `model_metadata.json` is required because it contains the
feature order and preprocessing information used by the application.

## Dataset

The model was developed using the **Rain in Australia / WeatherAUS**
dataset.

Dataset page:
<https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package>

The prediction target is `RainTomorrow`. The model uses these 22 input
features:

**Numerical features**

-   `MinTemp`
-   `MaxTemp`
-   `Rainfall`
-   `Evaporation`
-   `Sunshine`
-   `WindGustSpeed`
-   `WindSpeed9am`
-   `WindSpeed3pm`
-   `Humidity9am`
-   `Humidity3pm`
-   `Pressure9am`
-   `Pressure3pm`
-   `Cloud9am`
-   `Cloud3pm`
-   `Temp9am`
-   `Temp3pm`

**Categorical features**

-   `Location`
-   `WindGustDir`
-   `WindDir9am`
-   `WindDir3pm`
-   `RainToday`
-   `Month`

## Model and preprocessing

The application loads a PyTorch feed-forward neural network. Its
architecture must match the architecture used to create
`weather_prediction_model.pt`.

Preprocessing is reproduced in the application using the saved metadata:

-   Numerical missing values are filled using training-set medians.
-   Numerical values are standardized using training-set means and
    standard deviations.
-   Categorical missing values are filled using training-set modes.
-   Categorical values are one-hot encoded using the categories saved
    from training.
-   Unseen categorical values are represented by an all-zero one-hot
    vector, matching `OneHotEncoder(handle_unknown="ignore")`.

The saved validation decision threshold is applied to the model's
estimated rain probability to produce the final `Yes` / `No`
classification.

**Do not edit the model architecture, feature order, preprocessing
values, category lists, or threshold independently of the saved model
artifacts.** Doing so can cause incorrect predictions or loading errors.

## Installation

### 1. Clone the repository

Replace the URL below with the repository URL if you are using a
different repository:

``` bash
git clone https://github.com/xahiru/PT22026.git
cd PT22026
```

If your project is inside a student-specific subfolder, change into that
folder before continuing. For example:

``` powershell
cd ald077
```

### 2. Create and activate a virtual environment (recommended)

On Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the application with the
environment's Python executable directly, or use Command Prompt:

``` cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

If `requirements.txt` is not yet present, create it with:

``` text
streamlit
pandas
numpy
torch
```

PyTorch installation availability depends on the Python version and
operating system. If installation fails, use the official PyTorch
installation selector to choose a compatible command:
<https://pytorch.org/get-started/locally/>.

## Model files

The application expects these files inside the `models/` directory:

-   `weather_prediction_model.pt`
-   `model_metadata.json`

`model_evaluation_results.json` is included for recording or displaying
evaluation results, if supported by the current application.

The trained `.pt` file and metadata are generated during model training.
They are not recreated by running `app.py`.

### If model files are not included in GitHub

Large or generated model artifacts may be distributed separately.
Download the matching files from the project maintainer's provided
storage location and place them in `models/`.

**Model download location:** `ADD_YOUR_MODEL_DOWNLOAD_LINK_HERE`

Do not use a model file from a different training run with this
metadata. The weights and metadata must match.

## Run the application

From the directory containing `app.py`, run:

``` bash
python -m streamlit run app.py
```

Then open the local address shown in the terminal. By default, Streamlit
commonly uses:

<http://localhost:8501>

To stop the server, press `Ctrl + C` in the terminal.

## How to use the application

1.  Start the Streamlit application.
2.  Enter the available weather observations in the input form.
3.  Select the categorical weather conditions, such as location, wind
    direction, and whether it rained today.
4.  Submit the form using the prediction button.
5.  Review the predicted class (`Yes` or `No`), estimated probability of
    rain, and decision threshold.
6.  Expand the submitted-input section to review the values used for the
    prediction.

The model's probability is an estimate produced by the trained neural
network. It is not a guarantee of tomorrow's weather.

## Evaluation

Model evaluation results are saved separately in:

``` text
models/model_evaluation_results.json
```

Use the actual results from that file when reporting accuracy,
precision, recall, F1-score, or ROC-AUC. Do not insert estimated or
invented performance figures into the README or project report.

## Git and repository notes

Before pushing the project:

-   Include source code, `README.md`, and `requirements.txt`.
-   Include `model_metadata.json` if the application needs it and your
    course/repository rules permit it.
-   Follow your training/project submission instructions for
    distributing `.pt` model weights. If the weights are stored
    externally, add the real download link above.
-   Do not commit the original dataset or other large data files unless
    explicitly required.
-   Do not commit virtual environments, Python cache files, secrets, or
    local configuration.

A useful `.gitignore` is:

``` gitignore
.venv/
venv/
__pycache__/
*.py[cod]
.streamlit/secrets.toml
.DS_Store

# Dataset files
*.csv
*.zip

# Model weights (if distributed separately)
models/*.pt
```

If you decide to store the `.pt` file in the repository instead, remove
the `models/*.pt` ignore rule and confirm that this is permitted by your
project requirements.

## Limitations

-   Predictions depend on the quality, coverage, and patterns of the
    training dataset.
-   Weather relationships can change across locations and time.
-   The application only accepts the features used during training.
-   Missing or out-of-range user-entered values may affect prediction
    quality.
-   The model's probability is not necessarily a calibrated real-world
    probability.
-   This project is for learning and demonstration, not operational
    meteorological use.

## Future improvements

-   Add a historical weather-data upload and batch prediction feature.
-   Add model evaluation visualizations using the saved evaluation
    results.
-   Improve input validation and provide sensible measurement ranges.
-   Add automated tests for preprocessing and inference.
-   Deploy the application to a suitable hosting platform.

## Acknowledgements

-   WeatherAUS / Rain in Australia dataset:
    <https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package>
-   PyTorch: <https://pytorch.org/>
-   Streamlit: <https://streamlit.io/>

## Author

**Student:** ADD_YOUR_NAME\
**Course / Training:** ADD_YOUR_COURSE_NAME\
**GitHub repository:** <https://github.com/xahiru/PT22026>
