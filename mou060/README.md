# 🎬 MovieMind --- Intelligent Movie Recommendation System

MovieMind is a full-stack movie recommendation platform that uses
machine learning to help users discover movies based on their
preferences and rating behavior.

The project combines a **PyTorch-based recommendation model**, a
**FastAPI backend**, a **PostgreSQL database**, and a **React +
TypeScript frontend** to provide personalized movie recommendations
through a web application.

The machine-learning component is developed and trained independently
using Jupyter Notebook and Google Colab. The resulting trained model is
exported and integrated into the backend for inference.

------------------------------------------------------------------------

## 📌 Project Overview

MovieMind aims to provide personalized movie recommendations by learning
relationships between users and movies.

  -----------------------------------------------------------------------
  Component               Technology              Responsibility
  ----------------------- ----------------------- -----------------------
  Machine Learning        Python, PyTorch         Model development,
                                                  training and evaluation

  ML Development          Jupyter Notebook,       Experimentation and
                          Google Colab            model training

  Backend                 FastAPI                 REST API and ML model
                                                  inference

  Database                PostgreSQL              Persistent application
                                                  data

  ORM                     Prisma                  Database access and
                                                  management

  Frontend                React, TypeScript       User interface

  Build Tool              Vite                    Frontend development
                                                  and bundling

  Deployment              Docker                  Application
                                                  containerization
  -----------------------------------------------------------------------

## 🧠 Machine Learning --- Jupyter Notebooks

The machine-learning component is one of the core parts of MovieMind.

Rather than training the recommendation model every time a user requests
recommendations, MovieMind follows an **offline training and online
inference approach**.

1.  The model is developed and trained in Jupyter Notebook.
2.  Training and evaluation are performed using Python and PyTorch.
3.  The trained model and its required metadata are exported.
4.  FastAPI loads the exported model.
5.  The backend uses the trained model to generate predictions and
    recommendations.

### Main Training Notebook

**Notebook:** `ml/notebooks/MovieMind_NCF_Training.ipynb`

This notebook contains the main Neural Collaborative Filtering (NCF)
model development and training workflow. It consists of 32 cells
covering the process from dataset preparation to recommendation
demonstration and model export.

### 1. Dataset Preparation

The notebook starts by obtaining and preparing the MovieLens dataset.

The preparation workflow includes:

-   Downloading the MovieLens dataset.
-   Loading movie and rating information.
-   Inspecting the dataset.
-   Preparing user and movie identifiers.
-   Converting rating information into a format suitable for PyTorch.
-   Preparing the data for model training and evaluation.

The dataset provides user--movie interactions that allow the
recommendation model to learn from historical rating behavior.

### 2. Neural Collaborative Filtering (NCF)

The main recommendation algorithm implemented in the notebook is
**Neural Collaborative Filtering (NCF)**.

NCF uses neural networks to learn the relationship between users and
movies instead of relying exclusively on traditional similarity
calculations.

The model uses:

-   **User Embeddings:** Learn a numerical representation for each user.
-   **Movie Embeddings:** Learn a numerical representation for each
    movie.
-   **Neural Layers:** Learn nonlinear relationships between user and
    movie representations.
-   **Rating Prediction:** Predict how a user may rate a particular
    movie.

The model produces predicted ratings on a scale of 0.5 to 5.

Conceptual prediction workflow:

``` text
User ID ────────┐
                ▼
          User Embedding
                │
                ├──────────────┐
                │              │
                │        Neural Network
                │              │
                │              ▼
                │        Rating Prediction
                │              ▲
                │              │
          Movie Embedding      │
                ▲              │
                │              │
Movie ID ───────┘──────────────┘
```

The model learns user and movie representations during training and uses
them to predict ratings for user--movie pairs.

### 3. Model Training

After preparing the data and defining the NCF architecture, the notebook
proceeds to model training.

The training process includes:

-   Preparing the model inputs.
-   Training the NCF model using PyTorch.
-   Learning user and movie representations.
-   Updating model parameters through the training process.
-   Evaluating model performance.

Training is performed in Google Colab, allowing the notebook to be
executed in a cloud-based Python environment.

### 4. Model Evaluation

Model performance is evaluated using two regression metrics:

**Root Mean Squared Error (RMSE)**

RMSE measures the average magnitude of prediction errors while
penalizing larger errors more heavily.

**Mean Absolute Error (MAE)**

MAE measures the average absolute difference between predicted ratings
and actual ratings.

Recorded evaluation results:

  Metric     Test Result
  -------- -------------
  RMSE            0.8738
  MAE             0.6735

These results provide an initial measurement of how accurately the model
predicts user ratings on the test data.

### 5. Recommendation Demonstration

The notebook also includes a recommendation demonstration.

This stage shows how the trained model can be used to predict ratings
and support movie recommendation generation.

The exported model can subsequently be used by the backend without
repeating the training process.

### 6. Model Export and Google Drive Integration

After training and evaluation, the notebook exports the trained model
and supporting files.

The artifacts are saved to Google Drive so that they can be downloaded
and integrated into the FastAPI backend.

The exported files are expected to include:

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `moviemind_ncf.pt`                  Trained PyTorch model checkpoint

  `model_config.json`                 Model configuration

  `user_mapping.json`                 User ID mapping

  `movie_mapping.json`                Movie ID mapping

  `movies.json`                       Movie metadata

  `training_metrics.json`             Training and evaluation metrics

  `model_architecture.py`             Model architecture definition

  `inference_example.py`              Example of loading and using the
                                      trained model

  `README.txt`                        Information about the exported
                                      model files
  -----------------------------------------------------------------------

The mapping files are important because the model uses numerical user
and movie identifiers. The backend must use the same mappings created
during training to ensure that inference uses the correct embedding
indices.

------------------------------------------------------------------------

## 🔌 Model Integration with FastAPI

The trained model is designed to be integrated into the FastAPI backend.

The backend is responsible for loading the exported model and exposing
its functionality through REST API endpoints.

Integration workflow:

``` text
Jupyter Notebook
       │
       ▼
Train NCF Model
       │
       ▼
Evaluate Model
       │
       ▼
Export Model Artifacts
       │
       ▼
FastAPI Backend
       │
       ▼
Load Trained Model
       │
       ▼
Prediction / Recommendation API
       │
       ▼
React Frontend
```

This separation provides several advantages:

-   Model training does not need to happen during user requests.
-   The backend can reuse the trained model.
-   Machine-learning experimentation remains independent of the web
    application.
-   The model can be updated by training and exporting a new checkpoint.

------------------------------------------------------------------------

## 🏗️ Project Structure

The following is the intended logical structure. Directory and file
names should be adjusted to match the actual repository.

``` text
MovieMind/
│
├── ml/
│   ├── notebooks/
│   │   └── MovieMind_NCF_Training.ipynb
│   │
│   ├── artifacts/
│   │   ├── moviemind_ncf.pt
│   │   ├── model_config.json
│   │   ├── user_mapping.json
│   │   ├── movie_mapping.json
│   │   ├── movies.json
│   │   └── training_metrics.json
│   │
│   └── src/
│       ├── model_architecture.py
│       └── inference_example.py
│
├── backend/
│   └── FastAPI application
│
├── frontend/
│   └── React + TypeScript application
│
├── docs/
│   └── Project documentation
│
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

## 🚀 Running the Machine Learning Notebook

The main model notebook is designed to run in Google Colab.

1.  Open Google Colab.
2.  Import `MovieMind_NCF_Training.ipynb`.
3.  Execute the notebook cells in order.
4.  Allow the notebook to download and prepare the dataset.
5.  Run the NCF model training and evaluation.
6.  Review the recommendation demonstration.
7.  Export the trained model and supporting artifacts to Google Drive.

**Notebook location:** `ml/notebooks/MovieMind_NCF_Training.ipynb`

**Environment:**

-   Python
-   PyTorch
-   Jupyter Notebook
-   Google Colab
-   Google Drive

------------------------------------------------------------------------

## 🔮 Future Improvements

Potential future improvements to the machine-learning component include:

-   Experimenting with additional collaborative filtering architectures.
-   Comparing NCF against simpler recommendation baselines.
-   Exploring content-based recommendation using movie metadata.
-   Combining multiple recommendation approaches into a hybrid
    recommendation system.
-   Improving recommendation quality through additional evaluation
    experiments.
-   Updating the model as new user ratings become available.

------------------------------------------------------------------------

## 👨‍💻 Project Development

MovieMind is developed as part of a Professional Training project.

The project emphasizes practical software engineering principles,
modular architecture, machine-learning integration, API development, and
full-stack application development.

**Main objective:** Build an intelligent movie recommendation platform
that connects a trained machine-learning model with a modern web
application.
