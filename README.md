# mlfs-book
O'Reilly book - Building Machine Learning Systems with a feature store: batch, real-time, and LLMs


## ML System Examples


[Dashboards for Example ML Systems](https://iammarcol.github.io/mlfs-book/)

## Course Comparison

| Course                         | MLOps | LLLMs             | Feature/Training/Inference | Working AI Systems | Focus |
|--------------------------------|-------|----------------------------|--------------------|------------------|
| Building AI Systems (O'Reilly) | Yes   | Fine-Tuning & RAG | Yes                        | High               | Project-based, Software Engineering, Fundamentals    |
| [Made With ML](https://madewithml.com/)                   | No          | Yes   | No                         | No                 | Software Engineering, Model Training   |
| [7 Steps MLOps](https://www.pauliusztin.me/courses/the-full-stack-7-steps-mlops-framework)            | Yes   | Separate Course    | Yes                        | Low                | Learning Tools and Project    |

# Air Quality Prediction Service

This repository contains the code and workflows for an air quality prediction service. The service leverages advanced machine learning models to predict air quality based on historical data and a robust feature engineering pipeline.

## Project Structure

The project consists of four main Jupyter notebooks, each handling a key stage of the service:

1. **Feature Backfill** (`1_air_quality_feature_backfill.ipynb`):  
   Implements a backfilling process for missing or incomplete features in the air quality dataset. This ensures the data is clean and ready for further processing.

2. **Feature Engineering Pipeline** (`2_air_quality_feature_pipeline.ipynb`):  
   Automates the creation and transformation of features from raw data. This pipeline integrates preprocessing, scaling, and encoding techniques to generate inputs suitable for model training.

3. **Model Training Pipeline** (`3_air_quality_training_pipeline.ipynb`):  
   Handles the training of machine learning models using the prepared features. Includes hyperparameter tuning, model evaluation, and saving the best-performing models.

4. **Batch Inference** (`4_air_quality_batch_inference.ipynb`):  
   Applies the trained model to new data in batch mode, producing air quality predictions for deployment or further analysis.

---

## Features

| **Feature Name**       | **Description**                                              | **Source**               |
|-------------------------|--------------------------------------------------------------|--------------------------|
| `date`                 | Timestamp for the measurement.                               | Sensor Data             |
| `pm25`                 | Concentration of PM2.5 pollutants in the air.               | Sensor Data             |
| `temperature_2m_mean`              | Average temperature over the day.                        | Metadata                |
| `precipitation_sum`                 | The total precipitation over the day.                           | Metadata                |
| `wind_speed_10m_max	`               | Average wind speed over the day.                        | Metadata                |
| `wind_direction_10m_dominant`                  | The most dominant wind direction over the day.     | Metadata                |
| `city`                  | City where the sensor is located.     | Metadata                |


### Input and Output Schemas

The **model schemas** used for training and prediction pipelines include the following:

- **Input Schema**: Contains the engineered features used for model training (e.g., scaled or encoded versions of the above features).
- **Output Schema**: Represents the target variable (e.g., predicted PM2.5 levels).

---

## Models

The air quality prediction service employs the following model type:
- **Gradient Boosting Models** (e.g., XGBoost, LightGBM):  
  Optimized for tabular data with high interpretability and performance.

---

## Getting Started

1. **Clone the Repository**:  
   ```bash
   git clone <repository-url>
   cd air-quality-prediction-service

2. **Set Up Environment**:  
   ```bash
   pip install -r requirements.txt

3. **Run Notebooks**:  
   Open and execute the notebooks in the following order:

    1. [1_air_quality_feature_backfill.ipynb](1_air_quality_feature_backfill.ipynb)
    2. [2_air_quality_feature_pipeline.ipynb](2_air_quality_feature_pipeline.ipynb)
    3. [3_air_quality_training_pipeline.ipynb](3_air_quality_training_pipeline.ipynb)
    4. [4_air_quality_batch_inference.ipynb](4_air_quality_batch_inference.ipynb)

4. **Predict Air Quality**:  
   Use the batch inference notebook to generate predictions on new data.


## Requirements

- Python 3.8+
- Jupyter Notebook
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - XGBoost
  - LightGBM
  - others (see `requirements.txt` for the full list)
