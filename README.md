# Federated Learning for Early Heart Disease Prediction

This repository presents a privacy-preserving machine learning pipeline to detect heart disease using Federated Learning (FL) with Multilayer Perceptron models. Designed for the healthcare context, it showcases a simulation where medical data remains distributed across several sites (clients), and collaborative model training is realized via the Federated Averaging algorithm. The global model is compared against a traditional centralized baseline, using the same architecture and data.

## Key Features

- **End-to-end Federated Learning simulation** with MLP neural networks
- Model aggregation with Federated Averaging
- Comprehensive benchmarking versus a centralized (non-FL) MLP model
- Robust evaluation: Accuracy, Precision, Recall, F1-score, ROC-AUC, AUPRC, and Log Loss metrics
- Plots for: convergence (metrics over communication rounds), confusion matrix, ROC and PR curves, and side-by-side FL vs. centralized performance comparison


---


## Models & Algorithms Used

- **Multilayer Perceptron (MLP)**: A fully-connected neural network with two hidden layers, trained both in federated and centralized modes.
- **Federated Averaging (FedAvg)**: Orchestrates global model updates by averaging participating client weights after local training.
- **Data Preprocessing**:
  - **StandardScaler**: Normalizes all numerical features.
  - **OneHotEncoder**: Encodes categorical attributes.
  - **SMOTE**: Handles class imbalance in the training set by synthesizing minority class samples.
- **Metrics & Visualizations**:
  - Model accuracy, precision, recall, F1-score, ROC-AUC, AUPRC, log loss at each round
  - Final evaluation includes confusion matrix, ROC and precision-recall curves
  - Bar charts comparing best FL and centralized model scores


---


## Dataset

- Publicly available heart disease dataset from Kaggle: `"fedesoriano/heart-failure-prediction"`.
- Label: `HeartDisease` (0 - absence, 1 - presence).
- Mixture of numerical and categorical features; prevalence of heart disease noted in the class distribution.


2. **Configure Kaggle API credentials** for auto-download of the dataset.
3. **Run the simulation script**:
- Models will be trained, evaluated, and all results and plots will be saved in the appropriate folders (`fl_plots/`).


---


## Authors
- Aniketh Gandhari
- Siddha Sankalp Topalle
- Vivin Chandrra Paasam
