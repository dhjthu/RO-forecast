# RO-forecast: Time-Series Prediction of RO System Performance under ZLD Conditions

This repository contains the full source code used in the manuscript titled:

A Multi-Model Ensemble for Advanced Prediction of Reverse Osmosis Performance in Full-Scale Zero-Liquid Discharge Systems

We develop and evaluate multiple deep learning models including LSTM, GRU, RNN, CNN, and ConvLSTM to forecast the performance of reverse osmosis systems under various zero liquid discharge conditions. The repository supports training, evaluation, and replication of all reported experiments.

-Project Structure

Main.py
 The main script for data preprocessing, training, and evaluation.
 
Models.py
 Contains architecture definitions for CNN, RNN, GRU, LSTM, and ConvLSTM.
 
Tools.py
 Includes utility functions for data loading, normalization, seeding, and model comparison.

-Environment Requirements

Python ≥ 3.8
Dependencies:

numpy

pandas

matplotlib

scikit-learn

torch

tqdm

openpyxl (for reading Excel data)

To install all packages:

pip install -r requirements.txt

-How to Use

Prepare your input Excel file (e.g., Data-KDE-f1.xlsx), formatted as:
 [Date, Feature1, Feature2, ..., FeatureN, TargetFlux]

Run training and testing:

python Main.py

The script will train all five models and export the predicted flux values to model_predictions_train_test_split.xlsx.

-Data Access

The raw dataset used in this study is not included in this repository but is available upon reasonable request from the corresponding author.

-Contact

For questions or dataset access, please contact the corresponding authors.

