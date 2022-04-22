import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cut_predictor import CutPredictor

# Load the data using pandas
data = pd.read_csv('../data/cut_x0.csv')
data = data.head(-1000) # remove last experiment

# Create the predictor
reg = CutPredictor(
    data = data,
    process_parameters = [
        'Blechdicke', 
        'Niederhalterkraft', 
        'Ziehspalt', 
        'Einlegeposition', 
        'Ziehtiefe'
    ],
    categorical = [
        'Ziehspalt', 
        'Einlegeposition', 
        'Ziehtiefe'
    ],
    position = 'tp',
    output = 'deviationc'
)

# Print a summary of the data
reg.data_summary()

# Start autotuning
best_config = reg.autotune(
    save_path='best_model',
    trials=100,
    max_epochs=50, 
    layers=[3, 5],
    neurons=[64, 256, 64],
    dropout=[0.0, 0.5, 0.1],
    learning_rate=[1e-5, 1e-3]
)

# Plots for the best model
reg.training_summary()
plt.show()