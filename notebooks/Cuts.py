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
    process_parameters = ['Blechdicke', 'Niederhalterkraft', 'Ziehspalt', 'Einlegeposition', 'Ziehtiefe'],
    categorical = ['Ziehspalt', 'Einlegeposition', 'Ziehtiefe'],
    position = 'tp',
    output = 'deviationc'
)
"""
reg = CutPredictor(
    data = data,
    process_parameters = ['Material_ID', 'Ziehtiefe', 'Einlegeposition', 'Niederhalterkraft', 'Stempel_ID'],
    categorical = [],
    position = 'tp',
    output = 'deviationc'
)
"""

# Print a summary of the data
reg.data_summary()

# Start autotuning
reg.autotune(
    trials=100,
    max_epochs=30, 
    layers=[3, 5],
    neurons=[64, 256, 32],
    dropout=[0.0, 0.5, 0.1],
    learning_rate=[1e-5, 1e-3]
)
# or a single network if you know what you are doing:
#reg.custom_model(
#    max_epochs = 30,
#    layers=[128, 128, 128, 128, 128],
#    dropout=0.0,
#    learning_rate=0.005
#)

# Plots for the best model
reg.training_summary()

# Plot single experiment
x, y = reg.predict({'Blechdicke': 1.01, 'Niederhalterkraft':410.0, 'Ziehspalt':2.4, 'Einlegeposition': -5, 'Ziehtiefe': 30}, nb_points=1000)
#x, y = reg.predict({'Material_ID': 3, 'Niederhalterkraft':410.0, 'Stempel_ID': 3, 'Einlegeposition': -5, 'Ziehtiefe': 30}, nb_points=1000)

plt.figure()
plt.plot(x, y)
plt.xlabel('tp')
plt.ylabel('deviationc')
plt.savefig("prediction1.png")

# Compare prediction and ground truth for a single experiment
id = 0
reg.compare(id*1000, (id+1)*1000)
plt.savefig("prediction2.png")


plt.show()