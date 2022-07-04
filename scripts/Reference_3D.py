import numpy as np
import pandas as pd

from cut_predictor import MeshPredictor

doe = pd.read_csv('../data/doe.csv')


data = pd.read_csv('../data/zt_all_raw.csv')
data.drop(data[data.doe_id == 1000].index, inplace=True)
data.drop(data[data.doe_id == 247].index, inplace=True)


reg = MeshPredictor()

reg.load_data(
    doe = doe,
    data = data,
    index='doe_id',
    process_parameters = [
        'Blechdicke', 
        'Niederhalterkraft', 
        'Ziehspalt', 
        'Einlegeposition', 
        'Ziehtiefe',
        'Stempel_ID',
    ],
    categorical = [
        'Ziehspalt', 
        'Ziehtiefe',
        'Stempel_ID',
    ],
    position = ['x', 'y', 'z'],
    output = ['deviation', 'thickness']
)

best_config = reg.autotune(
    save_path='../models/best_3d_model',
    trials=100,
    max_epochs=50, 
    layers=[4, 6],
    neurons=[64, 256, 64],
    dropout=[0.0, 0.5, 0.1],
    learning_rate=[1e-5, 1e-3]
)
print(best_config)