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
        'Rp0',
    ],
    categorical = [
        'Ziehspalt', 
        'Ziehtiefe',
    ],
    position = ['x', 'y', 'z'],
    output = ['deviation', 'thickness'],
    validation_split=0.1,
    validation_method='leaveoneout',
    position_scaler='minmax'
)

reg.save_config("../models/3d.pkl")

config = {
    'batch_size': 2048*16,
    'max_epochs': 20,
    'layers': [256, 128, 256, 256, 256],
    'dropout': 0.0,
    'learning_rate': 0.001,
    'activation': 'lrelu'
}

reg.custom_model(save_path='../models/best_3d_model', config=config, verbose=True)
reg.training_summary()