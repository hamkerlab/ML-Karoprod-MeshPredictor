import numpy as np
import pandas as pd

doe = pd.read_csv('../data/doe.csv')

data = pd.read_csv('../data/cut_x0_all.csv')
data.drop(data[data.doe_id == 1000].index, inplace=True)
data.drop(data[data.doe_id == 247].index, inplace=True)

from cut_predictor import CutPredictor

reg = CutPredictor()

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
    position = 'tp',
    output = ['deviationc'],
    validation_split=0.1,
    validation_method='leaveoneout',
    position_scaler='minmax'
)

reg.save_config("../models/cut_x0.pkl")

config = {
    'batch_size': 2048,
    'max_epochs': 100,
    'layers': [256, 256, 256, 256, 256],
    'dropout': 0.0,
    'learning_rate': 0.001
}


reg.custom_model(save_path='../models/best_x0_model', config=config, verbose=True)

