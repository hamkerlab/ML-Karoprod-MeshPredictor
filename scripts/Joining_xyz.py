import numpy as np
import pandas as pd

doe_single = pd.read_csv('../data/doe.csv')
doe_joining = pd.read_csv('../data/doe_joining.csv')

data = pd.read_csv('../data/joining.csv')


from cut_predictor import DoubleProjectionPredictor

reg = DoubleProjectionPredictor()

reg.load_data(
    doe_joining=doe_joining, 
    doe_single=doe_single, 
    data=data, 
    process_parameters_joining = [
        'Spanner_1',
        'Spanner_2',
        'Spanner_3',
        'Spanner_4',
        'Oberblech_MID',
        'Unterblech_MID',
    ], 
    process_parameters_single = [
        'Blechdicke', 
        'Niederhalterkraft', 
        'Ziehspalt', 
        'Einlegeposition', 
        'Ziehtiefe',
        'Rp0',
    ], 
    position = ['u', 'v'], 
    output = ['x', 'y', 'z'], 
    categorical_joining=[
    ], 
    categorical_single= [
        'Ziehspalt', 
        'Ziehtiefe',
    ], 
    index_joining='doe_id', 
    index_single='doe_id', 
    part_index='pos', 
    top_bottom=['Oberblech_ID', 'Unterblech_ID'],
    validation_split=0.1, 
    validation_method="random",
    # position_scaler="minmax",

)
reg.save_config("../models/joining_xyz.pkl")

config = {
    'batch_size': 2048*16,
    'max_epochs': 100,
    'layers': [512, 512, 512, 256, 256],
    'dropout': 0.0,
    'learning_rate': 0.001,
    'activation': 'lrelu'
}

reg.custom_model(save_path='../models/best_joining_xyz_model', config=config, verbose=True)
reg.training_summary()

#reg.save_h5('../models/joining_xyz.h5')
