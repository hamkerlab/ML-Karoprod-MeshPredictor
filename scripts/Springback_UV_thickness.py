import numpy as np
import pandas as pd

from mesh_predictor import ProjectionPredictor


doe = pd.read_csv('../data/doe.csv')

data = pd.read_csv('../data/springback_uvmap.csv')
data.drop(data[data.doe_id == 1000].index, inplace=True)
data.drop(data[data.doe_id == 247].index, inplace=True)


reg = ProjectionPredictor()

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
    position = ['u', 'v'],
    output = ['thickness', 'epseqpl', 'thinning'],
    validation_split=0.1,
    validation_method='leaveoneout',
    position_scaler='minmax'
)
reg.data_summary()

reg.save_config("../models/springback_uvmap_thickness.pkl")

# best_config = reg.autotune(
#     save_path='models/best_3d_model',
#     trials=100,
#     max_epochs=50, 
#     layers=[4, 6],
#     neurons=[128, 512, 64],
#     dropout=[0.0, 0.0, 0.1],
#     learning_rate=[1e-5, 1e-3]
# )
# print(best_config)

config = {
    'batch_size': 4096*8,
    'max_epochs': 100,
    'layers': [256, 256, 256, 256, 256],
    'dropout': 0.0,
    'learning_rate': 0.001,
    'activation': 'lrelu'
}

reg.custom_model(save_path='../models/best_uv_thickness_model', config=config, verbose=True)
reg.training_summary()

reg.save_h5("../models/springback_uvmap_thickness.h5")