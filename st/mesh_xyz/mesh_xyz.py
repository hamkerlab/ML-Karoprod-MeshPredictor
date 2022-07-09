import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 120
import os
import sys
sys.path.insert(0, "../../src")
from cut_predictor import ProjectionPredictor
reg = ProjectionPredictor()
reg.load_config('../../models/springback_uvmap_xyz.pkl')
reg.load_network('../../models/best_uv_xyz_model/')


x0_cut = pd.DataFrame({"u": .5, "v": np.linspace(0., 1. , 100)}).to_numpy()

x, y = reg.predict(process_parameters={
        'Blechdicke': 1.01,
        'Niederhalterkraft': 400.0,
        'Ziehspalt': 2.4,
        'Einlegeposition': -5,
        'Ziehtiefe': 30,
        'Stempel_ID': 3,
        'E': 191.37245,
        'Rp0': 138.22696,
        'Rp50': 449.528189,
    },
    positions=x0_cut
)

def plot_cut(x, y):
    plt.figure()
    plt.plot(y[1, :, :], y[2, :, :])
    plt.xlabel("y")
    plt.ylabel("z")
    plt.title("x0 cut")

plot_cut(x, y)


plt.show()

# import json
#
# from keras.models import model_from_json
#
# with open(model_architecture, 'r') as json_file:
#     architecture = json.load(json_file)
#     model = model_from_json(architecture)
