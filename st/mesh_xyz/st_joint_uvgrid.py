# -*-coding: utf-8-*-
import os
import streamlit as st
import pandas as pd
import numpy as np

import os
import sys
modulepath = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(modulepath, "../../src"))
from cut_predictor import DoubleProjectionPredictor
from banana_lib import UVJoint, UV2d

import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

st.set_page_config(
    page_title="ML@KaroProd - joint parts prediction",
    page_icon="ø",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("# joint parts prediction")

st.sidebar.markdown("## Options")

modelpath = os.path.join(modulepath, "../../models/")
uvgrids_path = os.path.join(modulepath, "../../uvgrids/")

uvgrids = sorted(["{:.4f}".format(float(x.replace(".h5", "").replace("uv_", ""))) for x in os.listdir(uvgrids_path) if x.startswith("uv_")])

reg_xyz = DoubleProjectionPredictor.from_h5('../../models/joining_xyz.h5')
reg_results = DoubleProjectionPredictor.from_h5('../../models/joining_thickness.h5')

st_results_name = st.sidebar.selectbox("result",
                                       [x for x in reg_results.output_attributes if x not in ["u", "v", "x", "y", "z"]],
                                       index=0)

param_clamp = {
    # Joining
    'Spanner_1': [-3, -5, 5],
    'Spanner_2': [-1, -5, 5],
    'Spanner_3': [3, -5, 5],
    'Spanner_4': [-2, -5, 5],
}

param_top = {

    # Top
    'Oberblech_MID': [5, 1, 6],
    'Blechdicke_top': [1.48, .99, 1.48],
    'Niederhalterkraft_top': [400, 10, 500],
    'Ziehspalt_top': [2.4, [1.6, 2.4]],
    'Einlegeposition_top': [-4, -5, 5],
    'Ziehtiefe_top': [50, [50, 70]],
    'Rp0_top': [235.160326, 133.18263199999998, 296.5565],
}
param_bot = {
    # Bottom
    'Unterblech_MID': [3, 1, 6],
    'Blechdicke_bot': [1.01, .99, 1.48],
    'Niederhalterkraft_bot': [400, 10, 500],
    'Ziehspalt_bot': [2.4, [1.6, 2.4]],
    'Einlegeposition_bot': [-4, -5, 5],
    'Ziehtiefe_bot': [30, [30]],
    'Rp0_bot': [235.160326, 133.18263199999998, 296.5565],
}

st_range = {}

for prozess_parameter_name in param_clamp.keys():
    print(prozess_parameter_name)
    value, min_value, max_value = param_clamp[prozess_parameter_name]
    st_range[prozess_parameter_name] = st.sidebar.slider(prozess_parameter_name,
                                                         min_value=min_value,
                                                         max_value=max_value,
                                                         value=value,
                                                         )
with st.expander("sheets attributes", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## top")
        for prozess_parameter_name in param_top.keys():
            values = param_top[prozess_parameter_name]
            if isinstance(values, list) and len(values)==3:
                value, min_value, max_value = values
                st_range[prozess_parameter_name] = st.slider(prozess_parameter_name,
                                                                     min_value=min_value,
                                                                     max_value=max_value,
                                                                     value=value,
                                                                     )
            elif isinstance(values, list) and len(values)==2:
                value, valuelist = values
                st_range[prozess_parameter_name] = st.selectbox(prozess_parameter_name, valuelist, index=0)
    with col2:
        st.markdown("## bot")
        for prozess_parameter_name in param_bot.keys():
            values = param_bot[prozess_parameter_name]
            if isinstance(values, list) and len(values)==3:
                value, min_value, max_value = values
                st_range[prozess_parameter_name] = st.slider(prozess_parameter_name,
                                                                     min_value=min_value,
                                                                     max_value=max_value,
                                                                     value=value,
                                                                     )
            elif isinstance(values, list) and len(values)==2:
                value, valuelist = values
                st_range[prozess_parameter_name] = st.selectbox(prozess_parameter_name, valuelist, index=0)


st_param = {}
for pname, pvalue in st_range.items():
    st_param[pname] = pvalue


st_elsize = st.sidebar.selectbox("grid resolution", uvgrids,
                                            index=uvgrids.index("0.0200"))


#st.write(st_param)
elsize = float(st_elsize)
filepath = f"../../uvgrids/uv_{elsize}.h5"
uv = UV2d.from_h5(filepath)

uvj = UVJoint(uv, reg_xyz=reg_xyz, reg_results=reg_results)
uvj.predict(st_param)
fig = uvj.plot_3d(result_name="thickness", figsize=(18,10))

result_name = st_results_name

with _lock:
    st.write(fig)
