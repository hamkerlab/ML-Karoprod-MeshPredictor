# -*-coding: utf-8-*-
import os
import streamlit as st
import pandas as pd
import numpy as np

import os
import sys
modulepath = os.path.dirname(__file__)

sys.path.insert(0, os.path.join(modulepath, "../../src"))
from cut_predictor import ProjectionPredictor

reg = ProjectionPredictor()


import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

st.set_page_config(
    page_title="ML@KaroProd - springback prediction",
    page_icon="ø",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("# springback prediction")

st.sidebar.markdown("## Options")

modelpath = os.path.join(modulepath, "../../models/")

#st.write(os.listdir(modelpath))


regeps = ProjectionPredictor.from_h5(os.path.join(modulepath, "../../models/springback_uv_thickness.h5"))
regxyz = ProjectionPredictor.from_h5(os.path.join(modulepath, "../../models/springback_uv_xyz.h5"))

param = {
    'Blechdicke': 1.01,
    'Niederhalterkraft': 110.0,
    'Ziehspalt': 2.4,
    'Einlegeposition': -5,
    'Ziehtiefe': 30,
    'Stempel_ID': 3,
    'E': 191.37245,
    'Rp0': 238.22696,
    'Rp50': 449.528189,
}

st_range = {}
for prozess_parameter_name in regxyz.process_parameters:
    if prozess_parameter_name not in regxyz.categorical_attributes:
        # print(prozess_parameter_name, regxyz.min_values.get(prozess_parameter_name), regxyz.max_values.get(prozess_parameter_name))
        st_range[prozess_parameter_name] = st.sidebar.slider(prozess_parameter_name,
                                                             min_value=float(
                                                                 regxyz.min_values.get(prozess_parameter_name)),
                                                             max_value=float(
                                                                 regxyz.max_values.get(prozess_parameter_name)),
                                                             value=float(param[prozess_parameter_name]),
                                                             )
st_cat = {}
for cat_name, cat_values in regxyz.categorical_values.items():
    st_cat[cat_name] = st.sidebar.selectbox(cat_name, sorted(cat_values),
                                            index=(sorted(cat_values).index(param[cat_name])))

st_param = {}
for pname, pvalue in st_cat.items():
    st_param[pname] = pvalue

for pname, pvalue in st_range.items():
    st_param[pname] = pvalue

st_u = st.sidebar.slider("u",
                         min_value=0.,
                         max_value=1.,
                         value=.5,
                         step=0.05,
                         )
st_n = st.sidebar.slider("no of cut points",
                         min_value=100,
                         max_value=10000,
                         value=10000,
                         step=10,
                         )
# st.write(st_param)
df = pd.DataFrame({"v": np.linspace(0., 1, st_n), "u": st_u})
dfr = regxyz.predict(st_param, df)
dfeps = regeps.predict(st_param, dfr)

st_results_name = st.sidebar.selectbox("result",
                                       [x for x in dfeps.columns if x not in ["u", "v", "x", "y", "z"]],
                                       index=0)

B = 400 / 25.4
H = 200 / 25.4
fig, ax = plt.subplots(1, figsize=(B, H))

#ax.scatter(dfr.y, dfr.z, c=np.linspace(0,1,len(dfr)))
ax.scatter(dfr.y, dfr.z, c=dfeps[st_results_name]/dfeps[st_results_name].max())

ax.plot(dfr.y, dfr.z)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_ylim(-10, 80)
with _lock:
    st.write(fig)

B = 400 / 25.4
H = 200 / 25.4
results_name = st_results_name
figr, ax = plt.subplots(1, figsize=(B, H))
ax.plot(dfeps.v, dfeps[results_name])
ax.scatter(dfr.v, dfeps[results_name], c=dfeps[st_results_name]/dfeps[st_results_name].max())

ax.set_xlabel("v")
ax.set_ylabel(results_name)
ax.set_ylim(0, dfeps[results_name].max()*1.1)
if results_name=="thickness":
    ax.axhline(st_range["Blechdicke"], ls="--", lw=1, alpha=.7, c="k")
with _lock:
    st.write(figr)

dfs = []
for u in np.linspace(0, 1, 21):
    df = pd.DataFrame({"u": u, "v": np.linspace(0, 1, 100)})
    dfs.append(df)
    dfs.append(pd.DataFrame({"u": [np.nan], "v": [np.nan]}))
for v in np.linspace(0, 1, 41):
    df = pd.DataFrame({"v": v, "u": np.linspace(0, 1, 100)})
    dfs.append(df)
    dfs.append(pd.DataFrame({"u": [np.nan], "v": [np.nan]}))
dfgrid = pd.concat(dfs, ignore_index=True)
dfgrid = regxyz.predict(st_param, dfgrid)

nn = 100
df_top = pd.DataFrame({"u": np.linspace(0, 1, nn), "v": 1})
df_right = pd.DataFrame({"u": 1., "v": np.linspace(1, 0, nn)})
df_bot = pd.DataFrame({"u": np.linspace(1, 0, nn), "v": 0})
df_left = pd.DataFrame({"u": 0., "v": np.linspace(0, 1, nn)})
df_contour = pd.concat([df_top, df_right, df_bot, df_left], ignore_index=True)
df_contour = regxyz.predict(st_param, df_contour)

figc3d = plt.figure("3d", figsize=(18, 10))
ax = figc3d.add_subplot(111, projection="3d")
ax.plot(df_contour.x, df_contour.y, df_contour.z, c="k", alpha=.5)
ax.plot(dfgrid.x, dfgrid.y, dfgrid.z, c="k", alpha=.5, lw=.5)
ax.plot(dfr.x, dfr.y, dfr.z, c="r", alpha=.7, lw=2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_zlim(-10, 80)
with _lock:
    st.sidebar.write(figc3d)

# figc, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))
# ax.plot(df_contour.x, df_contour.y, c="k", alpha=.5)
# ax.plot(dfr.x, dfr.y, c="r", alpha=.7, lw=2)
# with _lock:
#     st.write(figc)
