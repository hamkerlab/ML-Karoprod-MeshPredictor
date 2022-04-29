# -*-coding: utf-8-*-
import os
import streamlit as st
import pandas as pd
import numpy as np

modulepath = os.path.dirname(__file__)
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg

_lock = RendererAgg.lock

st.set_page_config(
    page_title="ML@KaroProd - x0cut",
    page_icon="ø",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODULEPATH = os.path.dirname(__file__)

# Initialization
if 'training_data' not in st.session_state:
    st.session_state['training_data'] = None
if 'reg' not in st.session_state:
    st.session_state['reg'] = None
if 'doe' not in st.session_state:
    st.session_state['doe'] = None
if 'x0cut_ref' not in st.session_state:
    st.session_state['x0cut_ref'] = None
if 'plot_hist' not in st.session_state:
    st.session_state['plot_hist'] = []

st.markdown("# x0cut: ground truth prediction")

st.sidebar.markdown("## Options")

if 'training_data' not in st.session_state or st.session_state.training_data is None:
    # Load the data using pandas
    #st.write('training_data' not in st.session_state, st.session_state.training_data is None)
    training_data = pd.read_csv(os.path.join(MODULEPATH, 'models', 'cut_x0.csv'))
    training_data = training_data.head(-1000)  # remove last experiment
    #st.dataframe(training_data.head(2))
    st.session_state.training_data = training_data
else:
    training_data = st.session_state.training_data

if 'doe' not in st.session_state or st.session_state.doe is None:
    # Load the data using pandas
    #st.write('doe' not in st.session_state, st.session_state.doe is None)
    doe = pd.read_csv(os.path.join(MODULEPATH, 'models', 'doe.csv'))
    doe["Niederhalterkraft"] = doe.Niederhalterkraft.astype(float)
    #doe = doe.head(-1000)  # remove last experiment
    #st.dataframe(doe.head(2))
    st.session_state.doe = doe
else:
    doe = st.session_state.doe

if 'doe_id' not in st.session_state:
    st.session_state.doe_id = 0
    # Streamlit will raise an Exception on trying to set the state of button


if 'x0cut_ref' not in st.session_state or st.session_state.x0cut_ref is None:
    # Load the data using pandas
    #st.write('doe' not in st.session_state, st.session_state.doe is None)
    x0cut_ref = pd.read_csv(os.path.join(MODULEPATH, 'models', 'x0cut_ref.csv'))

    #doe = doe.head(-1000)  # remove last experiment
    #st.dataframe(x0cut_ref.head(2))
    st.session_state.x0cut_ref = x0cut_ref
else:
    x0cut_ref = st.session_state.x0cut_ref

# if 'param' not in st.session_state:
#     st.session_state.param = {
#         'Blechdicke':1.1,
#         'Niederhalterkraft':200,
#         'Ziehspalt':2.4,
#         'Einlegeposition': 0,
#         'Ziehtiefe': 50
#     }
valid_doe = pd.DataFrame(training_data["doe_id"].unique(), columns=["doe_id"])
valid_doe["doe_idx"] = valid_doe.index
#st.dataframe(valid_doe)

doe = doe.merge(valid_doe, on="doe_id")
doe = doe.set_index("doe_idx")

#st.dataframe(doe[doe.doe_id.isin(training_data["doe_id"].unique())])
from cut_predictor import CutPredictor

if 'reg' not in st.session_state or st.session_state.reg is None:
    reg = CutPredictor(
        data=training_data,
        process_parameters=[
            'Blechdicke',
            'Niederhalterkraft',
            'Ziehspalt',
            'Einlegeposition',
            'Ziehtiefe'
        ],
        categorical=[
            'Ziehspalt',
            'Einlegeposition',
            'Ziehtiefe'
        ],
        position='tp',
        output='deviationc'
    )
    reg.load(load_path=os.path.join(MODULEPATH, 'models', '20220426-best_x0_model'))
else:
    reg = st.session_state.reg
Fmax = doe.Niederhalterkraft.max()
#st.write(f"F: {Fmax}")


def update_from_doe_idx(*args, **kwargs):
    #design = doe[doe.index == doe_idx]
    design_values = doe[doe.index==args[0]]
    #st.session_state["Blechdicke"] = design_values.Blechdicke.values[0]
    #st.write(f"!!!!{doe_idx}", args, kwargs, st.session_state["Blechdicke"])

doe_idx = st.sidebar.slider('doe_idx', 0, len(doe), step=1, on_change=update_from_doe_idx, args=(st.session_state.doe_id,), key="doe_idx")
design = doe.loc[doe_idx]#[doe.index == doe_idx]
#st.write(design.Blechdicke)
st.markdown(f"## Design {design.doe_id}")
#st.write(reg.data_summary())
st.write(f"output: {reg.output_attribute}")
#st.write(st.session_state["Blechdicke"])

Blechdicke = st.sidebar.slider("Blechdicke t [mm]", min_value=0.1, max_value=5.0, value=float(design.Blechdicke),)
Niederhalterkraft = st.sidebar.slider("Niederhalterkraft F [kN]", min_value=0.1, max_value=doe.Niederhalterkraft.max()*1.5, value=float(design.Niederhalterkraft), step=.1)
Ziehspalt = st.sidebar.selectbox('Ziehspalt zs [mm]', sorted(doe["Ziehspalt"].unique()), index=(sorted(doe["Ziehspalt"].unique()).index(design.Ziehspalt)))
Einlegeposition = st.sidebar.selectbox('Einlegeposition ep [mm]', sorted(doe["Einlegeposition"].unique()), index=(sorted(doe["Einlegeposition"].unique()).index(design.Einlegeposition)))
Ziehtiefe = st.sidebar.selectbox('Ziehtiefe zt [mm]', sorted(doe["Ziehtiefe"].unique()), index=(sorted(doe["Ziehtiefe"].unique()).index(design.Ziehtiefe)))

df_doe = doe[doe.index == doe_idx]

st.write(df_doe)
param = {
        'Blechdicke':Blechdicke,
        'Niederhalterkraft':Niederhalterkraft,
        'Ziehspalt':Ziehspalt,
        'Einlegeposition': Einlegeposition,
        'Ziehtiefe': Ziehtiefe
    }
if st.sidebar.button('Clear plot history'):
    st.session_state['plot_hist'] = []
    st.sidebar.write('plot history is empty')

#
# x, y = reg.predict({
#         'Blechdicke':Blechdicke,
#         'Niederhalterkraft':Niederhalterkraft,
#         'Ziehspalt':Ziehspalt,
#         'Einlegeposition': Einlegeposition,
#         'Ziehtiefe': Ziehtiefe
#     },
#     nb_points=1000)

idx = doe_idx
start, stop = idx*1000, (idx+1)*1000
fig = reg.compare_shape(start, stop, x0cut_ref)
with _lock:
    st.write(fig)

#
# class PlotData:
#     def __init__(self, name, df):
#         self.label = name
#         self.df = df
#         self.color = None
#
#     def __str__(self):
#         return self.label
#
#     def __repr__(self):
#         return self.label
#
#
# df = pd.DataFrame({"x":x.ravel(), "y":y.ravel()})
# label = f"x0cut(t={Blechdicke:.2f}|F={Niederhalterkraft:.1f}|zp={Ziehspalt:.1f}|ep={Einlegeposition:.1f}|z={Ziehtiefe:.0f})"
# plotdata = PlotData(label, df)
#
# with _lock:
#     fig, ax = plt.subplots(1, 1, dpi=90, figsize=(400/25.4,220/25.4))
#     for oldplotdata in st.session_state['plot_hist']:
#         ax.plot(oldplotdata.df.x, oldplotdata.df.y, label=oldplotdata.label, alpha=.7, lw=1.5)#, color=oldplotdata.color)
#
#     p, = ax.plot(plotdata.df.x, plotdata.df.y, label=plotdata.label, alpha=.8, lw=2)
#     color = p.get_color()
#     plotdata.color=color
#
#     ax.set_xlabel('tp')
#     ax.set_ylabel(reg.output_attribute)
#     ax.axhline(0, c="k", lw=0.5)
#     ax.legend(loc="best")
#     st.write(fig)
#     st.session_state['plot_hist'].append(plotdata)
#     st.session_state['plot_hist'] = st.session_state['plot_hist'][-1:]
#
#
#
# st.write(st.session_state['plot_hist'])