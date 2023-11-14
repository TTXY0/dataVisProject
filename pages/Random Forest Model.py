import streamlit as st
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import graphviz
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(layout = 'wide', page_title="Random Forest Classifier", page_icon="🌳")
st.sidebar.header("Random Forest Classifier")

import pytz

st.markdown(f"""# Random Forest Model""")
index = st.slider('Select Max_depth of the random forest ', 1, 5, 1)

X, y = make_blobs(centers=[[0, 0], [1, 1]], random_state=61526, n_samples=1000)


def plot_forest(max_depth=1):
    fig = plt.figure()
    ax = plt.gca()
    h = 0.02

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    if max_depth != 0:
        forest = RandomForestClassifier(n_estimators=20, max_depth=max_depth,
                                        random_state=1).fit(X, y)
        Z = forest.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=.4)
        ax.set_title("max_depth = %d" % max_depth)
    else:
        ax.set_title("data set")
    ax.scatter(X[:, 0], X[:, 1], c=np.array(['b', 'r'])[y], s=60)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    return fig

col1, col2 = st.columns([3,1], gap = 'large')
with col1 : 
    st.image(f"/Users/thomaswynn/Desktop/sleep_kaggle/rf_{index}.png", output_format="png", width = 1500)
with col2 : 
    st.pyplot(plot_forest(index))
# def plot_forest_interactive():
#     from ipywidgets import interactive, IntSlider
#     slider = IntSlider(min=0, max=5, step=1, value=0)
#     return interactive(plot_forest, max_depth=slider)
