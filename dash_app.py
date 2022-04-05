# plotly animation visualization with dash
# https://plotly.com/python/animations/
# https://plotly.com/building-machine-learning-web-apps-in-python/]
# https://github.com/plotly/dash-detr
# https://dash.plotly.com/layout
# https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf?_ga=2.199701159.823489852.1645622763-553656816.1642394730
# https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners

import os
# import json
import numpy as np
import pandas as pd
import cv2
# from itertools import combinations
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, html, dcc

app = Dash(__name__)

# set ROOT path
ROOT = os.getcwd() # ROOT dir is 

# read video file size
vid = cv2.VideoCapture(ROOT + "/track_result/exp3/vid_1_Trim.mp4")
vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
vid.release() # release vid memory

# read trajectory csv
df = pd.read_csv(ROOT + "/track_result/exp3/vid_1_Trim_trj.csv")

# animation in vscode
# https://github.com/microsoft/vscode-jupyter/issues/4364#issuecomment-817352686
# pio.renderers.default = 'notebook_connected'

# instant express plot
fig = px.scatter(df.loc[df["frame"].isin([152, 153, 154, 155, 156])],
           x="bb_centroid_x", y="bb_centroid_y_inv",
           animation_frame="frame", animation_group="clsid",
           size=None, color="class", hover_name="clsid", opacity=0.8,
           log_x=False, size_max=55, range_x=[0, vid_width], range_y=[0, vid_height])

# px.scatter(df, x="bb_centroid_x", y="bb_centroid_y_inv", 
#            color="class", title="String 'size' values mean discrete colors")

# go version

# dash app layout
app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

# Offline Saving Method (write_html)
# https://holypython.com/how-to-create-plotly-animations-the-ultimate-guide/