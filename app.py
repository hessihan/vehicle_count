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
pio.renderers.default = 'notebook_connected'

# instant express plot
px.scatter(df.loc[df["frame"].isin([152, 153, 154, 155, 156])],
           x="bb_centroid_x", y="bb_centroid_y_inv",
           animation_frame="frame", animation_group="clsid",
           size=None, color="class", hover_name="clsid", opacity=0.8,
           log_x=False, size_max=55, range_x=[0, vid_width], range_y=[0, vid_height])

px.scatter(df, x="bb_centroid_x", y="bb_centroid_y_inv", 
           color="class", title="String 'size' values mean discrete colors")

# go version
# https://plotly.com/python/animations/
# https://plotly.com/building-machine-learning-web-apps-in-python/]
# https://github.com/plotly/dash-detr
# https://megatenpa.com/python/compare/go-px-buttons-sliders/
# https://community.plotly.com/t/does-dash-support-opencv-video-from-webcam/11012
# https://github.com/plotly/dash-object-detection
# Python YOLOR + DeepSORT + StreamLit Computer Vision Dashboard Tutorial
# https://www.youtube.com/watch?v=mxRH275SyAU
# https://store.augmentedstartups.com/

# DEBUG: smaller df
df = df.loc[df["frame"].isin([152, 153, 154, 155, 156])]

# アニメーション下のスライダー設定
steps = []
for frame in df['frame'].unique():
    slider_step = {
        'args': [
            [frame],  # framesの引数nameと同じ値にする
            {
                'frame': {'duration': 300, 'redraw': False},
                'mode': 'immediate',
                'transition': {
                    'duration': 300
                },
            }
        ],
        'label': f'{frame}',
        'method': 'animate',
    }
    steps.append(slider_step)

sliders = {
    'active': 0,
    'currentvalue': {'prefix': 'sldeir: :'},
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'steps': steps,
}

fig = go.Figure(
    data=[
        go.Scatter(
            x = df[df["class"] == "car"]["bb_centroid_x"],
            y = df[df["class"] == "car"]["bb_centroid_y"],
            mode = "markers",
            name = "car",
            # marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
            text= df[df["class"] == "car"]["clsid"]),
        go.Scatter(
            x = df[df["class"] == "truck"]["bb_centroid_x"],
            y = df[df["class"] == "truck"]["bb_centroid_y"],
            mode = "markers",
            name = "truck",
            # marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
            text= df[df["class"] == "truck"]["clsid"])
    ],
    layout=go.Layout(
        width=660, height=400, #height=360,
        xaxis=dict(range=[0, vid_width], autorange=False, zeroline=False),
        yaxis=dict(range=[0, vid_height], autorange=False, zeroline=False),
        title="Vehicle Trajectories",
        hovermode="closest",
        updatemenus=[
            dict(
                type="buttons",
                buttons=[dict(label="Play",method="animate",args=[None])]
            )
            ],
        sliders=[sliders]
    ),
    frames=[
        go.Frame(
            data=[
                go.Scatter(
                    x = df[(df["class"] == "car") & (df["frame"] == frame)]["bb_centroid_x"],
                    y = df[(df["class"] == "car") & (df["frame"] == frame)]["bb_centroid_y"],
                    mode = "markers",
                    name = "car",
                    # marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df[(df["class"] == "car") & (df["frame"] == frame)]["clsid"]),
                go.Scatter(
                    x = df[(df["class"] == "truck") & (df["frame"] == frame)]["bb_centroid_x"],
                    y = df[(df["class"] == "truck") & (df["frame"] == frame)]["bb_centroid_y"],
                    mode = "markers",
                    name = "truck",
                    # marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df[(df["class"] == "truck") & (df["frame"] == frame)]["clsid"])
            ]
        ) for frame in df['frame'].unique() # is "frame" start from 0 or 1?
    ]
)
fig.show()


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

# # Offline Saving Method (write_html)
# # https://holypython.com/how-to-create-plotly-animations-the-ultimate-guide/