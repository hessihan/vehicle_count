from dash import Dash, dash_table, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import cv2

app = Dash(__name__)
from jupyter_dash import JupyterDash
from dash import Dash, dash_table, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import cv2

# read result table
res_csv_path = "/content/drive/MyDrive/vehicle_count/track_result/exp/vid_1_Trim_trj.csv"
df = pd.read_csv(res_csv_path)
df["unique_row_id"] = df.index # add unique row id

# read result table
res_csv_path = "/content/drive/MyDrive/vehicle_count/track_result/exp/vid_1_Trim_trj.csv"
df = pd.read_csv(res_csv_path)

# read virtual line

# get video height and width
cap = cv2.VideoCapture(vid_path)
vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cap.release()

display_vidframe = False
if display_vidframe:
    # Open and capture the track result mp4
    res_vid_path = "/content/drive/MyDrive/vehicle_count/track_result/exp/vid_1_Trim.mp4"
    cap = cv2.VideoCapture(res_vid_path)

# expand the height of the result cell
from IPython.display import Javascript
display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 1000})'''))

# App ##########################################################################
app = JupyterDash()

# https://zenn.dev/yassh_i/articles/ec5eaa06a83a69
app.layout = html.Div([
    html.H1(children='Track Result Checker'),
    html.Div(
        children=[
            # datatable
            dash_table.DataTable(
                id='result-table',
                columns = [{"name": i, "id": j} for i,j in zip(df, df.columns)],
                data=df.to_dict('records'),
                fixed_rows={'headers': True},
                sort_action='native',
                filter_action = 'native',
                editable=True,
                row_deletable=True,
                # https://dash.plotly.com/datatable/width
                style_table={'overflowX': 'auto'},
                style_cell={
                    'height': 'auto',
                    # all three widths are needed
                    'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                    'whiteSpace': 'normal'
                },
                # page_size = 25,
                # style_table={'height': 400},  # defaults to 500

            ),
            # Debug log
            html.Div(id="debug_log")
        ],
        style={'width': '49%', 'display': 'inline-block'}
    ),
    html.Div(
        [html.Div(dcc.Graph(id="fig"))],
        style={'width': '49%', 'display': 'inline-block'}
    )
])

# interactive datatable
# https://dash.plotly.com/datatable/interactivity
@app.callback(
    # https://community.plotly.com/t/multiple-outputs-in-dash-now-available/19437
    [
     Output("debug_log", "children"),
     Output('fig', 'figure'),
    ],
    [
     Input('result-table', 'derived_virtual_row_ids'),
     Input('result-table', 'selected_row_ids'),
     Input('result-table', 'active_cell')
    ]
)
def update_videoframe(row_ids, selected_row_ids, active_cell):    
    selected_id_set = set(selected_row_ids or [])

    if row_ids is None:
        dff = df
        # pandas Series works enough like a list for this to be OK
        row_ids = df['unique_row_id']
    else:
        dff = df.loc[row_ids]

    active_row_id = active_cell['row_id'] if active_cell else None
    
    selected_row = dff[dff["unique_row_id"] == active_row_id]
    selected_frame, bb_left, bb_top, bb_width, bb_height, = selected_row["frame"].values[0], selected_row["bb_left"].values[0], selected_row["bb_top"].values[0], selected_row["bb_width"].values[0], selected_row["bb_height"].values[0]
    
    # debug info
    debug_log = [
        html.Div(f'active_cell: {active_cell}'),
        html.Div(f'active_row_id: {active_row_id}'),
        html.Div(f'selected_row: {selected_row}'),
        html.Div(f'selected_frame: {selected_frame}'),
        # html.Div(f'dff: {dff}'),
        # html.Div(f'derived_virtual_row_ids: {row_ids}'),
        # html.Div(f'selected_row_ids: {selected_row_ids}')
    ]

    # frame figure
    fig = go.Figure()
    # whole vid frame
    fig.add_trace(go.Scatter(
        x=[0, 0, vid_width, vid_width], 
        y=[0, vid_height, 0, vid_height], 
        mode='markers', marker=dict(color='black', size=5),
        name="frame outline"
        ))
    # selected vehicle window
    fig.add_trace(go.Scatter(
        x=[bb_left, bb_left, bb_left+bb_width, bb_left+bb_width], 
        y=[bb_top, bb_top+bb_height, bb_top, bb_top+bb_height], 
        mode='markers', marker=dict(color='red', size=16),
        name="selected object"
        ))
    fig.update_layout(
        title=f"Selected Frame: {selected_frame}",
        # width=500,
        # height=500,
        margin=dict(l=40, r=40, b=40, t=40),        
    )    

    if display_vidframe:
        # background image
        # https://plotly.com/python/images/
        # https://plotly.com/python/imshow/
        # set current frame to selected_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame)
        print("read frame", cap.get(cv2.CAP_PROP_POS_FRAMES))
        # get current frame as np.ndarray
        ret, frame = cap.read()
        print("current frame", cap.get(cv2.CAP_PROP_POS_FRAMES))
        # converted to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig.add_trace(go.Image(z=frame))

    return debug_log, fig

if __name__ == '__main__':
    app.run_server(debug=True)