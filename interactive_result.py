from dash import Dash, dash_table, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import cv2

app = Dash(__name__)
res_txt_path = "track_result/exp6/vid_1_Trim.txt"
df = pd.read_table(res_txt_path, header=None, delimiter=" ")

df = df.drop([8, 9, 10], axis=1)
df.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "class", "conf"]
class_name = {0:"person", 1:"bicycle", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}
df = df.replace({"class": class_name})

# https://zenn.dev/yassh_i/articles/ec5eaa06a83a69
app.layout = html.Div([
    html.H1(children='Track Result Checker'),
    html.Div(
        dash_table.DataTable(
          id='result-table',
          columns = [{"name": i, "id": j} for i,j in zip(df, df.columns)],
          data=df.to_dict('records'),
          fixed_rows={'headers': True},
          # page_size = 25,
          # style_table={'height': 400},  # defaults to 500
          sort_action='native',
          filter_action = 'native',
          editable=True,
          row_deletable=True,
        ), style={'width': '49%', 'display': 'inline-block'}
    ),
    html.Div(id='vidfr-container', style={'width': '49%', 'display': 'inline-block'}),
])

# Open and capture the track result mp4
res_vid_path = "track_result/exp6/vid_1_Trim.mp4"
cap = cv2.VideoCapture(res_vid_path)

# interactive datatable
# https://dash.plotly.com/datatable/interactivity
@app.callback(
    Output('vidfr-container', 'children'),
    Input('result-table', 'active_cell') 
)
def update_videoframe(active_cell):
    selected_row = df.loc[active_cell["row"]]
    selected_frame = selected_row["frame"]
    bb_left, bb_top, bb_width, bb_height, = selected_row["bb_left"], selected_row["bb_top"], selected_row["bb_width"], selected_row["bb_height"]

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
    frame = go.Figure(go.Image(z=frame))
    frame.add_trace(go.Scatter(x=[bb_left, bb_left, bb_left+bb_width, bb_left+bb_width], y=[bb_top, bb_top+bb_height, bb_top, bb_top+bb_height], marker=dict(color='red', size=16)))
    frame.update_layout(
        # title=f"Selected Frame: {selected_frame}",
        # width=500,
        # height=500,
        margin=dict(l=0, r=0, b=0, t=0),        
    )    

    return [
        html.Div(f'active_cell: {active_cell}'),
        html.Div(f'selected_row: {selected_row}'),
        html.Div(html.H2(f'track vid frame - selected_frame: {selected_frame}')),
        dcc.Graph(figure=frame),
        # html.Img(src="/video_feed")
    ]

if __name__ == '__main__':
    app.run_server(debug=True)