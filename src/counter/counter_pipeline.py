# count vehicle
# virtual line
# A Multi-Class Multi-Movement Vehicle Counting Framework for Traffic Analysis in Complex Areas Using CCTV Systems
# Distinguish region
# shape based
# Robust Movement-Specific Vehicle Counting at Crowded Intersections

# virtual line
# 1) reproduce regions on image coordinate space from json file
# 2) calculate center of bb for trajectory processing
# 3) set region label for each track id and frame
# 4) count by direction (region moving order)

import os
import json
import numpy as np
import pandas as pd
import cv2
from itertools import permutations #, product, combinations
from tqdm import tqdm
import time
import argparse

# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.io as pio

# set ROOT path
ROOT = os.getcwd() # ROOT dir is 
if ROOT in __file__:
    # if interactive, delete "/src/counter" from ROOT
    ROOT = ROOT.replace("/src/counter", "")

def count(args):
    # parse argument from command line
    fpass, vlpass = args["fpass"], args["vlpass"]

    # read video file size
    vid = cv2.VideoCapture(ROOT + fpass + ".mp4")
    vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.release() # release vid memory
        
    # read tracking dataframe from yolov5_deepsort_pytorch
    df = pd.read_table(ROOT + fpass + ".txt", header=None, delimiter=" ")
    # drop x y z columns
    df = df.drop([8, 9, 10], axis=1)
    # add unique row id
    df["id"] = df.index
    # add MOT format header + calss + confidence
    # https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch/issues/55
    # https://motchallenge.net/instructions/
    df.columns = ["frame", "obj_id", "bb_left", "bb_top", "bb_width", "bb_height", "class", "conf", "id"]
    # coco dataset class
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    class_name = {0:"person", 1:"bicycle", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}
    df = df.replace({"class": class_name})

    # compute detected vehicle bounding box centroids
    def compute_centroid(x, y, w, h):
        # opencv receive integer only
        # np.rint() 偶数丸め
        # https://numpy.org/doc/stable/reference/generated/numpy.rint.html#numpy.rint
        centroid_x = np.rint(x + w / 2).astype("int64")
        centroid_y = np.rint(y + h / 2).astype("int64")
        return [centroid_x, centroid_y]

    df["bb_centroid_x"], df["bb_centroid_y_inv"] = compute_centroid(df["bb_left"], df["bb_top"], df["bb_width"], df["bb_height"])

    # invert y of centroid from cv coord (left-top origin) to normal coord (left-bottom origin)
    df["bb_centroid_y"] = vid_height - df["bb_centroid_y_inv"]

    # id is unique within class, make new unique id between every track (class+id)
    df["clsid"] = df["class"] + df["obj_id"].astype("str")

    # reorder columns
    df = df[["id", "frame", "class", "obj_id", "clsid", "conf", "bb_left", "bb_top", "bb_width", "bb_height", "bb_centroid_x", "bb_centroid_y_inv", "bb_centroid_y"]]

    # save to track result folder
    # df.to_csv(ROOT + fpass + "_trj.csv", index=False)

    # read reagion info json file
    with open(ROOT + vlpass, 'r') as f:
        vline_info = json.load(f)
    vline_info = np.array(vline_info[0]["coords"])
    vline_info[:, 1] = vid_height - vline_info[:, 1] # inv

    # define vline
    # get 2 points combinations from 4 sides polygon, set vline id (name)
    vlnames = []
    vlcoods = []
    for i in range(len(vline_info)):
        if i < len(vline_info)-1:
            vlnames.append("vl" + str(i))
            vlcoods.append([vline_info[i].tolist(), vline_info[i+1].tolist()])
        else:
            vlnames.append("vl" + str(i))
            vlcoods.append([vline_info[i].tolist(), vline_info[0].tolist()])
    vlines = dict(zip(vlnames, vlcoods))

    # define movement (directions), all possible permutations
    movprm = list(permutations(list(vlines.keys()), 2))
    movnames = ["mov" + str(i) for i in range(len(movprm))]
    movs = dict(zip(movnames, movprm))

    # save vline movement info as json file
    vlmov_info = [
        {
            "virtual_lines": vlines,
            "movements": movs
        }
    ]
    with open(ROOT + fpass + '_vlmov.json', 'w') as f:
        json.dump(vlmov_info, f)    

    # def show_vline_cv(vline_coords):
    #     window_name = 'Virtual Line'
    #     img = np.zeros((vid_height, vid_width, 3), np.uint8)

    #     for vline in vline_coords.keys():
    #         start_point = vline_coords[vline][0]
    #         end_point = vline_coords[vline][1]
    #         color = (255, 0, 0)
    #         thickness = 9
    #         img = cv2.line(img, start_point, end_point, color, thickness)
    #         img = cv2.putText(img, str(start_point), start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
    #         img = cv2.putText(img, vline, np.rint((start_point+end_point)/2).astype("int64"), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # normal size window
    #     cv2.imshow(window_name, img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # show_vline_cv(vlines)

    # # extract individual vehicle instance frames
    # def plot_trajectory_cv(centroid_array):
    #     window_name = 'Trajectory'
    #     img = np.zeros((vid_height, vid_width, 3), np.uint8)
        
    #     color = (0, 255, 0)
    #     thickness = 3
    #     img = cv2.polylines(img, [centroid_array], False, color, thickness)
        
    #     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # normal size window
    #     cv2.imshow(window_name, img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # df_i = df[df["clsid"] == df["clsid"].unique()[0]]
    # df_i_centroid = df_i[["bb_centroid_x", "bb_centroid_y_inv"]].values
    # plot_trajectory_cv(df_i_centroid)

    # 線分ABと線分CDの交点を求める関数
    # https://rikoubou.hatenablog.com/entry/2019/04/03/163751
    def calc_cross_point(pointA, pointB, pointC, pointD):
        cross_points = (0,0)
        bunbo = (pointB[0] - pointA[0]) * (pointD[1] - pointC[1]) - (pointB[1] - pointA[1]) * (pointD[0] - pointC[0])

        # 直線が平行な場合
        if (bunbo == 0):
            return False, cross_points

        vectorAC = ((pointC[0] - pointA[0]), (pointC[1] - pointA[1]))
        r = ((pointD[1] - pointC[1]) * vectorAC[0] - (pointD[0] - pointC[0]) * vectorAC[1]) / bunbo
        s = ((pointB[1] - pointA[1]) * vectorAC[0] - (pointB[0] - pointA[0]) * vectorAC[1]) / bunbo

        # 線分AB、線分AC上に存在しない場合
        if (r <= 0) or (1 <= r) or (s <= 0) or (1 <= s):
            return False, cross_points

        # rを使った計算の場合
        distance = ((pointB[0] - pointA[0]) * r, (pointB[1] - pointA[1]) * r)
        cross_points = (int(pointA[0] + distance[0]), int(pointA[1] + distance[1]))

        # sを使った計算の場合
        # distance = ((pointD[0] - pointC[0]) * s, (pointD[1] - pointC[1]) * s)
        # cross_points = (int(pointC[0] + distance[0]), int(pointC[1] + distance[1]))

        return True, cross_points

    def check_vline_intersect(v):
        df_vlcross_rows = []
        for i in df[df["clsid"] == v].index[1:]:
            p_cur = df[df["clsid"] == v][["bb_centroid_x", "bb_centroid_y"]].loc[i].values
            p_pre = df[df["clsid"] == v][["bb_centroid_x", "bb_centroid_y"]].shift(1).loc[i].values
            for l in vlines.keys():
                cross_bool, cross_point = calc_cross_point(p_cur, p_pre, vlines[l][0], vlines[l][1])
                if cross_bool:
                    intersect_frame = df[df["clsid"] == v].loc[i]["frame"]
                    row_id = df[df["clsid"] == v].loc[i]["id"]
                    df_vlcross_rows.append([row_id, v, l, intersect_frame, cross_point])
                else:
                    pass
        return df_vlcross_rows # ["id", "clsid", "vline", "crossed_frame", "crossed_coord"]

    def search_vlcross():
        # for loop
        df_vlcross = []
        for v in tqdm(df["clsid"].unique()):
                df_vlcross += check_vline_intersect(v)
        # map
        # https://stackoverflow.com/questions/37418611/convert-a-nested-for-loop-to-a-map-equivalent
        # list conprehension
        df_vlcross = pd.DataFrame(df_vlcross)
        df_vlcross.columns = ["id", "clsid", "vline", "crossed_frame", "crossed_coord"]
        return df_vlcross

    start_time = time.time()
    df_vlcross = search_vlcross() # maybe slow?, not frame base loop, may not be implemented in realtime situation or sliders in plotly
    print("elapsed time: ", time.time() - start_time)
    df_vlcross.to_csv(ROOT + fpass + "_crossed.csv", index=False)

    # merge to main trj df
    df = pd.merge(df, df_vlcross[["id", "vline", "crossed_frame", "crossed_coord"]], how="left", on="id")
    df.to_csv(ROOT + fpass + "_trj.csv", index=False)

    # count vehicle foe each movement
    # if one vehicle crossed the particular vline several times, treat them as crossed once?
    # or more strictly, only allow 2 crossed row and 2 uique crossed vline set
    # ["clsid", "mov", "crossed_frame"]
    _df_uniquevl = df_vlcross.groupby('clsid').nunique()["vline"]
    moved_vehicles = _df_uniquevl.index[
        (df_vlcross.groupby('clsid').nunique()["vline"] == 2) & 
        (df_vlcross.groupby('clsid').count()["vline"] == 2)] # strict filter

    df_mov_count_rows = [] # ["clsid", "mov", "vl1", "vl2", "vl1_crossed_frame", "vl1_crossed_coord", "vl2_crossed_frame", "vl2_crossed_coord"]
    for v in moved_vehicles:
        mov_vl_set = tuple(df_vlcross[df_vlcross["clsid"] == v]["vline"].unique())
        
        def get_keys_from_value(d, val):
            return [k for k, v in d.items() if v == val]

        df_mov_count_rows.append(
            [
                v, get_keys_from_value(movs, mov_vl_set)[0], mov_vl_set[0], mov_vl_set[1], 
                df_vlcross[df_vlcross["clsid"] == v].iloc[0]["crossed_frame"], df_vlcross[df_vlcross["clsid"] == v].iloc[0]["crossed_coord"],
                df_vlcross[df_vlcross["clsid"] == v].iloc[1]["crossed_frame"], df_vlcross[df_vlcross["clsid"] == v].iloc[1]["crossed_coord"]
            ]
        )

    df_mov_count = pd.DataFrame(df_mov_count_rows, columns=["clsid", "mov", "vl1", "vl2", "vl1_crossed_frame", "vl1_crossed_coord", "vl2_crossed_frame", "vl2_crossed_coord"])
    df_mov_count.to_csv(ROOT + fpass + "_crossed_strict.csv", index=False)

if __name__ == "__main__":
    # https://stackoverflow.com/questions/7427101/simple-argparse-example-wanted-1-argument-3-results
    parser = argparse.ArgumentParser(description='Count vehicle with Yolov5 + DeepSort output')
    parser.add_argument('--fpass', help='Yolov5 + DeepSort out put file (.mp4 and .txt) pass with common file name', required=True)
    parser.add_argument('--vlpass', help='Virtual line .json file pass', required=True)
    args = vars(parser.parse_args())

    count(args)