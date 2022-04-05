# https://gipheshouse.com/2020/10/28/python-7-%E3%83%89%E3%83%A9%E3%83%83%E3%82%B0%EF%BC%86%E3%83%89%E3%83%AD%E3%83%83%E3%83%97%E3%81%A7%E5%9B%B3%E5%BD%A2%E3%82%92%E6%8F%8F%E3%81%8F%E3%80%90%E3%83%9E%E3%83%83%E3%83%97gui%E3%80%91/
# https://qiita.com/KentoSugiyama7974/items/b1a30a25dc4af7f1cdfe
# https://python.keicode.com/advanced/tkinter.php
# https://kuroro.blog/python/0KVm0XNc0gvKrbM4O9bD/
# https://mokumokucouple.com/python_tkinter/
# https://denno-sekai.com/tkinter-bind/
# https://imagingsolution.net/program/python/tkinter/widget_layout_pack/
# https://docs.opencv.org/4.x/db/d5b/tutorial_py_mouse_handling.html
# https://stackoverflow.com/questions/50234485/drawing-rectangle-in-opencv-python

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import scrolledtext
import tkinter.filedialog
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import pandas as pd
import json


class App():
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('Zone Writer')
        # self.window_width = int(self.window.winfo_screenwidth() * 2 / 3) # monitor width *2/3
        # self.window_height = int(self.window.winfo_screenheight() * 2 / 3) # monitor height *2/3
        self.window_width = 360
        self.window_height = 640
        self.window.geometry(str(self.window_width)+"x"+str(self.window_height))

        self.cvwindow_name = None # opencv imshow window name string
        
        self.drawing = False
        self.line_thickness = 5

        self.mode = 1 # 1: Single Lane, 2: cross road, 3: T road
        
        # lane mode attributes
        self.line_coord_lane = [[0, 0], [0, 0]] # initialize coordinate
        self.line_color_lane = (255, 0, 0)
        self.virtual_lines_lane = [] #[{"line_id": None, "coords": None}]
        self.click_count_lane = 0

        # cross road mode attributes
        self.line1_coord_cross = [[0, 0], [0, 0]]
        self.line2_coord_cross = [[0, 0], [0, 0]]
        self.line_color_cross = (0, 255, 0)
        self.virtual_lines_cross = []
        self.click_count_cross = 0

        # T road mode attributes
        self.center_coord_t = [0, 0] # center point of 3 part separating lines
        self.end_coord_t = [0, 0] # direction point of 3 part separating lines
        self.line_color_t = (0, 0, 255)
        self.virtual_lines_t = [] # line segment
        self.click_count_t = 0

        self.setup_display()
        self.window.mainloop()

    # 画面を描画
    def setup_display(self):
        # window中にPanedWindowを作る
        self.pw_main = tk.PanedWindow(self.window, orient='horizontal')
        self.pw_main.pack(expand=True, fill = tk.BOTH, side="left")
        # Add draw line frame in panedwindow
        _fr_menu = self.menu_frame(self.pw_main)
        self.pw_main.add(_fr_menu)
        # Add log frame in panedwindow
        _fr_log = self.log_frame(self.pw_main)
        self.pw_main.add(_fr_log)
        # 画面初期化
        # self.init()

    # frame for open file button and save button
    def menu_frame(self, _pw):
        # define file frame
        _fr_menu = tk.LabelFrame(_pw, text="file")
        # define button widgets
        btn_openfile = tk.Button(_fr_menu, text="open file", command=self.openfile)
        btn_openfile.pack()
        button_save = tk.Button(_fr_menu, text="save region as json", command=self.save_json)
        button_save.pack()
        return _fr_menu

    # show line edge points coordinates
    def log_frame(self, _pw):
        # define log frame
        _fr_log = tk.LabelFrame(_pw, text="log_frame")
        # define scrolled text widget
        self.log_txt = scrolledtext.ScrolledText(_fr_log, width=int(self.window_width/2), height=self.window_height)
        self.log_txt.pack()
        # 初期値の挿入
        self.log_txt.insert('1.0', '##### Region Writer #####\n')
        return _fr_log

    # select image function for open file button
    def openfile(self, frame_id=0):
        f_path = tk.filedialog.askopenfilename(
            title="open file", initialdir="./images", 
            filetypes=[("Image file", ".mp4 .MP4")])
        str_file_path = str(f_path) # 絶対パスになっている
        self.log_txt.insert(tk.END, f'open file {str_file_path}\n')

        # read video file
        self.vid = MyVideoCapture(str_file_path)
        # set current frame
        self.vid.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        # Get a frame from the video source
        ret, self.frame = self.vid.get_frame()
        clean_frame = np.copy(self.frame) # keep clean original frame
        # open new cv2 window
        self.cvwindow_name = str_file_path
        # Create a window and bind the function to window
        cv2.namedWindow(self.cvwindow_name, cv2.WINDOW_NORMAL) # resize as normal
        self.frame_copy = self.frame.copy()
        cv2.setMouseCallback(self.cvwindow_name, self.draw_region)
        # wait for esc key to be pressed to exit
        while(1):
            # show original frame (may be written line)
            cv2.imshow(self.cvwindow_name, self.frame_copy)

            # waitKey thing
            k = cv2.waitKey(1) & 0xFF
            if k == ord("1"): # toggle to lane mode
                self.mode = 1
                print("lane mode")
            elif k == ord("2"):
                self.mode = 2
                print("cross road mode")
            elif k == ord("3"):
                self.mode = 3
                print("T road mode")                                
            elif k == ord('x'): #resets the image (removes lines) with "x" key
                self.frame = clean_frame
                self.start_x, self.start_y = 0, 0 # initialize start coordinate
                self.temp_end_x, self.temp_end_y = 0, 0 # initialize temporary coordinate
                self.direction_count = 0
                self.saved_direction = [["line_id", "start_x", "start_y", "end_x", "end_y"]]
            if cv2.waitKey(1) & 0xFF == 27: # exit opencv window with "esc" key
                print("Break Opencv imshow window loop.")
                break
        cv2.destroyAllWindows()
            

    # mouse callback function
    # draw line for Region encoding 
    def draw_region(self, event, x, y, flags, param):
        if self.mode == 1:
            # lane mode
            # print("lane mode")
            if event == cv2.EVENT_LBUTTONDOWN:
                # click down lbutton, save coords
                if self.click_count_lane <= 1:
                    self.line_coord_lane[self.click_count_lane] = [x, y] # set start point
                    self.click_count_lane += 1
                    if self.click_count_lane == 2:
                        # write infinity line on self.frame_copy
                        # https://stackoverflow.com/questions/59578855/opencv-python-how-do-i-draw-a-line-using-the-gradient-and-the-first-point?rq=1
                        self.draw_line(self.frame_copy, self.line_coord_lane, self.line_color_lane, self.line_thickness)
                else:
                    print("draw ended")
                self.log_txt.insert(tk.END, f'lane virtual line: {self.line_coord_lane} \n')
            elif event == cv2.EVENT_RBUTTONDOWN:
                # click down rbutton, print `self.line_coord_lane`
                print(self.line_coord_lane)

        elif self.mode == 2:
            # cross road mode
            # print("cross road mode")
            if event == cv2.EVENT_LBUTTONDOWN:
                # click down lbutton, save coords
                if self.click_count_cross <= 1:
                    # save 1st virtual line
                    self.line1_coord_cross[self.click_count_cross] = [x, y]
                    self.click_count_cross += 1
                    if self.click_count_cross == 2:
                        # write infinity line on self.frame_copy
                        self.draw_line(self.frame_copy, self.line1_coord_cross, self.line_color_cross, self.line_thickness)
                elif self.click_count_cross <= 3:
                    # save 2nd virtual line
                    self.line2_coord_cross[self.click_count_cross-2] = [x, y]
                    self.click_count_cross += 1
                    if self.click_count_cross == 4:
                        # write infinity line on self.frame_copy
                        self.draw_line(self.frame_copy, self.line2_coord_cross, self.line_color_cross, self.line_thickness)
                else:
                    print("draw ended")
                self.log_txt.insert(tk.END, f'lane virtual line: {self.line1_coord_cross}, {self.line2_coord_cross} \n')

            elif event == cv2.EVENT_RBUTTONDOWN:
                # click down rbutton, print `self.line_coord_cross`
                print(self.line1_coord_cross, self.line2_coord_cross)
        
        elif self.mode == 3:
            # t road mode
            # print("t road mode")
            # if event == cv2.EVENT_LBUTTONDOWN:
            #     # click down lbutton, save coords
            #     if self.click_count_cross <= 1:
            #         # save 1st virtual line
            #         self.line1_coord_cross[self.click_count_cross] = [x, y]
            #         self.click_count_lane += 1
            #     elif self.click_count_cross <= 3:
            #         # save 2nd virtual line
            #         self.line2_coord_cross[self.click_count_cross-2] = [x, y]
            #         self.click_count_lane += 1
            #     else:
            #         print("draw ended")

            # elif event == cv2.EVENT_RBUTTONDOWN:
            #     # click down rbutton, print `self.line_coord_cross`
            #     print(self.line_coord_cross)
            pass

    def slope(self, line_coord):
        ###finding slope
        x1, y1 = line_coord[0]
        x2, y2 = line_coord[1]
        if x1 != x2:
            return ( (y2 - y1) / (x2 - x1) )
        else:
            return 'NA'

    def draw_line(self, image, line_coord, color, thickness):
        x1, y1 = line_coord[0]
        x2, y2 = line_coord[1]
        b = self.slope(line_coord)
        h, w = image.shape[:2]
        if b != 'NA':
            ### here we are essentially extending the line to x=0 and x=width
            ### and calculating the y associated with it
            ##starting point
            px = 0
            py = - (x1 - 0) * b + y1
            ##ending point
            qx = w
            qy = - (x2 - w) * b + y2
        else:
        ### if slope is zero, draw a line with x=x1 and y=0 and y=height
            px, py = x1, 0
            qx, qy = x1, h
        cv2.line(image, (int(px), int(py)), (int(qx), int(qy)), color, thickness)

    def save_json(self):
        if self.mode == 1:
            region_info = [
                {
                    "label": "lane_virtual_line",
                    "coords": self.line_coord_lane
                }
            ]
        elif self.mode == 2:
            region_info = [
                {
                    "label": "cross_virtual_line1",
                    "coords": self.line1_coord_cross
                },
                {
                    "label": "cross_virtual_line2",
                    "coords": self.line2_coord_cross
                }
            ]
        elif self.mode == 3:
            region_info = []

        with open('./region/region_info.json', 'w') as f:
            json.dump(region_info, f)
        self.log_txt.insert(tk.END, 'file saved in ./region/region_info.json \n')

    # リセットボタン処理
    def init(self):
        pass

# opencv VideoCapture thing
class MyVideoCapture:
    def __init__(self, video_source):
        # Open the video source
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", video_source)
        
        # Get video source width and height
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # get frame as ndarray
    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                # return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # But opencv imshow() based on BGR, so we'll not switch the order
                return (ret, frame)
            else:
                # No frame
                return (ret, None)
        else:
            # video capture not opened
            # raise ValueError("File not opened")
            return (None, None)

if __name__ == '__main__':
    App()