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
import os


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
        self.mode = True # if True, draw zone. Press 'm' to toggle to direction mode
        self.line_thickness = 5
        self.start_x, self.start_y = 0, 0 # initialize start coordinate
        self.temp_end_x, self.temp_end_y = 0, 0 # initialize temporary coordinate

        # zone mode attributes
        self.zone_color = (0, 0, 255)
        self.zone_count = 0
        self.zone_node_count = 0 # only 4 nodes acceptable
        self.zone_data = []
        self.saved_zone = [["zone_id", "node1_x", "node1_y", "node2_x", "node2_y", "node3_x", "node3_y", "node4_x", "node4_y"]]

        # direction mode attributes
        self.direction_color = (0, 255, 0)
        self.direction_count = 0
        self.saved_direction = [["direction_id", "start_x", "start_y", "end_x", "end_y"]]

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
        button_save = tk.Button(_fr_menu, text="save", command=self.save)
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
        self.log_txt.insert('1.0', '##### Zone Writer #####\n')
        return _fr_log

    # select image function for open file button
    def openfile(self, frame_id=0):
        f_path = tk.filedialog.askopenfilename(
            title="open file", initialdir="./images", 
            filetypes=[("Image file", ".mp4")]) # don't show "MP4"
        str_file_path = str(f_path) # 絶対パスになっている
        self.basename = os.path.splitext(os.path.basename(str_file_path))[0] #  basename without ext
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
        cv2.setMouseCallback(self.cvwindow_name, self.draw_zone)
        # wait for esc key to be pressed to exit
        while(1):
            # show original frame (may be written line)
            cv2.imshow(self.cvwindow_name, self.frame)
            if not cv2.EVENT_MOUSEMOVE:
                # after lbuttondown, when mouse stop move, draw from temporary coords
                copy = self.frame.copy()
                if self.mode:
                    if self.zone_node_count <= 1:
                        copy = cv2.line(copy, (self.start_x,self.start_y), (self.temp_end_x,self.temp_end_y), self.zone_color, self.line_thickness)
                    else:
                        copy = cv2.line(copy, (self.start_x,self.start_y), (self.temp_end_x,self.temp_end_y), self.zone_color, self.line_thickness)
                        pts = np.array(self.zone_data)
                        pts = pts.reshape((-1,1,2))
                        copy = cv2.polylines(copy, [pts], True, self.zone_color, self.line_thickness)
                else:
                    copy = cv2.arrowedLine(copy, (self.start_x,self.start_y), (self.temp_end_x,self.temp_end_y), self.direction_color, self.line_thickness)
                cv2.imshow(self.cvwindow_name, copy)
            # waitKey thing
            k = cv2.waitKey(1) & 0xFF
            if k == ord('m'): #toggle between zone and direction
                self.mode = not self.mode
                if self.mode:
                    print("zone mode")
                else:
                    print("direction mode")
                self.start_x, self.start_y = 0, 0 # initialize start coordinate
                self.temp_end_x, self.temp_end_y = 0, 0 # initialize temporary coordinate
            elif k == ord('x'): #resets the image (removes lines) with "x" key
                self.frame = clean_frame
                self.start_x, self.start_y = 0, 0 # initialize start coordinate
                self.temp_end_x, self.temp_end_y = 0, 0 # initialize temporary coordinate
                self.direction_count = 0
                self.saved_direction = [["line_id", "start_x", "start_y", "end_x", "end_y"]]
            if cv2.waitKey(1) & 0xFF == 27: # exit opencv window with "esc" key
                break
        cv2.destroyAllWindows()

    # mouse callback function
    def draw_zone(self, event, x, y, flags, param):
        if self.mode:
            # zone
            # print("zone mode")
            # while self.zone_node_count <= 3:
            if event == cv2.EVENT_LBUTTONDOWN:
                # click down lbutton
                # print('inside mouse lbutton down event....')
                # node draw
                self.start_x, self.start_y = x, y # set start coord
                self.temp_end_x, self.temp_end_y = x, y # clear temporary coord
                print(f"node {self.zone_node_count}: {self.start_x}, {self.start_y}")
                self.zone_data.append([self.start_x, self.start_y])
                print(self.zone_data)
                self.zone_node_count += 1
                if self.zone_node_count <= 3:
                    self.drawing = True
                else:
                    self.drawing = False
                    print("node draw ended")
                    self.log_txt.insert(tk.END, f'zone: , {self.zone_data}\n')
                    # self.zone_node_count = 0

            elif event == cv2.EVENT_MOUSEMOVE:
                # moving mouse while keep clicking down lbutton
                copy = self.frame.copy()
                if self.drawing == True:
                    self.temp_end_x, self.temp_end_y = x, y # set temporary coord
                    # draw temp line in copy
                    cv2.line(copy, (self.start_x,self.start_y), (self.temp_end_x,self.temp_end_y), self.zone_color, self.line_thickness)
                    # show copy
                    cv2.imshow(self.cvwindow_name, copy)

        else:
            # direction
            if event == cv2.EVENT_LBUTTONDOWN:
                # click down lbutton
                # print('inside mouse lbutton down event....')
                self.drawing = True
                self.start_x, self.start_y = x, y # set start coord
                self.temp_end_x, self.temp_end_y = x, y # clear temporary coord

            elif event == cv2.EVENT_MOUSEMOVE:
                # moving mouse while keep clicking down lbutton
                copy = self.frame.copy()
                if self.drawing == True:
                    self.temp_end_x, self.temp_end_y = x, y # set temporary coord
                    # draw temp line in copy
                    cv2.arrowedLine(copy, (self.start_x,self.start_y), (self.temp_end_x,self.temp_end_y), self.direction_color, self.line_thickness)
                    # show copy
                    cv2.imshow(self.cvwindow_name, copy)

            elif event == cv2.EVENT_LBUTTONUP:
                # release mouse left click
                # print('....inside mouse lbutton up event')
                self.drawing = False
                # draw final line in original frame
                cv2.arrowedLine(self.frame, (self.start_x,self.start_y), (x, y), self.direction_color, self.line_thickness)
                cv2.putText(self.frame, f'line: {self.direction_count}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, self.direction_color, 2, cv2.LINE_AA)
                # save line in list
                self.saved_direction.append([self.direction_count, self.start_x, self.start_y, x, y])
                self.log_txt.insert(tk.END, f'line: {self.direction_count}, {(self.start_x,self.start_y)}, {(x, y)}\n')
                self.direction_count += 1

    def save(self):
        if self.mode:
            # save virtual line in json
            vline_info = [
                {
                    "label": "virtual_line",
                    "coords": self.zone_data
                }
            ]
            vline_save_path = './vline/' + self.basename + '_vline_info.json'   
            with open(vline_save_path, 'w') as f:
                json.dump(vline_info, f)
            self.log_txt.insert(tk.END, 'virtual line saved in' + vline_save_path + '\n')

        else:
            df = pd.DataFrame(self.saved_direction)
            df.to_csv("./zone/directions.csv", encoding="utf-8", header=False, index=False)
            self.log_txt.insert(tk.END, 'direction file saved in ./zone/directions.csv\n')

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