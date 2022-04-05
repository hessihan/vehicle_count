#!/bin/sh

start_time=`date +%s`

# run Yolov5_DeepSort_Pytorch/track.py for object tracking
cd src/Yolov5_DeepSort_Pytorch
# python  track.py --source ../../images/sample/vid_1_Trim.mp4 --yolo_model yolov5s.pt --project ../../track_result --class 0 1 2 3 5 7 --save-txt --save-vid
python  track.py --source ../../images/sample/vid_1.mp4 --yolo_model yolov5s.pt --project ../../track_result --class 0 1 2 3 5 7 --save-txt --save-vid

end_time=`date +%s`

time=$((end_time - start_time))

echo $time