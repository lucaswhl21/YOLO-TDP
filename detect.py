import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLOv10


model = YOLOv10('G:\\All_Radar\\CODE\\yolov10\\my_yolo\\my_yolo_CABM\\runs\detect\\train15\\weights\\yolov10-meibian.pt')
# model = YOLOv10('G:\\All_Radar\\CODE\\yolov10\\my_yolo\\my_yolo_CABM\\yolov10n.pt')
results = model('G:\\All_Radar\\CODE\\yolov10\my_yolo\\my_yolo_CABM\\feature map\\1.JPG', save=True,visualize=True)


