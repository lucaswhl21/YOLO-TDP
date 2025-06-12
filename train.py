
 
from ultralytics import YOLOv10
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
model = YOLOv10("G:\\All_Radar\\CODE\\yolov10\\my_yolo\\my_yolo_CABM\\ultralytics\\cfg\\models\\v10\\my_yolov10n_SE_1detect_GSconv_involution.yaml")  # build a new model from scratch

 
# Train the model
results = model.train(data="my_data.yaml", epochs=5000, imgsz=640,batch=32,workers=0,device=0,patience=30, cos_lr=True)


