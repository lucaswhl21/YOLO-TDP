
 
from ultralytics import YOLOv10

model = YOLOv10('yolov10n.pt')

 
# Train the model
results = model.val(data="coco128.yaml",  imgsz=640)


