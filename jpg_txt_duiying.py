import os

# 图片文件所在文件夹路径
image_folder_path = "G:\All_Radar\CODE\yolov10\my_yolo\my_yolo_CABM\\random_huadon\images"
# txt文件所在文件夹路径
txt_folder_path = "G:\All_Radar\CODE\yolov10\my_yolo\my_yolo_CABM\\random_huadon\labels"

# 获取图片文件列表
image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
# 获取txt文件列表
txt_files = [os.path.splitext(f)[0] for f in os.listdir(txt_folder_path) if f.endswith('.txt')]

# 遍历图片文件
for image_file in image_files:
    image_name_prefix = os.path.splitext(image_file)[0]
    corresponding_txt_name = image_name_prefix + ".txt"
    if corresponding_txt_name not in txt_files:
        # 如果不存在对应的txt文件，则创建一个空的txt文件
        new_txt_path = os.path.join(txt_folder_path, corresponding_txt_name)
        with open(new_txt_path, 'w') as f:
            pass