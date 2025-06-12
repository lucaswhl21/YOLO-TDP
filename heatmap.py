from ultralytics.nn.tasks import YOLOv10DetectionModel as YOLOv10
from ultralytics.nn.tasks import attempt_load_weights
import torch
import numpy as np
from tqdm import trange
from PIL import Image
import cv2
import os
import shutil
import warnings
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from ultralytics.utils.torch_utils import intersect_dicts
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, LayerCAM, RandomCAM, EigenGradCAM

 
 # 注意！ 热力图生成的方法可以选以下几种：GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM 
 # 注意！ 如果遇到环境问题跑不通的，推荐使用B站：Ai学术叫叫兽的云服务器一分钟环境搭建教程，免费的哈！地址在这：【YOLOv10环境搭建：一镜到底，手把手教学，傻瓜式操作，一分钟完全掌握yolov10安装、使用、训练大全，从环境搭建到模型训练、推理，从入门到精通！】 https://www.bilibili.com/video/BV1ZYY9edEpn/?share_source=copy_web&vd_source=b14815a6bc1c88b120b00762984f3b84
# 若依然有问题，B站私信本博主：Ai学术叫叫兽解决即可！必须遥遥领先！！！
 
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
 
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
 
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
 
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
 
 
class ActivationsAndGradients:
    """ Class for extracting activations and registering gradients from targetted intermediate layers """
 
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))

            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))
 
    def save_activation(self, module, input, output):
        activation = output
 
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())
 
    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:

            return
 

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients
 
        output.register_hook(_store_grad)
 
    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
 
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)

        post_result, pre_post_boxes, post_boxes = self.post_process(model_output['one2many'][0])
        return [[post_result, pre_post_boxes]]
 
    def release(self):
        for handle in self.handles:
            handle.remove()
 
 
class yolov10_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
 
    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)
 
 
class yolov10_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model_names = ckpt['model'].names
        model = YOLOv10(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)  # load

        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')
        target = yolov10_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers, use_cuda=device.type == 'cuda')
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int64)
        self.__dict__.update(locals())
 
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result
 
    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img
 
    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
 
    def process(self, img_path, save_path):
        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
 
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return
 
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
 
        pred = self.model(tensor)['one2many']
        pred = self.post_process(pred)
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img,
                                                               grayscale_cam)
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                cam_image = self.draw_detections(data[:4], self.colors[int(data[5])],
                                                 f'{self.model_names[int(data[5])]} {float(data[4]):.2f}',
                                                 cam_image)
 
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)
 
    def __call__(self, img_path, save_path):
        # remove dir if exist # 注意！ 热力图生成的方法可以选以下几种：GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM 
 # 注意！ 如果遇到环境问题跑不通的，推荐使用B站：Ai学术叫叫兽的云服务器一分钟环境搭建教程，免费的哈！地址在这：【YOLOv10环境搭建：一镜到底，手把手教学，傻瓜式操作，一分钟完全掌握yolov10安装、使用、训练大全，从环境搭建到模型训练、推理，从入门到精通！】 https://www.bilibili.com/video/BV1ZYY9edEpn/?share_source=copy_web&vd_source=b14815a6bc1c88b120b00762984f3b84
# 若依然有问题，B站私信本博主：Ai学术叫叫兽解决即可！必须遥遥领先！！！
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)
 
        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            self.process(img_path, f'{save_path}/result.png')
 
 # 注意！ 热力图生成的方法可以选以下几种：GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, , LayerCAM, RandomCAM, EigenGradCAM
 # 注意！ 如果遇到环境问题跑不通的，推荐使用B站：Ai学术叫叫兽的云服务器一分钟环境搭建教程，免费的哈！地址在这：【YOLOv10环境搭建：一镜到底，手把手教学，傻瓜式操作，一分钟完全掌握yolov10安装、使用、训练大全，从环境搭建到模型训练、推理，从入门到精通！】 https://www.bilibili.com/video/BV1ZYY9edEpn/?share_source=copy_web&vd_source=b14815a6bc1c88b120b00762984f3b84
# 若依然有问题，B站私信本博主：Ai学术叫叫兽解决即可！必须遥遥领先！！！

# yolov10n.yaml  my_yolov10n_SE_1detect_GSconv_involution.yaml
def get_params():
    params = {
        'weight': 'G:\\All_Radar\\CODE\\yolov10\\my_yolo\\my_yolo_CABM\\runs\detect\\train15\\weights\\best.pt',
        'cfg': 'G:\\All_Radar\\CODE\\yolov10\\my_yolo\\my_yolo_CABM\\ultralytics\\cfg\\models\\v10\\my_yolov10n_SE_1detect_GSconv_involution.yaml',
        'device': 'cuda',
        'method': 'LayerCAM',
        'layer': [15],
        'backward_type': 'all',  
        'conf_threshold': 0.4,
        'ratio': 1,
        'show_box': True,
        'renormalize': False,

    }
    return params
 
 
def _main(img_path, result_path):
    model = yolov10_heatmap(**get_params())
    model(img_path, result_path)
 
 
if __name__ == '__main__':
    _main('G:\\All_Radar\\CODE\\yolov10\my_yolo\\my_yolo_CABM\\paper_6th_zhang_pic', 'result')
    pass