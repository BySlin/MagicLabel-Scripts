import os
from multiprocessing import Queue, Pipe
from pathlib import Path
from typing import Union, TYPE_CHECKING, Callable

from custom import custom_framework
from lib.SSE import ServerSentEvents
from lib.process_msg_thread import ProcessMsgThread
from lib.utils import module_exists
from yolo_task import predict_task_process

# 加载自定义包
custom_framework()

# 当前目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 模型目录
models_directory = os.path.join(current_directory, "..")

# 加载ultralytics是否成功
load_ultralytics_success = module_exists("ultralytics")
print("[MagicLabel]load_ultralytics_success:", load_ultralytics_success)
# 加载yolov5是否成功
load_yolov5_success = module_exists("yolov5")
print("[MagicLabel]load_yolov5_success:", load_yolov5_success)
# 加载sahi是否成功
load_sahi_success = module_exists("sahi")
print("[MagicLabel]load_sahi_success:", load_sahi_success)
# 加载SAM2是否成功
load_sam2_success = module_exists("sam2")
print("[MagicLabel]load_sam2_success:", load_sam2_success)
# 加载clip是否成功
load_clip_success = module_exists("clip")
print("[MagicLabel]load_clip_success:", load_clip_success)

if TYPE_CHECKING:
  from ultralytics import FastSAM, SAM, YOLOE, YOLOWorld
  from sam2.sam2_video_predictor import SAM2VideoPredictor
  from sam2.sam2_image_predictor import SAM2ImagePredictor
  from clip.model import CLIP
  from torch import Tensor
  from PIL import Image

# SSE事件
sse_events = ServerSentEvents()

# 推理消息
predict_msg_queue: Union[Queue, None] = None

# 推理进程
predict_process: Union[ProcessMsgThread, None] = None
# 推理进程消息通道
predict_main_conn, predict_process_conn = Pipe()

# SAM模型
sam_model: Union["FastSAM", "SAM", "YOLOE", "YOLOWorld", None] = None

# SAM2模型
sam2_video_predictor: Union["SAM2VideoPredictor", None] = None
sam2_image_predictor: Union["SAM2ImagePredictor", None] = None

# CLIP模型
clip_model: Union["CLIP", None] = None
clip_preprocess: Union["Callable[[Image], Tensor]", None] = None

def initialize():
  global sse_events, predict_msg_queue, predict_process, predict_process_conn, sam_model, load_ultralytics_success

  predict_msg_queue = Queue()

  # 启动识别进程
  predict_process = ProcessMsgThread(
    msg_queue=predict_msg_queue,
    sse=sse_events,
    target=predict_task_process,
    args=(
      predict_process_conn,
      predict_msg_queue,
    ),
  )
  predict_process.start()

def destroy():
  global predict_process
  if predict_process is not None:
    if predict_process is not None:
      predict_process.terminate()
      predict_process = None

def set_sam_model(model_path):
  global sam_model, load_ultralytics_success
  if load_ultralytics_success:
    try:
        from ultralytics import FastSAM, SAM, YOLOE
    except ImportError:
        return False
    stem = Path(model_path).stem.lower()
    if "fastsam" in stem:
      if FastSAM is not None:
        sam_model = FastSAM(model_path)
        return True
    elif "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem or "mobile_" in stem:
      if SAM is not None:
        sam_model = SAM(model_path)
        return True
    elif "yoloe" in stem:
      if YOLOE is not None:
        sam_model = YOLOE(model_path)
        return True
  return False


def set_sam2_video_model(config_file, model_path):
  global sam2_video_predictor
  if load_sam2_success:
    try:
      from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
      return False
    sam2_video_predictor = build_sam2_video_predictor(config_file, model_path)
    return True
  return False


def set_sam2_image_model(config_file, model_path):
  global sam2_image_predictor
  if load_sam2_success:
    try:
      from sam2.build_sam import build_sam2
      from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
      return False
    sam2_image_predictor = SAM2ImagePredictor(build_sam2(config_file, model_path))
    return True
  return False


def load_clip_model():
  global clip_model, clip_preprocess
  if load_clip_success:
    try:
      import clip
    except ImportError:
      return False
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
    return True
  return False


def extract_clip_feature(img):
  global clip_model, clip_preprocess
  try:
    import cv2
    import torch
    from PIL import Image
  except ImportError:
    return None
  """单张图像提特征，保持原代码兼容（主要用于支持图）"""
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  pil_img = Image.fromarray(img_rgb)
  inp = clip_preprocess(pil_img).unsqueeze(0).to("cuda")
  with torch.no_grad():
    feat = clip_model.encode_image(inp)
    feat /= feat.norm(dim=-1, keepdim=True)
  return feat
