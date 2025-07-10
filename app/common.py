import os
import traceback
from multiprocessing import Queue, Pipe, Manager
from pathlib import Path
from typing import Union

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
# 加载yolov5是否成功
load_yolov5_success = module_exists("yolov5")
# 加载sahi是否成功
load_sahi_success = module_exists("sahi")
# 加载SAM2是否成功
load_sam2_success = module_exists("sam2")

# SSE事件
sse_events = ServerSentEvents()

# 多进程管理器
manager = None
# 推理消息
predict_msg_queue = None

# 推理进程
predict_process = None
# 推理进程消息通道
predict_main_conn, predict_process_conn = Pipe()

# SAM模型
sam_model = None

# SAM2模型
sam2_video_predictor = None
sam2_image_predictor = None

def initialize():
  global sse_events, manager, predict_msg_queue, predict_process, predict_process_conn, sam_model, load_ultralytics_success

  manager = Manager()
  predict_msg_queue = manager.Queue()

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
    except ImportError:
      return False
    sam2_image_predictor = build_sam2(config_file, model_path)
    return True
  return False
