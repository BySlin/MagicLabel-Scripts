import os
from multiprocessing import Queue, Pipe, Manager
from pathlib import Path
from typing import Union

from custom import custom_framework
from lib.SSE import ServerSentEvents
from lib.process_msg_thread import ProcessMsgThread
from yolo_task import predict_task_process

# 加载自定义包
custom_framework()

# 当前目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 模型目录
models_directory = os.path.join(current_directory, "..")

# 加载ultralytics是否成功
load_ultralytics_success = False
# 加载yolov5是否成功
load_yolov5_success = False
# 加载sahi是否成功
load_sahi_success = False

try:
  from ultralytics import FastSAM, SAM, YOLOE, YOLOWorld

  load_ultralytics_success = True
except ImportError:
  FastSAM = None
  SAM = None
  YOLOE = None
  YOLOWorld = None
  load_ultralytics_success = False

try:
  import yolov5

  load_yolov5_success = True
except ImportError:
  load_yolov5_success = False

try:
  from sahi import AutoDetectionModel

  load_sahi_success = True
except ImportError:
  load_sahi_success = False

try:
  from sam2.build_sam import build_sam2_video_predictor, build_sam2
  from sam2.sam2_video_predictor import SAM2VideoPredictor
  from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
  build_sam2 = None
  build_sam2_video_predictor = None
  SAM2VideoPredictor = None
  SAM2ImagePredictor = None

# SSE事件
sse_events = ServerSentEvents()

# 多进程管理器
manager: Union[Manager, None] = None
# 推理消息
predict_msg_queue: Union[Queue, None] = None
# 导出消息
export_msg_queue: Union[Queue, None] = None
# 模型验证消息
model_predict_msg_queue: Union[Queue, None] = None

# 推理进程
predict_process: Union[ProcessMsgThread, None] = None
# 推理进程消息通道
predict_main_conn, predict_process_conn = Pipe()
# 导出进程
export_process: Union[ProcessMsgThread, None] = None
# 模型验证进程
model_predict_process: Union[ProcessMsgThread, None] = None

# SAM模型
sam_model: Union[FastSAM, SAM, YOLOE, YOLOWorld, None] = None

# SAM2模型
sam2_video_predictor: Union[SAM2VideoPredictor, None] = None
sam2_image_predictor: Union[SAM2ImagePredictor, None] = None

def initialize():
  global sse_events, manager, predict_msg_queue, export_msg_queue, model_predict_msg_queue, predict_process, predict_process_conn, sam_model, load_ultralytics_success

  manager = Manager()
  predict_msg_queue = manager.Queue()
  export_msg_queue = manager.Queue()
  model_predict_msg_queue = manager.Queue()

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
  global predict_process, export_process, model_predict_process
  if predict_process is not None:
    if predict_process is not None:
      predict_process.terminate()
      predict_process = None
    if export_process is not None:
      export_process.terminate()
      export_process = None
    if model_predict_process is not None:
      model_predict_process.terminate()
      model_predict_process = None


def set_sam_model(model_path):
  global sam_model, load_ultralytics_success
  if load_ultralytics_success:
    stem = Path(model_path).stem.lower()
    if "fastsam" in stem:
      sam_model = FastSAM(model_path)
      return True
    elif "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem or "mobile_" in stem:
      sam_model = SAM(model_path)
      return True
    elif "yoloe" in stem:
      sam_model = YOLOE(model_path)
      return True
  return False


def set_sam2_video_model(config_file, model_path):
  global sam2_video_predictor
  if build_sam2_video_predictor is not None:
    sam2_video_predictor = build_sam2_video_predictor(config_file, model_path)
    return True
  return False


def set_sam2_image_model(config_file, model_path):
  global sam2_image_predictor
  if build_sam2 is not None:
    sam2_image_predictor = build_sam2(config_file, model_path)
    return True
  return False
