import base64
import concurrent.futures
import json
import os.path
import threading
import time
from typing import Union, TYPE_CHECKING

import common
from lib.SSE import create_sse_msg
from lib.SimpleHttpServer import Router, RequestHandler
from lib.utils import is_blank, is_not_blank

yolo_router = Router("/api/yolo")

if TYPE_CHECKING:
  from torch import Tensor

support_feat_tensor: Union["Tensor", None] = None

@yolo_router.register("/sse_events")
def sse_events(handler: RequestHandler):
  def send_status():
    time.sleep(0.5)
    common.sse_events.send_raw(
      create_sse_msg(
        "on_load_framework_status",
        {
          "ultralytics": common.load_ultralytics_success,
          "yolov5": common.load_yolov5_success,
          "sahi": common.load_sahi_success,
        },
      )
    )

  threading.Thread(
    target=send_status,
    daemon=True,
  ).start()
  return common.sse_events.response()


@yolo_router.register("/set_predict_model", method="POST")
def set_predict_model(handler: RequestHandler):
  requestBody = handler.read_json()
  model = requestBody["model"]
  framework = requestBody["framework"]
  task = requestBody["task"]

  if is_blank(model) or is_blank(framework):
    return {"success": False, "msg": "运行参数不能为空"}

  common.predict_main_conn.send(
    json.dumps(
      {
        "event": "set_predict_model",
        "model": model,
        "framework": framework,
        "task": task,
      }
    )
  )

  return {"success": True, "msg": "操作成功"}


@yolo_router.register("/predict", method="POST")
def predict(handler: RequestHandler):
  requestBody = handler.read_json()
  command = requestBody["command"]
  cwd = requestBody["cwd"]
  framework = requestBody["framework"]
  sahiSettings = requestBody["sahiSettings"]
  skipExistsAnnotationFile = requestBody.get("skipExistsAnnotationFile", False)

  if (
    is_blank(command)
    or is_blank(cwd)
    or is_blank(framework)
    or sahiSettings is None
  ):
    return {"success": False, "msg": "运行参数不能为空"}

  common.predict_main_conn.send(
    json.dumps(
      {
        "event": "predict",
        "command": command,
        "cwd": cwd,
        "framework": framework,
        "sahiSettings": sahiSettings,
        "skipExistsAnnotationFile": skipExistsAnnotationFile
      }
    )
  )

  return {"success": True, "msg": "操作成功"}


@yolo_router.register("/stop_predict")
def stop_predict(handler: RequestHandler):
  common.predict_main_conn.send(
    json.dumps(
      {
        "event": "stop_predict",
      }
    )
  )

  return {"success": True, "msg": "操作成功"}


@yolo_router.register("/set_sam_model", method="POST")
def set_sam_model(handler: RequestHandler):
  requestBody = handler.read_json()
  model = requestBody["model"]

  if is_blank(model):
    return {"success": False, "msg": "运行参数不能为空"}

  if common.set_sam_model(model):
    return {"success": True, "msg": "加载模型成功"}
  else:
    return {"success": False, "msg": "加载模型失败"}


@yolo_router.register("/sam_model_predict", method="POST")
def sam_model_predict(handler: RequestHandler):
  requestBody = handler.read_json()
  # 工作目录
  cwd = requestBody["cwd"]
  # 输入图片路径
  source = requestBody["source"]
  # 输入参数
  data = requestBody["data"]
  # 单目标检测还是多目标检测
  single = requestBody["single"]

  if is_not_blank(cwd):
    os.chdir(cwd)

  if is_blank(source):
    return {"success": False, "msg": "运行参数不能为空"}

  if common.sam_model is None:
    return {"success": False, "msg": "SAM模型未加载"}

  if single is None:
    single = True

  try:
    from ultralytics import FastSAM
  except ImportError:
    FastSAM = None

  try:
    from ultralytics import SAM
  except ImportError:
    SAM = None

  try:
    from ultralytics import YOLOE
  except ImportError:
    YOLOE = None

  try:
    from ultralytics import YOLOWorld
  except ImportError:
    YOLOWorld = None

  bboxes = []
  points = []
  labels = []
  texts = None

  if type(data) == str:
    texts = data
  else:
    for item in data:
      if item["type"] == "addPoint" or item["type"] == "reducePoint":
        points.append(item["point"])
        labels.append(item["label"])
      elif item["type"] == "addRectangle":
        bboxes.append(item["bbox"])

  if len(bboxes) == 0:
    bboxes = None

  if len(points) == 0:
    points = None

  if len(labels) == 0:
    labels = None

  if isinstance(common.sam_model, FastSAM) or isinstance(common.sam_model, SAM):
    if texts is not None:
      results = common.sam_model(
        source=source, texts=texts, save=False, verbose=False
      )
    else:
      if isinstance(common.sam_model, SAM):
        results = common.sam_model(
          source=source,
          bboxes=[bboxes] if bboxes is not None else None,
          points=[points] if single else points,
          labels=[labels] if single else labels,
          save=False,
          verbose=False,
        )
      else:
        results = common.sam_model(
          source=source,
          bboxes=bboxes,
          points=points,
          labels=labels,
          save=False,
          verbose=False,
        )
  elif isinstance(common.sam_model, YOLOE):
    names = [texts]
    common.sam_model.set_classes(names, common.sam_model.get_text_pe(names))
    results = common.sam_model.predict(source=source, save=False, verbose=False)

  boxes = []
  masks = []
  for result in results:
    for box, mask in zip(result.boxes, result.masks):
      masks.append(mask.xy[0].tolist())
      boxes.append(box.xywh[0].tolist())

  return {"success": True, "data": {"boxes": boxes, "masks": masks}}


@yolo_router.register("/get_video_frame")
def get_video_frame(handler: RequestHandler):
  video_path = handler.get_query_param("video_path")
  frame_index = handler.get_query_param("frame_index")
  if is_blank(video_path) or is_blank(frame_index):
    return {"success": False, "msg": "运行参数不能为空"}
  frame_index = int(frame_index)

  try:
    import cv2
  except ImportError:
    return {"success": False, "msg": "未安装opencv-python"}

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    return {"success": False, "msg": "打开视频文件失败"}

  frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

  # 设置帧的位置，0为第一帧
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

  ret, frame = cap.read()
  if not ret:
    print(f"Error: could not read frame at index {frame_index}")
    cap.release()
    return {"success": False, "msg": "读取视频帧失败"}

  cap.release()

  # 转换为base64编码
  _, buffer = cv2.imencode(".jpg", frame)
  frame_base64 = base64.b64encode(buffer).decode("utf-8")
  return {
    "success": True,
    "data": {
      "image": "data:/image/jpeg;base64," + frame_base64,
      "frameCount": frame_count,
    }}


@yolo_router.register("/set_sam2_video_model", method="POST")
def set_sam2_video_model(handler: RequestHandler):
  requestBody = handler.read_json()
  model = requestBody["model"]
  config = requestBody["config"]
  if is_blank(model) or is_blank(config):
    return {"success": False, "msg": "运行参数不能为空"}

  if common.set_sam2_video_model(config, model):
    return {"success": True, "msg": "加载模型成功"}
  else:
    return {"success": False, "msg": "加载模型失败"}


@yolo_router.register("/sam2_video_predict", method="POST")
def sam2_video_predict(handler: RequestHandler):
  requestBody = handler.read_json()
  # 输入图片路径
  folderPath = requestBody["folderPath"]
  # 起始帧
  fileIndex = requestBody["fileIndex"]
  # 标记框
  boxs = requestBody["boxs"]

  if is_blank(folderPath):
    return {"success": False, "msg": "运行参数不能为空"}

  if common.sam2_video_predictor is None:
    return {"success": False, "msg": "SAM模型未加载"}

  try:
    from sam2_utils import init_state, mask_to_bbox_normalized
  except ImportError:
    return {"success": False, "msg": "未安装SAM2"}

  inference_state = init_state(predictor=common.sam2_video_predictor,
                               video_path=os.path.normpath(folderPath))
  common.sam2_video_predictor.reset_state(inference_state)

  video_width = inference_state["video_width"]
  video_height = inference_state["video_height"]
  img_prefix = inference_state["img_prefix"]

  for idx, box in enumerate(boxs):
    common.sam2_video_predictor.add_new_points_or_box(
      inference_state=inference_state,
      frame_idx=fileIndex,
      obj_id=idx,
      box=box["bbox"],
    )

  # 收集所有帧的分割结果
  for out_frame_idx, out_obj_ids, out_mask_logits in common.sam2_video_predictor.propagate_in_video(inference_state):
    if out_frame_idx == fileIndex:
      continue
    for i, out_obj_id in enumerate(out_obj_ids):
      mask = (out_mask_logits[i] > 0.0).cpu().numpy()
      h, w = mask.shape[-2:]
      mask = mask.reshape(h, w, 1)
      bbox = mask_to_bbox_normalized(mask, video_width, video_height)
      if bbox is not None:
        with open(f"{folderPath}/DetectLabels/{img_prefix}_{out_frame_idx}.txt", "a") as f:
          f.write(f"{boxs[out_obj_id]['clsIndex']} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

  common.sam2_video_predictor.reset_state(inference_state)
  return {"success": True, "msg": "操作成功"}


@yolo_router.register("/set_sam2_image_model", method="POST")
def set_sam2_image_model(handler: RequestHandler):
  requestBody = handler.read_json()
  model = requestBody["model"]
  config = requestBody["config"]
  if is_blank(model) or is_blank(config):
    return {"success": False, "msg": "运行参数不能为空"}

  if common.set_sam2_image_model(config, model):
    return {"success": True, "msg": "加载模型成功"}
  else:
    return {"success": False, "msg": "加载模型失败"}


@yolo_router.register("/set_clip_feat", method="POST")
def set_clip_feat(handler: RequestHandler):
  try:
    import cv2
    import torch
  except ImportError:
    return {"success": False, "msg": "未安装opencv-python"}

  requestBody = handler.read_json()
  path = requestBody["path"]
  # 标记框
  boxs = requestBody["boxs"]
  if is_blank(path):
    return {"success": False, "msg": "运行参数不能为空"}

  if common.clip_model is None:
    common.load_clip_model()

  global support_feat_tensor

  for box in boxs:
    support_img = cv2.imread(path)
    x, y, w, h = box["bbox"]
    support_crop = support_img[y:y + h, x:x + w]
    support_feat = common.extract_clip_feature(support_crop)
    support_feat_tensor = (box.clsIndex, torch.from_numpy(support_feat).to("cuda"))
  return {"success": True, "msg": "设置特征成功"}


@yolo_router.register("/auto_detect", method="POST")
def auto_detect(handler: RequestHandler):
  try:
    import cv2
    import torch
    from PIL import Image
  except ImportError:
    return {"success": False, "msg": "未安装opencv-python"}

  requestBody = handler.read_json()
  path = requestBody["path"]

  if common.clip_model is None:
    return {"success": False, "msg": "CLIP模型未加载"}

  test_img = cv2.imread(path)
  h_test, w_test = test_img.shape[:2]

  # 5. 网格采样候选点并多线程预处理patch（不推理）
  step = 10
  size = 100

  # 采样点生成（只生成坐标）
  coords = []
  for cy in range(step // 2, h_test, step):
    for cx in range(step // 2, w_test, step):
      coords.append((cx, cy))

  def preprocess_patch(coord):
    cx, cy = coord
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w_test)
    y2 = min(cy + size // 2, h_test)
    crop = test_img[y1:y2, x1:x2]
    if crop.size == 0:
      return None  # 跳过空patch

    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    inp = common.clip_preprocess(pil_img)  # CxHxW tensor
    return (coord, inp, (x1, y1, x2, y2))

  patch_tensors = []
  candidate_points = []

  with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(preprocess_patch, coords))

  for res in results:
    if res is None:
      continue
    (cx, cy), inp, box = res
    patch_tensors.append(inp)
    candidate_points.append((cx, cy, *box))

  if len(patch_tensors) == 0:
    return {"success": False, "msg": "识别图片识别，未读取到任何特征"}

  return {"success": True, "msg": "设置特征成功"}
