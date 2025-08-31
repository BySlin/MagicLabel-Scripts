import base64
import concurrent.futures
import json
import os.path
import threading
import time
from glob import glob
from typing import Union, TYPE_CHECKING

import common
from lib.SSE import create_sse_msg
from lib.SimpleHttpServer import Router, RequestHandler
from lib.utils import is_blank, is_not_blank

yolo_router = Router("/api/yolo")

if TYPE_CHECKING:
  from torch import Tensor

cls_in_support_feat_tensor: Union[dict[int, "Tensor"], None] = None

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
  coverMode = requestBody.get("coverMode", 0)

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
        "coverMode": coverMode
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

  global cls_in_support_feat_tensor
  cls_in_support_feat_tensor = {}
  for box in boxs:
    support_img = cv2.imread(path)
    x, y, w, h = box["bbox"]
    support_crop = support_img[y:y + h, x:x + w]
    support_feat = common.extract_clip_feature(support_crop)
    cls_in_support_feat_tensor[box["clsIndex"]] = (w, h, support_feat)
  return {"success": True, "msg": "设置特征成功"}


@yolo_router.register("/auto_detect", method="POST")
def auto_detect(handler: RequestHandler):
  try:
    import cv2
    import torch
    import numpy as np
    from PIL import Image
    from sam2_utils import mask_to_bbox_normalized
  except ImportError:
    return {"success": False, "msg": "未安装opencv-python"}

  requestBody = handler.read_json()
  # 输入图片路径
  folderPath = requestBody["folderPath"]
  # 图片路径
  path = requestBody["path"]

  # 获取图片文件名不带后缀
  name_without_ext = os.path.splitext(os.path.basename(path))[0]

  if common.clip_model is None:
    return {"success": False, "msg": "CLIP模型未加载"}

  if common.sam2_image_predictor is None:
    return {"success": False, "msg": "SAM2模型未加载"}

  test_img = cv2.imread(path)
  h_test, w_test = test_img.shape[:2]

  step = requestBody.get("step", 10)
  size = requestBody.get("size", 100)
  K = requestBody.get("k", 3)  # 多点提示数
  sim_threshold = requestBody.get("sim_threshold", 0.8)  # 相似度阈值，不满足停止
  max_iters = requestBody.get("max_iters", 10)  # 最大迭代识别目标数
  fill_color = (0, 0, 255)  # BGR红色填充

  def preprocess_patch(coord, img):
    cx, cy = coord
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, img.shape[1])
    y2 = min(cy + size // 2, img.shape[0])
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
      return None
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    inp = common.clip_preprocess(pil_img)
    return (coord, inp, (x1, y1, x2, y2))

  for key in cls_in_support_feat_tensor:
    w, h, support_feat = cls_in_support_feat_tensor[key]
    size = min(size, w)
    step = min(step, max(w_test // w, 20))
    # 初始化填充掩码（记录已识别区域）
    filled_mask = np.zeros((h_test, w_test), dtype=np.uint8)

    for iter_idx in range(max_iters):
      coords = []
      for cy in range(step // 2, h_test, step):
        for cx in range(step // 2, w_test, step):
          if filled_mask[cy, cx] == 1:  # 跳过已填充区域
            continue
          coords.append((cx, cy))

      if len(coords) == 0:
        break

      patch_tensors = []
      candidate_points = []

      with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda c: preprocess_patch(c, test_img), coords))

      for res in results:
        if res is None:
          continue
        (cx, cy), inp, box = res
        patch_tensors.append(inp)
        candidate_points.append((cx, cy, *box))

      if len(patch_tensors) == 0:
        break

      batch = torch.stack(patch_tensors).to("cuda")

      with torch.no_grad():
        feats = common.clip_model.encode_image(batch)
        feats /= feats.norm(dim=-1, keepdim=True)

      sim_scores = feats @ support_feat.T  # [N, M]
      candidate_scores = sim_scores.cpu().numpy().squeeze()

      max_score = candidate_scores.max()
      if max_score < sim_threshold:
        break

      topk_indices = np.argsort(candidate_scores)[-K:]
      topk_points = [candidate_points[i] for i in topk_indices]
      input_points = np.array([[pt[0], pt[1]] for pt in topk_points])
      input_labels = np.ones(len(input_points), dtype=np.int32)

      # 通过SAM2预测分割
      common.sam2_image_predictor.set_image(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
      masks, scores, logits = common.sam2_image_predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
      )
      mask = masks[0]
      h, w = mask.shape[-2:]
      mask = mask.reshape(h, w, 1)

      mask_uint8 = (mask * 255).astype(np.uint8)
      if mask_uint8.shape != (h_test, w_test):
        mask_uint8 = cv2.resize(mask_uint8, (w_test, h_test), interpolation=cv2.INTER_NEAREST)
      mask_binary = mask_uint8 > 128

      bbox = mask_to_bbox_normalized(mask, w_test, h_test)
      if bbox is not None:
        # 兜底相似度检测
        x1, y1, w_box, h_box = cv2.boundingRect(mask_binary.astype(np.uint8))
        x2, y2 = x1 + w_box, y1 + h_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_test, x2), min(h_test, y2)
        cropped_img = test_img[y1:y2, x1:x2]
        if cropped_img.size == 0:
          continue
        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        pil_cropped = Image.fromarray(cropped_img_rgb)
        inp_cropped = common.clip_preprocess(pil_cropped).unsqueeze(0).to("cuda")
        with torch.no_grad():
          cropped_feat = common.clip_model.encode_image(inp_cropped)
          cropped_feat /= cropped_feat.norm(dim=-1, keepdim=True)
        sim_cropped = (cropped_feat @ support_feat.T).item()
        if sim_cropped < sim_threshold:
          break

        # 写入标签
        with open(f"{folderPath}/DetectLabels/{name_without_ext}.txt", "a") as f:
          f.write(f"{key} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

      # 找mask轮廓
      contours, _ = cv2.findContours(mask_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if len(contours) == 0:
        continue

      # 找最小外接矩形
      x, y, w, h = cv2.boundingRect(contours[0])
      # 用矩形区域更新 filled_mask
      filled_mask[y:y + h, x:x + w] = 1
      # 矩形区域纯色填充
      test_img[y:y + h, x:x + w] = fill_color
      # 用纯色覆盖目标区域
      test_img[mask_binary] = fill_color

  return {"success": True, "msg": "预测结束"}


@yolo_router.register("/find_template_detect", method="POST")
def find_template_detect(handler: RequestHandler):
    try:
        import cv2
        import numpy as np
        from template_search import TemplateSearch
    except ImportError:
        return {"success": False, "msg": "未安装opencv-python"}
    data = handler.read_json()
    template_dir = data["templateDir"]
    # 输入图片路径
    folderPath = data["folderPath"]
    if is_blank(template_dir):
        return {"success": False, "msg": "未输入模板目录"}

    if is_blank(folderPath):
        return {"success": False, "msg": "未输入图片目录"}

    template_image_files = (
            glob(os.path.join(template_dir, "*.jpg"))
            + glob(os.path.join(template_dir, "*.jpeg"))
            + glob(os.path.join(template_dir, "*.png"))
            + glob(os.path.join(template_dir, "*.bmp"))
    )

    image_files = (
            glob(os.path.join(folderPath, "*.jpg"))
            + glob(os.path.join(folderPath, "*.jpeg"))
            + glob(os.path.join(folderPath, "*.png"))
            + glob(os.path.join(folderPath, "*.bmp"))
    )
    for image_file in image_files:
        for template_image_file in template_image_files:
            # 获取template_image_file的文件名
            template_image_file_name = os.path.basename(template_image_file)
            # 文件名按下划线分割
            template_parts = template_image_file_name.split('_')
            if len(template_parts) > 0:
                clsIndex = template_parts[1]
            else:
                clsIndex = "0"
            results = TemplateSearch.find_image(image_file, template_image_file)
            for result in results:
                with open(f"{folderPath}/DetectLabels/{os.path.splitext(os.path.basename(image_file))[0]}.txt",
                          "a") as f:
                    f.write(
                        f"{clsIndex} {result['n_centerX']} {result['n_centerY']} {result['n_width']} {result['n_height']}\n")

    return {"success": True, "msg": "找图结束"}