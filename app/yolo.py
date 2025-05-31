import json
import os.path
import threading
import time

import common
from lib.SSE import create_sse_msg
from lib.SimpleHttpServer import Router
from lib.process_msg_thread import ProcessMsgThread
from lib.utils import is_blank
from yolo_task import export_task_process, model_predict_task_process

yolo_router = Router("/api/yolo")


@yolo_router.register("/sse_events")
def sse_events(handler):
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
def set_predict_model(handler):
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
def predict(handler):
  requestBody = handler.read_json()
  command = requestBody["command"]
  cwd = requestBody["cwd"]
  framework = requestBody["framework"]
  sahiSettings = requestBody["sahiSettings"]

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
      }
    )
  )

  return {"success": True, "msg": "操作成功"}


@yolo_router.register("/stop_predict")
def stop_predict(handler):
  common.predict_main_conn.send(
    json.dumps(
      {
        "event": "stop_predict",
      }
    )
  )

  return {"success": True, "msg": "操作成功"}

@yolo_router.register("/export", method="POST")
def export(handler):
  requestBody = handler.read_json()
  command = requestBody["command"]
  outputPath = requestBody["outputPath"]
  framework = requestBody["framework"]

  if is_blank(command) or is_blank(outputPath) or is_blank(framework):
    return {"success": False, "msg": "运行参数不能为空"}

  if command is not None and outputPath is not None and framework is not None:
    if common.export_process is None or not common.export_process.is_alive():
      common.export_process = ProcessMsgThread(
        msg_queue=common.export_msg_queue,
        sse=common.sse_events,
        target=export_task_process,
        args=(
          framework,
          outputPath,
          command,
          common.export_msg_queue,
        ),
      )
      common.export_process.start()

      return {"success": True, "msg": "操作成功"}
    else:
      return {"success": True, "msg": "导出进程正在运行中"}


@yolo_router.register("/stop_export")
def stop_export(handler):
  if common.export_process is not None and common.export_process.is_alive():
    common.export_process.terminate()
    common.export_process = None

  common.sse_events.send_raw(create_sse_msg("on_export_end", {"isStop": True}))
  print("AutoLabel_Export_End")
  return {"success": True, "msg": "操作成功"}


@yolo_router.register("/model_predict", method="POST")
def model_predict(handler):
  requestBody = handler.read_json()
  command = requestBody["command"]
  source = requestBody["source"]
  framework = requestBody["framework"]

  if is_blank(command) or is_blank(source) or is_blank(framework):
    return {"success": False, "msg": "运行参数不能为空"}

  if command is not None and source is not None and framework is not None:
    if common.model_predict_process is None or not common.model_predict_process.is_alive():
      common.model_predict_process = ProcessMsgThread(
        msg_queue=common.model_predict_msg_queue,
        sse=common.sse_events,
        target=model_predict_task_process,
        args=(
          framework,
          source,
          command,
          common.model_predict_msg_queue,
        ),
      )
      common.model_predict_process.start()

      return {"success": True, "msg": "操作成功"}
    else:
      return {"success": True, "msg": "模型验证进程正在运行中"}


@yolo_router.register("/stop_model_predict")
def stop_model_predict(handler):
  if common.model_predict_process is not None and common.model_predict_process.is_alive():
    common.model_predict_process.terminate()
    common.model_predict_process = None

  common.sse_events.send_raw(create_sse_msg("on_model_predict_end", {"isStop": True}))
  print("AutoLabel_Model_Predict_End")
  return {"success": True, "msg": "操作成功"}


@yolo_router.register("/set_sam_model", method="POST")
def set_sam_model(handler):
  requestBody = handler.read_json()
  model = requestBody["model"]

  if is_blank(model):
    return {"success": False, "msg": "运行参数不能为空"}

  if not os.path.isabs(model):
    model = os.path.join(common.models_directory, model)

  if common.set_sam_model(model):
    return {"success": True, "msg": "切换模型成功"}
  else:
    return {"success": False, "msg": "设置SAM模型失败"}


@yolo_router.register("/sam_model_predict", method="POST")
def sam_model_predict(handler):
  requestBody = handler.read_json()
  source = requestBody["source"]
  data = requestBody["data"]
  single = requestBody["single"]

  if is_blank(source):
    return {"success": False, "msg": "运行参数不能为空"}

  if common.sam_model is None:
    return {"success": False, "msg": "SAM模型未加载"}

  try:
    from ultralytics import FastSAM, SAM
  except ImportError:
    FastSAM = None
    SAM = None

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
  boxes = []
  masks = []
  for result in results:
    for box, mask in zip(result.boxes, result.masks):
      masks.append(mask.xy[0].tolist())
      boxes.append(box.xywh[0].tolist())

  return {"success": True, "data": {"boxes": boxes, "masks": masks}}
