import json
import os
import re
import sys
import threading
import traceback
from glob import glob
from typing import Union

from lib.QueueWriter import QueueWriter
from lib.SSE import create_sse_msg
from lib.utils import empty_dir


def convert_coco_json(annotations, label_file, w, h, classes, use_segments=False):
  try:
    from ultralytics.data.converter import merge_multi_segment
    import numpy as np
  except ImportError:
    print("Please install ultralytics to use this function.")
    return

  # Write labels file
  bboxes = []
  segments = []
  for ann in annotations:
    if ann["iscrowd"]:
      continue
    cls = ann["category_id"]
    if cls not in classes:
      continue
    # The COCO box format is [top left x, top left y, width, height]
    box = np.array(ann["bbox"], dtype=np.float64)
    if box.size == 4:
      box[:2] += box[2:] / 2  # xy top-left corner to center
      box[[0, 2]] /= w  # normalize x
      box[[1, 3]] /= h  # normalize y
      if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
        continue

    box = [cls] + box.tolist()
    if box not in bboxes:
      bboxes.append(box)
    # Segments
    if use_segments:
      if len(ann["segmentation"]) > 1:
        s = merge_multi_segment(ann["segmentation"])
        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
      else:
        s = [
          j for i in ann["segmentation"] for j in i
        ]  # all segments concatenated
        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
      s = [cls] + s
      if s not in segments:
        segments.append(s)

  box_len = len(bboxes)
  if box_len > 0:
    # Write
    with open(label_file, "a", encoding='utf-8') as file:
      for i in range(box_len):
        line = (
          *(segments[i] if use_segments else bboxes[i]),
        )  # cls, box or segments
        file.write(("%g " * len(line)).rstrip() % line + "\n")

def yaml_load(file):
  try:
    from ultralytics.utils import YAML
    return YAML.load(file)
  except ImportError:
    try:
      from ultralytics.utils import yaml_load as yolo_yaml_load
      return yolo_yaml_load(file)
    except ImportError:
      return None

def parse_args(command: str):
  try:
    from ultralytics.cfg import (
      TASKS,
      MODES,
      merge_equals_args,
      parse_key_value_pair,
      check_dict_alignment,
    )
    from ultralytics.utils import (
      LOGGER,
      checks,
      DEFAULT_CFG_PATH,
      DEFAULT_CFG_DICT,
      DEFAULT_CFG,
    )
  except ImportError:
    print("MagicLabel_Import_Yolo_Error")
    return None

  args = re.findall(r'(\w+="[^"]*"|\w+=\S+)', command)
  full_args_dict = {
    **DEFAULT_CFG_DICT,
    **{k: None for k in TASKS},
    **{k: None for k in MODES},
  }

  overrides = {}  # basic overrides, i.e. imgsz=320
  for a in merge_equals_args(args):  # merge spaces around '=' sign
    if "=" in a:
      try:
        k, v = parse_key_value_pair(a)
        if k == "cfg" and v is not None:  # custom.yaml passed
          LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
          overrides = {
            k: val
            for k, val in yaml_load(checks.check_yaml(v)).items()
            if k != "cfg"
          }
        else:
          overrides[k] = v
      except (NameError, SyntaxError, ValueError, AssertionError) as e:
        check_dict_alignment(full_args_dict, {a: ""}, e)
    else:
      check_dict_alignment(full_args_dict, {a: ""})
  # Check keys
  check_dict_alignment(full_args_dict, overrides)

  # Model
  model = overrides.pop("model", DEFAULT_CFG.model)
  if model is None:
    model = "yolo11n.pt"
  overrides["model"] = model

  return overrides

def predict_task_process(conn, msg_queue):
  sys.stdout = QueueWriter("on_predict_log", msg_queue)
  sys.stderr = QueueWriter("on_predict_log", msg_queue)

  try:
    msg_queue.put(create_sse_msg("on_env_start"))
    import torch

    msg_queue.put(
      create_sse_msg(
        "on_env_end",
        {
          "torchVersion": torch.__version__,
          "cudaVersion": torch.version.cuda,
          "cudnnVersion": torch.backends.cudnn.version(),
          "cudaIsAvailable": torch.cuda.is_available(),
          "cudnnIsAvailable": torch.backends.cudnn.is_available(),
        },
      )
    )
  except ImportError:
    sys.__stdout__.write("MagicLabel_Import_Yolo_Error\n")
    msg_queue.put(create_sse_msg("on_env_error"))
    return

  try:
    from ultralytics import YOLO
    from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS

    IMG_FORMATS.clear()
    VID_FORMATS.clear()
    IMG_FORMATS.update(["jpg", "jpeg", "png", "bmp"])
    load_ultralytics_success = True
  except ImportError:
    YOLO = None
    load_ultralytics_success = False

  try:
    import yolov5
    from yolov5.models.common import AutoShape, Detections
    from yolov5.utils.general import xyxy2xywh

    load_yolov5_success = True
  except ImportError:
    yolov5 = None
    AutoShape = None
    Detections = None
    xyxy2xywh = None
    load_yolov5_success = False

  # 如果ultralytics和yolov5都没有加载成功
  if not load_ultralytics_success and not load_yolov5_success:
    sys.__stdout__.write("MagicLabel_Import_Yolo_Error\n")
    return

  # 加载sahi是否成功
  load_sahi_success = False
  try:
    from sahi import AutoDetectionModel
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from sahi.models.yolov5 import Yolov5DetectionModel
    from sahi.predict import get_sliced_prediction

    load_sahi_success = True
  except ImportError:
    load_sahi_success = False
    AutoDetectionModel = None
    UltralyticsDetectionModel = None
    Yolov5DetectionModel = None
    get_sliced_prediction = None
    sys.__stdout__.write("MagicLabel_Import_Sahi_Error\n")

  def predict_thread(framework_: str, cwd: str,
                     yolo_model: Union[YOLO, AutoShape, UltralyticsDetectionModel, Yolov5DetectionModel],
                     predict_params, sahi_settings, skip_exists_annotation_file
  ):
    try:
      os.chdir(cwd)

      msg_queue.put(create_sse_msg("on_predict_start"))

      is_sahi_model = load_sahi_success and (
        isinstance(yolo_model, UltralyticsDetectionModel) or isinstance(yolo_model, Yolov5DetectionModel))

      task = "detect"
      if is_sahi_model:
        if framework_ == "ultralytics":
          task = yolo_model.model.task
        model_names = yolo_model.model.names
      else:
        if framework_ == "ultralytics":
          task = yolo_model.task
        model_names = yolo_model.names

      use_segments = False
      use_detect = False
      use_obb = False
      mode = "Detect"
      if task == "detect":
        mode = "Detect"
        use_detect = True
      elif task == "obb":
        mode = "OBB"
        use_segments = True
        use_obb = True
      elif task == "segment":
        mode = "Seg"
        use_segments = True
      elif task == "pose":
        mode = "Pose"
      elif task == "classify":
        mode = "Cls"

      config_dir = os.path.normpath(os.path.join(cwd, f"{mode}Labels"))

      source = predict_params["source"]
      source_is_dir = os.path.isdir(source)
      if source_is_dir and not skip_exists_annotation_file:
        empty_dir(config_dir)

      if not os.path.exists(config_dir):
        os.makedirs(config_dir)

      classes_txt_path = os.path.join(config_dir, "classes.txt")
      # 将类别名称写入 classes.txt
      with open(classes_txt_path, "w", encoding='utf-8') as file:
        for idx, name in model_names.items():
          file.write(name + "\n")

      sahi_ = sahi_settings["sahi"]
      slice_width_ = sahi_settings["sahi_slice_width"]
      slice_height_ = sahi_settings["sahi_slice_height"]
      overlap_width_ratio_ = sahi_settings["sahi_overlap_width_ratio"]
      overlap_height_ratio_ = sahi_settings["sahi_overlap_height_ratio"]

      if source_is_dir:
        image_files = (
          glob(os.path.join(source, "*.jpg"))
          + glob(os.path.join(source, "*.jpeg"))
          + glob(os.path.join(source, "*.png"))
          + glob(os.path.join(source, "*.bmp"))
        )
        # 是否开启了跳过已存在标注文件
        if skip_exists_annotation_file:
          image_files = [
            image_file for image_file in image_files
            if not os.path.exists(
              os.path.join(
                config_dir,
                f"{os.path.splitext(os.path.basename(image_file))[0]}.txt",
              )
            )
          ]
      else:
        image_files = [source]

      image_len = len(image_files)
      conf_ = predict_params["conf"]
      iou_ = predict_params["iou"]
      device_ = predict_params["device"]
      classes_ = predict_params["classes"]
      imgsz_ = predict_params["imgsz"]

      is_stop = False
      if sahi_ and is_sahi_model:
        if use_detect or use_segments:
          yolo_model.confidence_threshold = conf_
          yolo_model.mask_threshold = conf_
          yolo_model.device = device_
          for i, image_file in enumerate(image_files):
            prediction_result = get_sliced_prediction(
              detection_model=yolo_model,
              image=image_file,
              slice_width=slice_width_,
              slice_height=slice_height_,
              overlap_width_ratio=overlap_width_ratio_,
              overlap_height_ratio=overlap_height_ratio_,
              postprocess_type="NMS" if use_obb else "GREEDYNMM",
              verbose=0,
            )
            width = prediction_result.image_width
            height = prediction_result.image_height
            durations_in_seconds = prediction_result.durations_in_seconds
            annotations = prediction_result.to_coco_annotations()
            label_file = os.path.join(
              config_dir,
              f"{os.path.splitext(os.path.basename(image_file))[0]}.txt",
            )
            convert_coco_json(
              annotations,
              label_file,
              width,
              height,
              classes_,
              use_segments=use_segments,
            )
            print(
              f'image {i + 1}/{image_len} {image_file}: slice: {durations_in_seconds["slice"]:.2f}s, prediction: {durations_in_seconds["prediction"]:.2f}s'
            )

            if not is_predicting:
              is_stop = True
              break
        else:
          print("sahi只支持矩形框、旋转框、多边形")
      else:
        if sahi_:
          print("未安装sahi,将使用yolo执行,如需使用请安装 pip install sahi")

        yolov5_model: [AutoShape, None] = None
        if is_sahi_model:
          if framework_ == "ultralytics":
            results = yolo_model.model.predict(
              **predict_params, stream=True, verbose=True
            )
          elif framework_ == "yolov5":
            yolov5_model = yolo_model.model
        else:
          if framework_ == "ultralytics":
            results = yolo_model.predict(
              **predict_params, stream=True, verbose=True
            )
          elif framework_ == "yolov5":
            yolov5_model = yolo_model

        if framework_ == "ultralytics":
          for result in results:
            txt_file = os.path.join(
              config_dir,
              f"{os.path.splitext(os.path.basename(result.path))[0]}.txt",
            )
            if os.path.exists(txt_file):
              if source_is_dir and skip_exists_annotation_file:
                continue
              else:
                os.remove(txt_file)
            if mode == "Cls":
              # 如果置信度大于等于阈值，则写入文件
              if result.probs.top1conf.item() >= conf_:
                with open(txt_file, "w", encoding='utf-8') as file:
                  file.write(
                    f"{result.probs.top1} {model_names[result.probs.top1]}\n"
                  )
            else:
              result.save_txt(txt_file)
            if not is_predicting:
              is_stop = True
              break
        elif framework_ == "yolov5":
          yolov5_model.conf = conf_
          yolov5_model.iou = iou_
          yolov5_model.classes = classes_
          yolov5_model.to(device_)
          for i, image_file in enumerate(image_files):
            results: Detections = yolov5_model(image_file, size=imgsz_)

            s, crops = "", []
            im = results.ims[0]
            pred = results.pred[0]
            s += f"image {i + 1}/{image_len} {image_file}: {im.shape[0]}x{im.shape[1]} "  # string
            if pred.shape[0]:
              for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                s += f"{n} {model_names[int(c)]}{'s' * (n > 1)}, "  # add to string
            else:
              s += "(no detections)"
            s += f", {results.t[0]:.1f}ms"
            print(s)

            gn = torch.tensor(im.shape)[
              [1, 0, 1, 0]
            ]  # normalization gain whwh
            for *xyxy, conf, cls in reversed(pred):
              xywh = (
                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                .view(-1)
                .tolist()
              )  # normalized xywh
              line = (cls, *xywh)  # label format
              txt_file = os.path.join(
                config_dir,
                f"{os.path.splitext(os.path.basename(image_file))[0]}.txt",
              )
              with open(txt_file, "a", encoding='utf-8') as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")
            if not is_predicting:
              is_stop = True
              break

      msg_queue.put(
        create_sse_msg("on_predict_end", {"mode": mode, "isStop": is_stop, "singleFile": image_len == 1})
      )
    except:
      traceback_info = traceback.format_exc()
      print(traceback_info)
      msg_queue.put(create_sse_msg("on_predict_error"))

  model: Union[
    YOLO, AutoShape, UltralyticsDetectionModel, Yolov5DetectionModel, None
  ] = None
  is_predicting = False
  # 在子进程中等待接收数据
  while True:
    try:
      msg = conn.recv()  # 从管道接收数据
      event_data = json.loads(msg)
      event_ = event_data["event"]
      if event_ == "set_predict_model":
        model_path = event_data["model"]
        is_pt_model = model_path.endswith(".pt")
        framework = event_data["framework"]
        task = event_data["task"]
        is_predicting = False
        if load_sahi_success and is_pt_model:
          if framework == "ultralytics":
            model = AutoDetectionModel.from_pretrained(
              model_type="ultralytics",
              model_path=model_path,
            )
            msg_queue.put(
              create_sse_msg("on_predict_model_set", model.model.names)
            )

          elif framework == "yolov5":
            model = AutoDetectionModel.from_pretrained(
              model_type="yolov5",
              model_path=model_path,
            )
            msg_queue.put(
              create_sse_msg("on_predict_model_set", model.model.names)
            )

        else:

          if framework == "ultralytics":
            if is_pt_model:
              model = YOLO(model_path)
            else:
              model = YOLO(model_path, task=task)

            msg_queue.put(
              create_sse_msg("on_predict_model_set", model.names)
            )

          elif framework == "yolov5":
            model = yolov5.load(model_path)
            msg_queue.put(
              create_sse_msg("on_predict_model_set", model.model.names)
            )

      elif event_ == "predict":
        framework = event_data["framework"]
        params = parse_args(event_data["command"])

        if params is None:
          sys.__stdout__.write("MagicLabel_Import_Yolo_Error\n")
        else:
          is_predicting = True
          threading.Thread(
            target=predict_thread,
            args=(
              framework,
              event_data["cwd"],
              model,
              params,
              event_data["sahiSettings"],
              event_data["skipExistsAnnotationFile"],
            ),
          ).start()
      elif event_ == "stop_predict":
        is_predicting = False
    except:
      break
