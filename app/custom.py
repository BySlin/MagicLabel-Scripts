import os
import sys


def custom_framework():
  # ultralytics 加载类型 pip 或 custom
  ultralytics_load_type = os.environ.get("ultralyticsLoadType", "pip")
  if ultralytics_load_type == "custom":
    # ultralytics 路径
    ultralytics_path = os.environ["ultralytics"]
    sys.path.insert(0, os.path.normpath(ultralytics_path))

  # yolov5 加载类型 pip 或 custom
  yolov5_load_type = os.environ.get("yolov5LoadType", "pip")
  if yolov5_load_type == "custom":
    # yolov5 路径
    yolov5_path = os.environ["yolov5"]
    sys.path.insert(0, os.path.normpath(yolov5_path))
    sys.path.insert(0, os.path.normpath(os.path.dirname(yolov5_path)))
