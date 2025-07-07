import os
import sys

# ultralytics 加载类型 pip 或 custom
ultralytics_load_type = os.environ.get("ultralyticsLoadType", "pip")
if ultralytics_load_type == "custom":
  # ultralytics 路径
  ultralytics_path = os.environ["ultralytics"]
  # 加载自定义的 ultralytics 包
  sys.path.insert(0, os.path.normpath(ultralytics_path))

# yolov5 加载类型 pip 或 custom
yolov5_load_type = os.environ.get("yolov5LoadType", "pip")
if yolov5_load_type == "custom":
  # yolov5 路径
  yolov5_path = os.environ["yolov5"]
  # 加载 yolov5 路径
  sys.path.insert(0, os.path.normpath(yolov5_path))
  sys.path.insert(0, os.path.normpath(os.path.dirname(yolov5_path)))

import argparse

import cv2
import numpy as np

from utils import check_and_install
from yolo_window import win_name, is_image_file, is_windows, get_hwnd, activate_window, get_window_rect, show_image, \
  str_to_int_if_possible


def model_predict(task, framework, model, source: str, conf, iou, imgsz):
  if framework == "ultralytics":
    from ultralytics import YOLO
    yolo_model = YOLO(model, task=task)
  elif framework == "yolov5":
    import yolov5
    yolo_model = yolov5.load(model)
    yolo_model.conf = conf
    yolo_model.iou = iou

  cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(win_name, 1280, 719)

  is_screen = source.startswith("screen")
  is_window = source.startswith("window")
  is_rect = source.startswith("rect")
  is_image = is_image_file(source)
  monitor_number = 1
  mon = {
    "left": 0,
    "top": 0,
    "width": 1920,
    "height": 1080,
  }

  if is_screen:
    check_and_install("mss")
    src, index, _ = source.split(":")
    source = src
    monitor_number = int(index) + 1
  elif is_window:
    check_and_install("mss")
    if is_windows():
      window_name = source[7:]
      hwnd = get_hwnd(window_name)
      activate_window(hwnd)
      x, y, right, bottom = get_window_rect(hwnd)
      mon = {
        "left": x,
        "top": y,
        "width": right - x,
        "height": bottom - y,
      }
    else:
      source = 'screen'
      monitor_number = 1
  elif is_rect:
    check_and_install("mss")
    _, x, y, w, h = source.split("-")
    mon = {
      "left": int(x),
      "top": int(y),
      "width": int(w),
      "height": int(h),
    }

  def predict():
    if framework == "ultralytics":
      results = yolo_model.predict(
        source=source,
        imgsz=imgsz,
        iou=iou,
        conf=conf,
        stream=True,
        save=False,
      )
      for result in results:
        img = result.plot()
        try:
          show_image(img)
        except:
          break
    elif framework == "yolov5":
      if not is_image and not is_screen and not is_window and not is_rect:
        cap = cv2.VideoCapture(str_to_int_if_possible(source))
        while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
            break
          results = yolo_model(frame, size=imgsz)
          results.print()
          img = results.render()[0]
          try:
            show_image(img)
          except:
            break
      else:
        results = yolo_model(source, size=imgsz)
        results.print()
        img = results.render()[0]
        try:
          show_image(img)
        except:
          pass

  if is_window or is_screen or is_rect:
    from mss import mss
    with mss() as sct:
      if is_screen:
        mon = sct.monitors[monitor_number]
      while True:
        # 检测窗口是否关闭
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
          break
        source = np.asarray(sct.grab(mon))[:, :, :3]  # BGRA to BGR
        predict()
  else:
    predict()
  cv2.waitKey(0)
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(description="MagicLabel")
  parser.add_argument('--task', help='任务类型', default="detect")
  parser.add_argument('--framework', help='执行框架', default="ultralytics")
  parser.add_argument('--model', help='模型路径')
  parser.add_argument('--source', help='识别文件')
  parser.add_argument('--conf', help='置信度', type=float, default=0.25)
  parser.add_argument('--iou', help='iou', type=float, default=0.45)
  parser.add_argument('--imgsz', help='图片大小', type=str, default='640')

  args = parser.parse_args()
  task = args.task
  framework = args.framework
  model = args.model
  source = args.source
  conf = args.conf
  iou = args.iou
  imgsz = args.imgsz
  if isinstance(imgsz, str):
    imgsz = [int(x) for x in imgsz.split(',')]
    if len(imgsz) == 1:
      imgsz = [imgsz[0], imgsz[0]]

  model_predict(task, framework, model, source, conf, iou, imgsz)


if __name__ == "__main__":
  main()
