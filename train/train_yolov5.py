import os
import sys

# yolov5 加载类型 pip 或 custom
yolov5_load_type = os.environ.get("yolov5LoadType", "pip")
if yolov5_load_type == "custom":
  # yolov5 路径
  yolov5_path = os.environ["yolov5"]
  # 加载 yolov5 路径
  sys.path.insert(0, os.path.normpath(yolov5_path))
  sys.path.insert(0, os.path.normpath(os.path.dirname(yolov5_path)))

if __name__ == "__main__":
  from yolov5.train import parse_opt, main as train_main

  opt = parse_opt()
  train_main(opt)
