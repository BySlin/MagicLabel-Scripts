import os
import sys
from unittest.mock import patch

# ultralytics 加载类型 pip 或 custom
ultralytics_load_type = os.environ.get("ultralyticsLoadType", "pip")
if ultralytics_load_type == "custom":
  # ultralytics 路径
  ultralytics_path = os.environ["ultralytics"]
  # 加载自定义的 ultralytics 包
  sys.path.insert(0, os.path.normpath(ultralytics_path))

if __name__ == "__main__":
  with patch('ultralytics.utils.checks.ONLINE', True):
    from ultralytics.cfg import entrypoint

    entrypoint()
