import os
import sys
from unittest.mock import patch

from utils import check_and_install

# ultralytics 加载类型 pip 或 custom
ultralytics_load_type = os.environ.get("ultralyticsLoadType", "pip")
if ultralytics_load_type == "custom":
  # ultralytics 路径
  ultralytics_path = os.environ["ultralytics"]
  # 加载自定义的 ultralytics 包
  sys.path.insert(0, os.path.normpath(ultralytics_path))

if __name__ == "__main__":
  if any(arg == 'format=engine' for arg in sys.argv):
    check_and_install("tensorrt")

  with patch('ultralytics.utils.checks.ONLINE', True):
    from ultralytics.utils.callbacks.base import default_callbacks
    from ultralytics.cfg import entrypoint


    def on_export_start(export_self):
      if hasattr(export_self, "metadata") and export_self.metadata:
        export_self.metadata["ExportUtils"] = "MagicLabel"


    default_callbacks.get("on_export_start").append(on_export_start)

    entrypoint()
