import importlib
import subprocess
import sys


def check_and_install(*packages):
  """
  检查并安装指定的Python库。
  如果库未安装，将尝试使用pip安装。
  :param packages: 一个或多个包名字符串
  """
  for package in packages:
    try:
      importlib.import_module(package)
    except ImportError:
      print(f"{package} 未安装, 正在安装...")
      try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} 安装成功.")
      except subprocess.CalledProcessError as e:
        print(f"安装 {package} 失败，错误信息: {e}")
