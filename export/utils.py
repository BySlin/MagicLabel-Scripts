import importlib
import os
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


def exec_python_command(*params):
  """
  用python执行指定的命令。
  :param params: 命令参数
  """
  try:
    subprocess.check_call([sys.executable, *params])
    return True
  except subprocess.CalledProcessError as e:
    return False


def exec_command(*params):
  """
  执行指定的命令。
  :param params: 命令参数
  """
  try:
    subprocess.check_call(params)
    return True
  except subprocess.CalledProcessError as e:
    return False


def rename_file_overwrite(src, dst):
  """
  重命名文件，如果目标文件存在则先删除，保证覆盖。
  """
  if os.path.exists(dst):
    os.remove(dst)  # 删除目标文件
  os.rename(src, dst)  # 重命名
