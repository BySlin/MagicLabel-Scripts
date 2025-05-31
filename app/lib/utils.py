import importlib
import subprocess
import sys


def is_not_blank(s: str) -> bool:
  """
  检查字符串是否不是None且不是空字符串（包括只包含空白字符的字符串）

  参数：
  s (str): 要检查的字符串

  返回：
  bool: 如果字符串不是None且不是空字符串则返回True，否则返回False
  """
  return bool(s and s.strip())


def is_blank(s: str) -> bool:
  """
  检查字符串是否是None或空字符串（包括只包含空白字符的字符串）

  参数：
  s (str): 要检查的字符串

  返回：
  bool: 如果字符串是None或空字符串则返回True，否则返回False
  """
  return not is_not_blank(s)


def check_and_install(package):
  """
  检查并安装指定的Python库。
  如果库未安装，将尝试使用pip安装。
  :param package:
  :return:
  """
  try:
    # 尝试导入库
    importlib.import_module(package)
  except ImportError:
    print(f"{package} 未安装, 正在安装...")
    try:
      # 使用 pip 安装库并等待完成
      subprocess.check_call([sys.executable, "-m", "pip", "install", package])
      print(f"{package} 安装成功.")
    except subprocess.CalledProcessError as e:
      print(f"安装 {package} 失败，错误信息: {e}")
