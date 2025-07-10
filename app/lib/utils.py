import importlib.util
import os
import shutil


def module_exists(module_name):
  """
  检查模块是否存在

  参数：
  module_name (str): 模块名称

  返回：
  bool: 如果模块存在则返回True，否则返回False
  """
  return importlib.util.find_spec(module_name) is not None


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


def empty_dir(dir_path):
  """
  如果目录不存在，则创建目录；
  如果目录存在，则删除该目录下所有内容（文件和子目录），保留该目录本身。
  """
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    return

  if not os.path.isdir(dir_path):
    raise NotADirectoryError(f"{dir_path} 不是一个目录")

  for entry in os.listdir(dir_path):
    entry_path = os.path.join(dir_path, entry)
    try:
      if os.path.isfile(entry_path) or os.path.islink(entry_path):
        os.unlink(entry_path)  # 删除文件或符号链接
      elif os.path.isdir(entry_path):
        shutil.rmtree(entry_path)  # 递归删除子目录及其内容
    except Exception as e:
      print(f"删除 {entry_path} 失败: {e}")
