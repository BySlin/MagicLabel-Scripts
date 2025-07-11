import importlib.util
import os
import shutil
import subprocess
import sys


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


def kill_process_tree(pid: int, include_parent=True):
  """
  结束指定pid及其所有子进程，按顺序先子进程再父进程

  参数：
  pid (int): 要结束的进程的PID
  include_parent (bool): 是否包含父进程，默认为True
  """
  check_and_install("psutil")
  import psutil
  try:
    parent = psutil.Process(pid)
  except psutil.NoSuchProcess:
    print(f"Process {pid} 不存在")
    return

  children = parent.children(recursive=True)
  # 先结束所有子进程
  for p in children:
    try:
      print(f"Terminating child process {p.pid} ({p.name()})")
      p.terminate()
    except Exception as e:
      print(f"Failed to terminate child process {p.pid}: {e}")

  # 等待子进程结束
  gone, alive = psutil.wait_procs(children, timeout=3)
  for p in alive:
    try:
      print(f"Killing child process {p.pid} ({p.name()})")
      p.kill()
    except Exception as e:
      print(f"Failed to kill child process {p.pid}: {e}")

  # 结束父进程
  if include_parent:
    try:
      print(f"Terminating parent process {parent.pid} ({parent.name()})")
      parent.terminate()
      parent.wait(timeout=3)
    except psutil.NoSuchProcess:
      pass
    except psutil.TimeoutExpired:
      try:
        print(f"Killing parent process {parent.pid} ({parent.name()})")
        parent.kill()
      except Exception as e:
        print(f"Failed to kill parent process {parent.pid}: {e}")
    except Exception as e:
      print(f"Failed to terminate parent process {parent.pid}: {e}")


def check_and_kill_port_process_and_children(port: int):
  check_and_install("psutil")
  import psutil
  """
  检测端口占用，结束占用进程及其所有子进程

  参数：
  port (int): 要检测的端口号
  """
  for conn in psutil.net_connections():
    if conn.laddr and conn.laddr.port == port:
      pid = conn.pid
      if pid:
        print(f"端口 {port} 被 PID {pid} 占用，准备结束该进程及其子进程")
        kill_process_tree(pid)
      break
