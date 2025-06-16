import queue
import threading
from multiprocessing import Process

from .SSE import ServerSentEvents


class ProcessMsgThread:
  def __init__(self, msg_queue, sse: ServerSentEvents, target, args=()):
    # 初始化方法
    self.read_run = False  # 初始化读取运行标志为False
    self.process = Process(target=target, args=args)  # 创建一个新的进程，目标函数为target，参数为args
    self.msg_queue = msg_queue  # 初始化消息队列
    self.sse = sse  # 初始化服务器发送事件对象
    self.read_thread = threading.Thread(target=self.read_msg_thread, daemon=True)  # 创建一个守护线程，目标函数为read_msg_thread

  def read_msg_thread(self):
    # 读取消息线程的方法
    while self.read_run or self.process.is_alive():
      # 当读取运行标志为True或进程仍在运行时，循环继续
      try:
        message = self.msg_queue.get(timeout=1)  # 从消息队列中获取消息，超时时间为1秒
        self.sse.send_raw(message)  # 通过服务器发送事件对象发送原始消息
      except queue.Empty:
        continue  # 如果队列为空，继续下一次循环
      except Exception:
        break  # 如果发生其他异常，退出循环

  def start(self):
    # 启动方法
    self.read_run = True  # 设置读取运行标志为True
    self.process.start()  # 启动进程
    self.read_thread.start()  # 启动读取消息线程

  def terminate(self):
    # 终止方法
    self.process.terminate()  # 终止进程
    self.read_run = False  # 设置读取运行标志为False

  def join(self):
    # 等待方法
    self.process.join()  # 等待进程结束
    self.read_thread.join()  # 等待读取消息线程结束

  def is_alive(self):
    # 检查是否存活方法
    return self.process.is_alive()  # 返回进程是否仍在运行
