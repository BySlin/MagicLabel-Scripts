import queue
import threading
from multiprocessing import Process

from .SSE import ServerSentEvents


class ProcessMsgThread:
  def __init__(self, msg_queue, sse: ServerSentEvents, target, args=()):
    self.read_run = False
    self.process = Process(target=target, args=args)
    self.msg_queue = msg_queue
    self.sse = sse
    self.read_thread = threading.Thread(target=self.read_msg_thread, daemon=True)

  def read_msg_thread(self):
    while self.read_run or self.process.is_alive():
      try:
        message = self.msg_queue.get(timeout=1)
        self.sse.send_raw(message)
      except queue.Empty:
        continue
      except Exception:
        break

  def start(self):
    self.read_run = True
    self.process.start()
    self.read_thread.start()

  def terminate(self):
    self.process.terminate()
    self.read_run = False

  def join(self):
    self.process.join()
    self.read_thread.join()

  def is_alive(self):
    return self.process.is_alive()
