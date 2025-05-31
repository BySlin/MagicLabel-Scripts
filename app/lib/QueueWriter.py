import json

from .SSE import create_sse_msg


# 日志转发到队列
class QueueWriter:
  def __init__(self, _event, _queue):
    self.event = _event
    self.queue = _queue

  def write(self, msg):
    self.queue.put(create_sse_msg(self.event, json.dumps(msg)))

  def flush(self):
    pass
