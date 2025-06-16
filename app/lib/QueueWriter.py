import json

from .SSE import create_sse_msg


# 日志转发到队列
class QueueWriter:
  # 初始化QueueWriter类，传入_event和_queue参数
  def __init__(self, _event, _queue):
    self.event = _event
    self.queue = _queue

  # 将消息写入队列
  def write(self, msg):
    # 将消息转换为sse消息，并放入队列中
    self.queue.put(create_sse_msg(self.event, json.dumps(msg)))

  # 刷新队列
  def flush(self):
    pass
