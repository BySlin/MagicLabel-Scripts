import json
import uuid
from queue import Queue, Full, Empty
from typing import List, Dict, Union


# 创建一个唯一的SSE消息
def create_sse_msg(event: str, data: Union[str, Dict, None] = None):
  # 生成一个唯一的event_id
  event_id = str(uuid.uuid4())
  # 创建一个消息，包含event_id和event
  message = f"id: {event_id}\n"
  message += f"event: {event}\n"
  # 判断data的类型，如果是字典，则将字典转换为json格式，否则将data转换为字符串
  if isinstance(data, dict):
    message += f"data: {json.dumps(data)}\n\n"
  elif isinstance(data, str):
    message += f"data: {data}\n\n"
  else:
    message += f"data: {event}\n\n"
  # 返回消息
  return message


# 服务器发送事件类
class ServerSentEvents:
  # 消息ID
  msg_id: int = 0
  # 监听器列表
  listeners: List[Queue] = []

  # 响应请求
  def response(self):

    # 处理流
    def stream(handle):
      handle.send_response(200)
      handle.send_header("Content-Type", "text/event-stream; charset=utf-8")
      handle.send_header("Cache-Control", "no-cache")
      handle.send_header("Connection", "keep-alive")
      handle.end_headers()

      # 创建一个队列，用于存储消息
      queue = Queue(5)
      self.listeners.append(queue)
      try:
        while True:
          try:
            # 从队列中获取消息
            msg = queue.get(timeout=1)
            # 发送消息
            handle.wfile.write(msg.encode("utf-8"))
            handle.wfile.flush()
          except Empty:
            try:
              # 发送ping消息
              handle.wfile.write(create_sse_msg("ping").encode("utf-8"))
              handle.wfile.flush()
            except:
              break
            continue
          except:
            break
      finally:
        # 移除监听器
        self.listeners.remove(queue)

    return stream

  # 发送消息
  def send(self, payload: Dict = None, event: str = "data"):
    self.msg_id += 1
    # 将消息转换为json格式
    msg_str = json.dumps(payload) if payload else "{}"
    # 创建消息
    msg = f"id: {self.msg_id}\nevent: {event}\ndata: {msg_str}\n\n"

    # 发送消息给所有监听器
    for i in reversed(range(len(self.listeners))):
      try:
        self.listeners[i].put_nowait(msg)
      except Full:
        # 如果队列已满，则移除监听器
        del self.listeners[i]

  # 发送原始消息
  def send_raw(self, msg: str):
    # 发送消息给所有监听器
    for i in reversed(range(len(self.listeners))):
      try:
        self.listeners[i].put_nowait(msg)
      except Full:
        # 如果队列已满，则移除监听器
        del self.listeners[i]
