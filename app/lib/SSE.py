import json
import uuid
from queue import Queue, Full, Empty
from typing import List, Dict, Union


def create_sse_msg(event: str, data: Union[str, Dict, None] = None):
  event_id = str(uuid.uuid4())
  message = f"id: {event_id}\n"
  message += f"event: {event}\n"
  if isinstance(data, dict):
    message += f"data: {json.dumps(data)}\n\n"
  elif isinstance(data, str):
    message += f"data: {data}\n\n"
  else:
    message += f"data: {event}\n\n"
  return message


class ServerSentEvents:
  """A simple implementation of Server-Sent Events for a custom HTTP server."""

  msg_id: int = 0
  listeners: List[Queue] = []

  def response(self):
    """Yields a stream of messages for the client."""

    def stream(handle):
      handle.send_response(200)
      handle.send_header("Content-Type", "text/event-stream; charset=utf-8")
      handle.send_header("Cache-Control", "no-cache")
      handle.send_header("Connection", "keep-alive")
      handle.end_headers()

      queue = Queue(5)
      self.listeners.append(queue)
      try:
        while True:
          try:
            msg = queue.get(timeout=1)
            handle.wfile.write(msg.encode("utf-8"))
            handle.wfile.flush()
          except Empty:
            try:
              handle.wfile.write(create_sse_msg("ping").encode("utf-8"))
              handle.wfile.flush()
            except:
              break
            continue
          except:
            break
      finally:
        self.listeners.remove(queue)

    return stream

  def send(self, payload: Dict = None, event: str = "data"):
    """Sends a new event to the opened channel."""
    self.msg_id += 1
    msg_str = json.dumps(payload) if payload else "{}"
    msg = f"id: {self.msg_id}\nevent: {event}\ndata: {msg_str}\n\n"

    for i in reversed(range(len(self.listeners))):
      try:
        self.listeners[i].put_nowait(msg)
      except Full:
        del self.listeners[i]

  def send_raw(self, msg: str):
    """Sends a raw message to the opened channel."""
    for i in reversed(range(len(self.listeners))):
      try:
        self.listeners[i].put_nowait(msg)
      except Full:
        del self.listeners[i]
