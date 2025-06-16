import os
import sys
from multiprocessing import set_start_method
from typing import Union

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

from lib.SimpleHttpServer import createSimpleHttpServer, SimpleHttpServer

httpServer: Union[SimpleHttpServer, None] = None


# 启动web服务器
def start_web_server():
  global httpServer
  from yolo import yolo_router

  httpServer = createSimpleHttpServer([yolo_router], "localhost", 50018)
  with httpServer:
    print("MagicLabelServerLoaded")
    httpServer.daemon_threads = True
    httpServer.serve_forever()


# 停止web服务器
def stop_web_server():
  global httpServer
  httpServer.shutdown()
  httpServer.server_close()


def main():
  import common

  try:
    set_start_method("spawn")
    common.initialize()
    start_web_server()
  except KeyboardInterrupt:
    common.destroy()
  finally:
    stop_web_server()


if __name__ == "__main__":
  main()
