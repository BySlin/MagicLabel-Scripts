from multiprocessing import set_start_method
from typing import Union

from lib.SimpleHttpServer import createSimpleHttpServer, SimpleHttpServer
from lib.utils import check_and_kill_port_process_and_children

httpServer: Union[SimpleHttpServer, None] = None


# 启动web服务器
def start_web_server():
  global httpServer
  from yolo import yolo_router

  # http通讯端口
  port = 50018
  # 检查端口是否被占用，如果被占用则关闭进程
  check_and_kill_port_process_and_children(port)
  # 创建web服务器
  httpServer = createSimpleHttpServer([yolo_router], "localhost", port)
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
