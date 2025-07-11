import http.server
import json
import socketserver
from urllib.parse import urlparse, parse_qs


class Router:
  def __init__(self, prefix=""):
    self.prefix = prefix  # 路由前缀
    self.routes = {}  # 路由字典，存储路径和请求方法到处理函数的映射

  def register(self, path, method="GET"):
    # 注册路由的装饰器
    def decorator(func):
      full_path = self.prefix + path  # 完整路径
      self.routes[(full_path, method)] = func  # 将路径和方法映射到处理函数
      return func  # 返回处理函数

    return decorator  # 返回装饰器函数

  def match_route(self, path, method="GET"):
    # 匹配路由
    return self.routes.get((path, method))  # 返回匹配的路由处理函数，如果没有匹配则返回None


class RequestHandler(http.server.SimpleHTTPRequestHandler):
  def __init__(self, *args, routers=None, **kwargs):
    self.query_params = {}  # 查询参数字典
    self.routers = routers or []  # 路由列表
    super().__init__(*args, **kwargs)  # 调用父类构造函数

  def handle(self):
    try:
      super().handle()  # 调用父类的handle方法处理请求
    except:
      pass  # 捕获所有异常但不处理

  def do_GET(self):
    self.handle_request("GET")  # 处理GET请求

  def do_POST(self):
    self.handle_request("POST")  # 处理POST请求

  def do_OPTIONS(self):
    self.send_response(200)
    self.end_headers()

  def handle_request(self, method):
    # 处理请求的通用方法
    parsed_url = urlparse(self.path)  # 解析URL
    path = parsed_url.path  # 获取路径
    query_params = parse_qs(parsed_url.query)  # 解析查询参数
    self.query_params = query_params  # 存储查询参数
    for router in self.routers:
      route = router.match_route(path, method)  # 匹配路由
      if route:
        response = route(self)  # 调用路由处理函数
        if callable(response):
          response(self)  # 如果响应是可调用对象，则调用它
        else:
          if response is not None:
            self.send_json(response)  # 发送JSON响应
        return
    self.send_json({"success": False, "msg": "Not found"}, status=404)  # 如果没有匹配的路由，发送404错误

  def send_json(self, data, status=200):
    # 发送JSON响应
    response = json.dumps(data)  # 将数据转换为JSON字符串
    self.send_response(status)  # 发送HTTP响应状态码
    self.send_header("Content-Type", "application/json; charset=utf-8")  # 设置内容类型为JSON
    self.end_headers()  # 结束头部
    self.wfile.write(response.encode("utf-8"))  # 写入响应内容

  def send_error_json(self, message, status=400):
    # 发送错误JSON响应
    self.send_json({"success": False, "msg": message}, status=status)  # 调用send_json方法发送错误信息

  # 工具函数
  def read_json(self):
    # 读取JSON请求体
    content_length = int(self.headers.get("Content-Length", 0))  # 获取内容长度
    post_data = self.rfile.read(content_length)  # 读取POST数据
    try:
      return json.loads(post_data)  # 尝试解析JSON数据
    except json.JSONDecodeError:
      self.send_error_json("Invalid JSON", status=400)  # 如果解析失败，发送错误信息
      return None  # 返回None

  def get_query_param(self, key, default=None):
    # 获取查询参数
    values = self.query_params.get(key)  # 获取参数值
    if values:
      return values[0]  # 返回第一个值
    return default  # 如果没有值，返回默认值

  def end_headers(self):
    self.send_header("Access-Control-Allow-Origin", "*")
    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, last-event-id")
    super().end_headers()


class SimpleHttpServer(socketserver.ThreadingTCPServer):
  # 简单的HTTP服务器类，继承自ThreadingTCPServer
  allow_reuse_address = True  # 允许重用地址
  daemon_threads = True  # 设置为守护线程


def createSimpleHandler(routers):
  """
  创建一个带有指定路由的请求处理类。

  Args:
      routers (list): 路由对象列表，用于处理路由。

  Returns:
      Callable: 配置好的请求处理类。
  """

  class CustomRequestHandler(RequestHandler):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, routers=routers, **kwargs)  # 调用父类构造函数，传入路由列表

    def log_message(self, format, *args):
      pass  # 重写为空，禁止日志打印

  return CustomRequestHandler  # 返回自定义请求处理类


def createSimpleHttpServer(routers=None, host="localhost", port=8000):
  # 创建简单的HTTP服务器
  return SimpleHttpServer((host, port), createSimpleHandler(routers))  # 返回SimpleHttpServer实例
