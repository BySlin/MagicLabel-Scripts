import http.server
import json
import socketserver


class Router:
  def __init__(self, prefix=""):
    self.prefix = prefix
    self.routes = {}

  def register(self, path, method="GET"):
    def decorator(func):
      full_path = self.prefix + path
      self.routes[(full_path, method)] = func
      return func

    return decorator

  def match_route(self, path, method="GET"):
    return self.routes.get((path, method))


class RequestHandler(http.server.SimpleHTTPRequestHandler):
  def __init__(self, *args, routers=None, **kwargs):
    self.routers = routers or []
    super().__init__(*args, **kwargs)

  def handle(self):
    try:
      super().handle()
    except:
      pass

  def do_GET(self):
    self.handle_request("GET")

  def do_POST(self):
    self.handle_request("POST")

  def handle_request(self, method):
    path = self.path
    for router in self.routers:
      route = router.match_route(path, method)
      if route:
        response = route(self)
        if callable(response):
          response(self)
        else:
          self.send_json(response)
        return
    self.send_json({"success": False, "msg": "Not found"}, status=404)

  def send_json(self, data, status=200):
    response = json.dumps(data)
    self.send_response(status)
    self.send_header("Content-Type", "application/json; charset=utf-8")
    self.end_headers()
    self.wfile.write(response.encode("utf-8"))

  def send_error_json(self, message, status=400):
    self.send_json({"success": False, "msg": message}, status=status)

  # Utility functions
  def read_json(self):
    content_length = int(self.headers.get("Content-Length", 0))
    post_data = self.rfile.read(content_length)
    try:
      return json.loads(post_data)
    except json.JSONDecodeError:
      self.send_error_json("Invalid JSON", status=400)
      return None


class SimpleHttpServer(socketserver.ThreadingTCPServer):
  # Enable the reuse address option
  allow_reuse_address = True


def createSimpleHandler(routers):
  """
  Create a request handler class with the specified routers.

  Args:
      routers (list): A list of Router objects to be used for route handling.

  Returns:
      Callable: A configured request handler class.
  """

  class CustomRequestHandler(RequestHandler):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, routers=routers, **kwargs)

  return CustomRequestHandler


def createSimpleHttpServer(routers=None, host="localhost", port=8000):
  return SimpleHttpServer((host, port), createSimpleHandler(routers))
