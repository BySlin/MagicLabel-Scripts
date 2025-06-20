import ctypes
import os
import platform

import cv2
import numpy as np

win_name = "YoloResult"
is_first = True
default_window_width = 1280
default_window_height = 720


# 判断是否为 Windows 系统
def is_windows():
  return platform.system() == 'Windows'


# 判断是否是图片文件
def is_image_file(filename):
  return os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']


# 字符串转整数，如果无法转换则返回原字符串
def str_to_int_if_possible(s):
  try:
    return int(s)
  except ValueError:
    return s


# 激活窗口
def activate_window(hwnd):
  from ctypes import wintypes
  user32 = ctypes.windll.user32
  user32.ShowWindow(hwnd, 5)  # SW_SHOW (5)
  SetWindowPos = user32.SetWindowPos
  SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           ctypes.c_uint]
  SetWindowPos.restype = wintypes.BOOL
  SetWindowPos(hwnd, 0, 0, 0, 0, 0, 0x0001 | 0x0002)


# 获取窗口句柄
def get_hwnd(window_title):
  from ctypes import wintypes
  user32 = ctypes.windll.user32
  FindWindow = user32.FindWindowW
  FindWindow.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
  FindWindow.restype = wintypes.HWND
  hwnd = FindWindow(None, window_title)
  if not hwnd:
    raise Exception(f"找不到窗口: {window_title}")
  return hwnd


# 获取系统 DPI
def get_system_dpi(hwnd):
  from ctypes import wintypes
  # 获取窗口的 DPI
  dpi = ctypes.c_int()
  # 使用 GetDpiForWindow 函数获取特定窗口的 DPI
  user32 = ctypes.windll.user32
  user32.GetDpiForWindow.argtypes = [wintypes.HWND]
  user32.GetDpiForWindow.restype = ctypes.c_int
  dpi.value = user32.GetDpiForWindow(hwnd)
  return dpi.value


# 获取窗口矩形（左，上，右，下）
def get_window_rect(hwnd):
  from ctypes import wintypes
  user32 = ctypes.windll.user32
  rect = wintypes.RECT()
  user32.GetWindowRect(hwnd, ctypes.byref(rect))

  # 获取窗口的 DPI 并计算缩放因子
  dpi = get_system_dpi(hwnd)
  scale_factor = dpi / 96  # 96 DPI 是默认设置

  # 返回经过缩放调整的坐标
  return (
    int(rect.left * scale_factor),
    int(rect.top * scale_factor),
    int(rect.right * scale_factor),
    int(rect.bottom * scale_factor)
  )


def show_image(img):
  global is_first
  if is_first:
    is_first = False
    cv2.resizeWindow(win_name, default_window_width, default_window_height)
  x, y, win_w, win_h = cv2.getWindowImageRect(win_name)
  if win_w == 0 or win_h == 0:
    win_w = default_window_width
    win_h = default_window_height

  h, w = img.shape[:2]
  aspect_ratio = w / h
  # 计算缩放尺寸，保持比例完全显示
  win_ratio = win_w / win_h
  if win_ratio > aspect_ratio:
    # 窗口宽度较大，高度决定大小
    new_h = win_h
    new_w = int(aspect_ratio * new_h)
  else:
    new_w = win_w
    new_h = int(new_w / aspect_ratio)

  # 缩放图像
  resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

  # 创建黑色背景画布
  canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

  # 计算偏移居中
  x_offset = (win_w - new_w) // 2
  y_offset = (win_h - new_h) // 2

  # 将缩放图像放到画布中间
  canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

  # 显示
  cv2.imshow(win_name, canvas)
  cv2.waitKey(1)
