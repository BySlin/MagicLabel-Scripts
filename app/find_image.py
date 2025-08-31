import os.path
from glob import glob

from lib.SimpleHttpServer import Router, RequestHandler
from lib.utils import is_blank

image_router = Router("/api/image")

is_stop = False


@image_router.register("/start_find_template", method="POST")
def start_find_template(handler: RequestHandler):
    global is_stop
    try:
        import cv2
        import numpy as np
        from template_search import TemplateSearch
    except ImportError:
        return {"success": False, "msg": "未安装opencv-python"}
    data = handler.read_json()
    template_dir = data["templateDir"]
    # 输入图片路径
    folderPath = data["folderPath"]
    threshold = data["threshold"]
    limit = data["limit"]
    method = data["method"]
    if is_blank(template_dir):
        return {"success": False, "msg": "未输入模板目录"}

    if is_blank(folderPath):
        return {"success": False, "msg": "未输入图片目录"}

    template_image_files = (
            glob(os.path.join(template_dir, "*.jpg"))
            + glob(os.path.join(template_dir, "*.jpeg"))
            + glob(os.path.join(template_dir, "*.png"))
            + glob(os.path.join(template_dir, "*.bmp"))
    )

    image_files = (
            glob(os.path.join(folderPath, "*.jpg"))
            + glob(os.path.join(folderPath, "*.jpeg"))
            + glob(os.path.join(folderPath, "*.png"))
            + glob(os.path.join(folderPath, "*.bmp"))
    )
    # 确保DetectLabels目录存在
    detect_labels_dir = os.path.join(folderPath, "DetectLabels")
    os.makedirs(detect_labels_dir, exist_ok=True)

    def read_existing_labels(file_path):
        """读取现有标签文件内容"""
        if not os.path.exists(file_path):
            return set()
        with open(file_path, "r") as f:
            return set(line.strip() for line in f.readlines() if line.strip())

    for image_file in image_files:
        if is_stop:
            is_stop = False
            return {"success": True, "msg": "停止找图成功"}
        image_basename = os.path.splitext(os.path.basename(image_file))[0]
        label_file_path = os.path.join(detect_labels_dir, f"{image_basename}.txt")

        # 读取现有标签内容
        existing_labels = read_existing_labels(label_file_path)

        for template_image_file in template_image_files:
            if is_stop:
                is_stop = False
                return {"success": True, "msg": "停止找图成功"}
            # 获取template_image_file的文件名
            template_image_file_name = os.path.basename(template_image_file)
            # 文件名按下划线分割
            template_parts = template_image_file_name.split('_')
            if len(template_parts) > 0:
                clsIndex = template_parts[1]
            else:
                clsIndex = "0"
            results = TemplateSearch.find_image(image_file, template_image_file, threshold, limit, method)
            for result in results:
                # 构建标签行
                label_line = f"{clsIndex} {result['n_centerX']} {result['n_centerY']} {result['n_width']} {result['n_height']}"
                # 检查是否已存在相同标签，避免重复写入
                if label_line not in existing_labels:
                    with open(label_file_path, "a") as f:
                        f.write(f"{label_line}\n")
                    existing_labels.add(label_line)

    return {"success": True, "msg": "找图结束"}


@image_router.register("/stop_find_template", method="POST")
def stop_find_template(handler: RequestHandler):
    global is_stop
    is_stop = True
    return {"success": True, "msg": "停止找图成功"}
