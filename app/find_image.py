import os.path
from glob import glob
import numpy as np

from lib.SimpleHttpServer import Router, RequestHandler
from lib.utils import is_blank

image_router = Router("/api/image")

is_stop = False


def nms(boxes, scores, iou_threshold):
    """
    执行非极大值抑制(NMS)

    Args:
        boxes: 检测框列表 [[x1, y1, x2, y2], ...]
        scores: 每个检测框的置信度分数
        iou_threshold: IOU阈值

    Returns:
        保留的检测框索引列表
    """
    if len(boxes) == 0:
        return []

    # 转换为numpy数组
    boxes = np.array(boxes)
    scores = np.array(scores)

    # 计算每个框的面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 按分数降序排列索引
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 选择分数最高的框
        i = order[0]
        keep.append(i)

        # 计算当前框与其他所有框的IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        # 计算交集宽度和高度
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        # 计算交集面积
        inter = w * h

        # 计算IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IoU小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


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

    # 判断folderPath是文件还是目录
    if os.path.isfile(folderPath):
        # 如果是文件，直接使用该文件
        image_files = [folderPath]
        # DetectLabels目录放在文件所在的目录下
        detect_labels_dir = os.path.join(os.path.dirname(folderPath), "DetectLabels")
    else:
        # 如果是目录，搜索目录下的图片文件
        image_files = (
                glob(os.path.join(folderPath, "*.jpg"))
                + glob(os.path.join(folderPath, "*.jpeg"))
                + glob(os.path.join(folderPath, "*.png"))
                + glob(os.path.join(folderPath, "*.bmp"))
        )
        # DetectLabels目录放在指定目录下
        detect_labels_dir = os.path.join(folderPath, "DetectLabels")

    # 确保DetectLabels目录存在
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

        # 用于存储所有检测结果
        all_results = []
        all_cls_indices = []

        for template_image_file in template_image_files:
            if is_stop:
                is_stop = False
                return {"success": True, "msg": "停止找图成功"}
            # 获取template_image_file的文件名
            template_image_file_name = os.path.basename(template_image_file)
            # 文件名按下划线分割
            template_parts = template_image_file_name.split("_")
            if len(template_parts) > 0:
                clsIndex = template_parts[1]
            else:
                clsIndex = "0"
            results = TemplateSearch.find_image(
                image_file, template_image_file, threshold, limit, method
            )

            # 保存结果和类别索引
            for result in results:
                all_results.append(result)
                all_cls_indices.append(clsIndex)

        # 如果有检测结果，执行NMS去重
        if all_results:
            # 提取检测框和置信度分数
            boxes = []
            scores = []
            for result in all_results:
                x1 = result['x']
                y1 = result['y']
                x2 = result['x'] + result['width']
                y2 = result['y'] + result['height']
                boxes.append([x1, y1, x2, y2])
                scores.append(result['confidence'])

            # 执行NMS，IOU阈值设为0.5
            keep_indices = nms(boxes, scores, 0.5)

            # 保存NMS后的结果
            for idx in keep_indices:
                result = all_results[idx]
                clsIndex = all_cls_indices[idx]
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
