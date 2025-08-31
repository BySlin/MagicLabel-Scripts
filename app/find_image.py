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


def nms_by_class(all_results, all_cls_indices, iou_threshold=0.5):
    """
    按类别执行非极大值抑制(NMS)

    Args:
        all_results: 所有检测结果列表
        all_cls_indices: 每个检测结果对应的类别索引列表
        iou_threshold: IOU阈值

    Returns:
        保留的检测结果索引列表
    """
    if len(all_results) == 0:
        return []

    # 按类别分组
    class_groups = {}
    for idx, cls_index in enumerate(all_cls_indices):
        if cls_index not in class_groups:
            class_groups[cls_index] = []
        class_groups[cls_index].append(idx)

    # 对每个类别分别执行NMS
    keep_indices = []
    for cls_index, indices in class_groups.items():
        # 提取当前类别的检测框和置信度分数
        boxes = []
        scores = []
        for i in indices:
            result = all_results[i]
            x1 = result['x']
            y1 = result['y']
            x2 = result['x'] + result['width']
            y2 = result['y'] + result['height']
            boxes.append([x1, y1, x2, y2])
            scores.append(result['confidence'])

        # 执行NMS
        keep = nms(boxes, scores, iou_threshold)

        # 将保留的索引添加到结果中
        for k in keep:
            keep_indices.append(indices[k])

    return keep_indices


def parse_yolo_label(label_line, image_width, image_height):
    """
    解析YOLO标签行，返回检测框坐标

    Args:
        label_line: YOLO标签行 (cls centerX centerY width height)
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        (cls_index, x1, y1, x2, y2): 类别索引和检测框坐标
    """
    parts = label_line.strip().split()
    if len(parts) != 5:
        return None

    cls_index = parts[0]
    center_x = float(parts[1]) * image_width
    center_y = float(parts[2]) * image_height
    width = float(parts[3]) * image_width
    height = float(parts[4]) * image_height

    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2

    return (cls_index, x1, y1, x2, y2)


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

        # 读取图像以获取尺寸
        import cv2
        image = cv2.imread(image_file)
        if image is None:
            continue
        image_height, image_width = image.shape[:2]

        image_basename = os.path.splitext(os.path.basename(image_file))[0]
        label_file_path = os.path.join(detect_labels_dir, f"{image_basename}.txt")

        # 读取现有标签内容
        existing_labels = read_existing_labels(label_file_path)

        # 用于存储所有检测结果（包括已有的和新检测的）
        all_results = []
        all_cls_indices = []

        # 解析已有的标签数据并添加到结果中
        for label_line in existing_labels:
            parsed = parse_yolo_label(label_line, image_width, image_height)
            if parsed is not None:
                cls_index, x1, y1, x2, y2 = parsed
                # 构造与新检测结果相同格式的数据
                result = {
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'confidence': 1.0,  # 已有标签的置信度设为1.0
                    'n_centerX': float(x1 + x2) / 2 / image_width,
                    'n_centerY': float(y1 + y2) / 2 / image_height,
                    'n_width': float(x2 - x1) / image_width,
                    'n_height': float(y2 - y1) / image_height
                }
                all_results.append(result)
                all_cls_indices.append(cls_index)

        for template_image_file in template_image_files:
            if is_stop:
                is_stop = False
                return {"success": True, "msg": "停止找图成功"}
            # 去除文件名后缀
            template_image_file_name = os.path.splitext(os.path.basename(template_image_file))[0]
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

        # 如果有检测结果，执行按类别NMS去重
        if all_results:
            # 执行按类别NMS，IOU阈值设为0.5
            keep_indices = nms_by_class(all_results, all_cls_indices, 0.5)

            # 清空标签文件
            with open(label_file_path, "w") as f:
                pass

            # 重新保存NMS后的结果（包括已有的和新检测的）
            for idx in keep_indices:
                result = all_results[idx]
                clsIndex = all_cls_indices[idx]
                # 构建标签行
                label_line = f"{clsIndex} {result['n_centerX']} {result['n_centerY']} {result['n_width']} {result['n_height']}"
                with open(label_file_path, "a") as f:
                    f.write(f"{label_line}\n")

    return {"success": True, "msg": "找图结束"}


@image_router.register("/stop_find_template", method="POST")
def stop_find_template(handler: RequestHandler):
    global is_stop
    is_stop = True
    return {"success": True, "msg": "停止找图成功"}
