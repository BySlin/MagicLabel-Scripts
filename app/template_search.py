import cv2
import numpy as np


class TemplateSearch:
    @staticmethod
    def find_image(source_image, template_image, threshold=0.95, limit=10, method=1, x=0, y=0, ex=0, ey=0):
        """
        模板匹配找图（增强版）

        Args:
            source_image: 大图片（OpenCV图像对象或文件路径）
            template_image: 小图片模板（OpenCV图像对象或文件路径）
            x: 找图区域 x 起始坐标
            y: 找图区域 y 起始坐标
            ex: 终点X坐标
            ey: 终点Y坐标
            threshold: 图片相似度阈值
            limit: 限制结果的数量
            method: 匹配方法 (0-5, 对应OpenCV的匹配方法)

        Returns:
            包含匹配区域坐标信息的字典列表
        """

        # 加载图像
        if isinstance(source_image, str):
            source_mat = cv2.imread(source_image)
        else:
            source_mat = source_image

        if isinstance(template_image, str):
            template_mat = cv2.imread(template_image)
        else:
            template_mat = template_image

        # 检查图像是否为空
        if source_mat is None or template_mat is None:
            print("[FindImage] 错误：无法获取图像数据")
            return []

        if source_mat.size == 0 or template_mat.size == 0:
            print("[FindImage] 错误：图像为空")
            return []

        # 计算搜索区域
        roi_x = max(0, x)
        roi_y = max(0, y)
        roi_width = (ex > 0 and ex <= source_mat.shape[1]) and ex or source_mat.shape[1]
        roi_height = (ey > 0 and ey <= source_mat.shape[0]) and ey or source_mat.shape[0]

        # 调整ROI的起始点和尺寸
        roi_width = roi_width - roi_x
        roi_height = roi_height - roi_y

        # 检查ROI是否合法
        if roi_width <= 0 or roi_height <= 0 or \
                roi_width < template_mat.shape[1] or roi_height < template_mat.shape[0]:
            print(
                f"[FindImage]搜索区域不合法: x={x}, y={y}, ex={ex}, ey={ey}, 计算得到宽度={roi_width}, 高度={roi_height}")
            return []

        # 检查ROI是否在图像边界内
        if roi_x < 0 or roi_y < 0 or \
                roi_x + roi_width > source_mat.shape[1] or roi_y + roi_height > source_mat.shape[0]:
            print(
                f"[FindImage]搜索区域超出图像边界: ROI({roi_x}, {roi_y}, {roi_width}, {roi_height}), 图像尺寸: {source_mat.shape[1]}x{source_mat.shape[0]}")
            return []

        # 创建搜索区域ROI
        search_area = source_mat[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # 执行模板匹配
        match_methods = [
            cv2.TM_SQDIFF,
            cv2.TM_SQDIFF_NORMED,
            cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED,
            cv2.TM_CCOEFF,
            cv2.TM_CCOEFF_NORMED
        ]

        match_method = match_methods[method] if 0 <= method < len(match_methods) else cv2.TM_CCOEFF_NORMED

        try:
            result = cv2.matchTemplate(search_area, template_mat, match_method)
        except Exception as e:
            print(f"[FindImage]模板匹配执行错误: {str(e)}")
            return []

        # 检查结果矩阵是否有效
        if result.size == 0 or result.shape[0] <= 0 or result.shape[1] <= 0:
            print("[FindImage]错误：模板匹配结果矩阵无效")
            return []

        # 严格使用用户指定阈值，不进行自适应搜索
        max_limit = limit if limit > 0 else float('inf')

        found_rects = TemplateSearch._perform_template_matching(
            result=result,
            method=method,
            threshold=threshold,
            limit=max_limit,
            template_mat=template_mat,
            roi_x=roi_x,
            roi_y=roi_y,
            source_width=source_mat.shape[1],
            source_height=source_mat.shape[0]
        )

        return found_rects

    @staticmethod
    def _perform_template_matching(result, method, threshold, limit, template_mat, roi_x, roi_y, source_width,
                                   source_height):
        """
        执行模板匹配搜索

        Args:
            result: 模板匹配结果矩阵
            method: 匹配方法
            threshold: 阈值
            limit: 限制数量
            template_mat: 模板图像
            roi_x: ROI区域X坐标
            roi_y: ROI区域Y坐标
            source_width: 源图像宽度
            source_height: 源图像高度

        Returns:
            匹配结果数组
        """
        found_rects = []

        # 根据匹配方法确定阈值比较方式
        is_inverted_method = (method == 0 or method == 1)  # TM_SQDIFF 和 TM_SQDIFF_NORMED

        # 使用非最大值抑制查找多个匹配
        # 注意：mask中255表示可用区域，0表示屏蔽区域
        used_mask = np.ones(result.shape, dtype=np.uint8) * 255  # 初始化为全部可用

        for i in range(int(limit)):
            # 应用mask来查找下一个最佳匹配
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result, used_mask)
            raw_match_value = min_val if is_inverted_method else max_val
            match_point = min_loc if is_inverted_method else max_loc

            # 检查是否找到有效位置
            if match_point[0] < 0 or match_point[1] < 0:
                break

            # 修正 confidence 计算逻辑
            confidence = TemplateSearch._calculate_confidence(raw_match_value, method, template_mat)

            # 检查是否满足阈值条件和质量要求
            meets_threshold = confidence >= threshold

            if not meets_threshold:
                break

            # 计算绝对坐标
            absolute_x = roi_x + match_point[0]
            absolute_y = roi_y + match_point[1]

            # 创建矩形区域
            rect = {
                "x": int(absolute_x),
                "y": int(absolute_y),
                "width": template_mat.shape[1],
                "height": template_mat.shape[0],
                "confidence": confidence,
                "n_x": absolute_x / source_width,
                "n_y": absolute_y / source_height,
                "n_width": template_mat.shape[1] / source_width,
                "n_height": template_mat.shape[0] / source_height,
                "centerX": int(absolute_x + template_mat.shape[1] / 2),
                "centerY": int(absolute_y + template_mat.shape[0] / 2),
                "n_centerX": (absolute_x + template_mat.shape[1] / 2) / source_width,
                "n_centerY": (absolute_y + template_mat.shape[0] / 2) / source_height,
            }

            found_rects.append(rect)

            # 在已使用区域标记，避免重复检测
            mask_rect_x = max(0, match_point[0] - template_mat.shape[1] // 2)
            mask_rect_y = max(0, match_point[1] - template_mat.shape[0] // 2)
            mask_rect_width = template_mat.shape[1]
            mask_rect_height = template_mat.shape[0]

            # 确保mask区域在结果图像范围内
            clamped_rect_x = max(0, min(mask_rect_x, result.shape[1] - 1))
            clamped_rect_y = max(0, min(mask_rect_y, result.shape[0] - 1))
            clamped_rect_width = min(mask_rect_width, result.shape[1] - max(0, mask_rect_x))
            clamped_rect_height = min(mask_rect_height, result.shape[0] - max(0, mask_rect_y))

            if clamped_rect_width > 0 and clamped_rect_height > 0:
                # 在mask中标记该区域为已使用(设置为0)
                used_mask[
                    clamped_rect_y:clamped_rect_y + clamped_rect_height,
                    clamped_rect_x:clamped_rect_x + clamped_rect_width
                ] = 0
            else:
                # 如果无法创建有效的mask区域，则退出循环
                break

        return found_rects

    @staticmethod
    def _calculate_confidence(raw_value, method, template_mat):
        """
        计算标准化的confidence值
        """
        if method == 1:  # TM_SQDIFF_NORMED - 值越小越好，范围是 [0, 1]
            return max(0.0, 1.0 - raw_value)

        elif method == 3:  # TM_CCORR_NORMED - 值越大越好，范围是 [0, 1]
            return raw_value

        elif method == 5:  # TM_CCOEFF_NORMED - 值越大越好，范围是 [-1, 1]
            # 负值表示反相关，直接返回0
            if raw_value <= 0:
                return 0.0
            # 将 [0, 1] 直接作为confidence
            return min(1.0, max(0.0, raw_value))

        else:
            return max(0.0, min(1.0, raw_value))
