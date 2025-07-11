import os
from collections import OrderedDict

import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import AsyncVideoFrameLoader, _load_img_as_tensor
from tqdm import tqdm


def split_filename(filename):
  # 去掉扩展名
  name_without_ext = os.path.splitext(filename)[0]

  # 找最后一个下划线位置
  idx = name_without_ext.rfind('_')
  if idx == -1:
    # 如果无下划线，返回整个名字为前缀，数字默认0
    return name_without_ext, 0

  prefix = name_without_ext[:idx]
  number_part = name_without_ext[idx + 1:]

  try:
    number = int(number_part)
  except ValueError:
    # 不能转换为整数，数字默认0
    number = 0

  return prefix, number


def extract_number(filename):
  # 去掉扩展名
  name_without_ext = os.path.splitext(filename)[0]
  # 找最后一个下划线位置
  idx = name_without_ext.rfind('_')
  if idx == -1:
    # 找不到下划线，返回0或其他默认值
    return 0
  # 截取最后一个下划线之后的部分
  number_part = name_without_ext[idx + 1:]
  try:
    return int(number_part)
  except ValueError:
    # 如果不能转换为整数，返回0或其他默认值
    return 0


def load_video_frames_from_jpg_images(
  video_path,
  image_size,
  offload_video_to_cpu,
  img_mean=(0.485, 0.456, 0.406),
  img_std=(0.229, 0.224, 0.225),
  async_loading_frames=False,
  compute_device=torch.device("cuda"),
):
  """
  Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

  The frames are resized to image_size x image_size and are loaded to GPU if
  `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

  You can load a frame asynchronously by setting `async_loading_frames` to `True`.
  """
  if isinstance(video_path, str) and os.path.isdir(video_path):
    jpg_folder = video_path
  else:
    raise NotImplementedError(
      "Only JPEG frames are supported at this moment. For video files, you may use "
      "ffmpeg (https://ffmpeg.org/) to extract frames into a folder of JPEG files, such as \n"
      "```\n"
      "ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'\n"
      "```\n"
      "where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks "
      "ffmpeg to start the JPEG file from 00000.jpg."
    )

  frame_names = [
    p
    for p in os.listdir(jpg_folder)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
  ]
  frame_names.sort(key=extract_number)
  num_frames = len(frame_names)
  if num_frames == 0:
    raise RuntimeError(f"no images found in {jpg_folder}")

  # 取第一个文件名提取前缀，默认所有文件名前缀一致
  img_prefix, _ = split_filename(frame_names[0])

  img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
  img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
  img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

  if async_loading_frames:
    lazy_images = AsyncVideoFrameLoader(
      img_paths,
      image_size,
      offload_video_to_cpu,
      img_mean,
      img_std,
      compute_device,
    )
    return lazy_images, lazy_images.video_height, lazy_images.video_width

  images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
  for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
    images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
  if not offload_video_to_cpu:
    images = images.to(compute_device)
    img_mean = img_mean.to(compute_device)
    img_std = img_std.to(compute_device)
  # normalize by mean and std
  images -= img_mean
  images /= img_std
  return images, video_height, video_width, img_prefix


@torch.inference_mode()
def init_state(
  predictor: SAM2VideoPredictor,
  video_path,
  offload_video_to_cpu=False,
  offload_state_to_cpu=False,
  async_loading_frames=False,
):
  """Initialize an inference state."""
  compute_device = predictor.device  # device of the model
  images, video_height, video_width, img_prefix = load_video_frames_from_jpg_images(
    video_path=video_path,
    image_size=predictor.image_size,
    offload_video_to_cpu=offload_video_to_cpu,
    async_loading_frames=async_loading_frames,
    compute_device=compute_device,
  )
  inference_state = {}
  inference_state["img_prefix"] = img_prefix
  inference_state["images"] = images
  inference_state["num_frames"] = len(images)
  # whether to offload the video frames to CPU memory
  # turning on this option saves the GPU memory with only a very small overhead
  inference_state["offload_video_to_cpu"] = offload_video_to_cpu
  # whether to offload the inference state to CPU memory
  # turning on this option saves the GPU memory at the cost of a lower tracking fps
  # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
  # and from 24 to 21 when tracking two objects)
  inference_state["offload_state_to_cpu"] = offload_state_to_cpu
  # the original video height and width, used for resizing final output scores
  inference_state["video_height"] = video_height
  inference_state["video_width"] = video_width
  inference_state["device"] = compute_device
  if offload_state_to_cpu:
    inference_state["storage_device"] = torch.device("cpu")
  else:
    inference_state["storage_device"] = compute_device
  # inputs on each frame
  inference_state["point_inputs_per_obj"] = {}
  inference_state["mask_inputs_per_obj"] = {}
  # visual features on a small number of recently visited frames for quick interactions
  inference_state["cached_features"] = {}
  # values that don't change across frames (so we only need to hold one copy of them)
  inference_state["constants"] = {}
  # mapping between client-side object id and model-side object index
  inference_state["obj_id_to_idx"] = OrderedDict()
  inference_state["obj_idx_to_id"] = OrderedDict()
  inference_state["obj_ids"] = []
  # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
  inference_state["output_dict_per_obj"] = {}
  # A temporary storage to hold new outputs when user interact with a frame
  # to add clicks or mask (it's merged into "output_dict" before propagation starts)
  inference_state["temp_output_dict_per_obj"] = {}
  # Frames that already holds consolidated outputs from click or mask inputs
  # (we directly use their consolidated outputs during tracking)
  # metadata for each tracking frame (e.g. which direction it's tracked)
  inference_state["frames_tracked_per_obj"] = {}
  # Warm up the visual backbone and cache the image feature on frame 0
  predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
  return inference_state


def mask_to_bbox_normalized(mask, img_width, img_height):
  """
  将二值mask转换成归一化的 (center_x, center_y, width, height) 边界框，归一化基于原图尺寸。

  参数:
      mask: np.ndarray, shape=(H_mask, W_mask), dtype=bool或其他，表示掩码
      img_width: int, 原始图片宽度
      img_height: int, 原始图片高度

  返回:
      bbox: tuple (center_x, center_y, width, height)，均归一化到[0,1]
            如果mask为空，返回None
  """
  try:
    import numpy as np
  except ImportError:
    return None

  mask = mask.astype(bool)
  rows = np.any(mask, axis=1)
  cols = np.any(mask, axis=0)

  if not rows.any() or not cols.any():
    return None

  y_min, y_max = np.where(rows)[0][[0, -1]]
  x_min, x_max = np.where(cols)[0][[0, -1]]

  width_mask = x_max - x_min + 1
  height_mask = y_max - y_min + 1

  center_x_mask = x_min + width_mask / 2
  center_y_mask = y_min + height_mask / 2

  # 注意：mask对应原图的大小可能不一样，这里默认mask是对原图大小的裁剪或缩放
  # 需要知道mask相对于原图的位置和缩放关系，否则无法准确映射

  # 如果mask是原图大小掩码，直接归一化
  center_x_norm = center_x_mask / img_width
  center_y_norm = center_y_mask / img_height
  width_norm = width_mask / img_width
  height_norm = height_mask / img_height

  return center_x_norm, center_y_norm, width_norm, height_norm
