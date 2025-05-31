from models.common import AutoShape, DetectMultiBackend
from models.experimental import attempt_load
from models.yolo import ClassificationModel, SegmentationModel
from utils.general import LOGGER, logging
from utils.torch_utils import select_device


def load_model(
        model_path, device=None, autoshape=True, verbose=False,
):
    """
    Creates a specified YOLOv5 model

    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent

    Returns:
        pytorch model

    (Adapted from yolov5.hubconf.create)
    """
    # set logging
    if not verbose:
        LOGGER.setLevel(logging.WARNING)

    # set device
    device = select_device(device)

    try:
        model = DetectMultiBackend(model_path, device=device, fuse=autoshape)  # detection model
        if autoshape:
            if model.pt and isinstance(model.model, ClassificationModel):
                LOGGER.warning(
                    "WARNING ⚠️ YOLOv5 ClassificationModel is not yet AutoShape compatible. "
                    "You must pass torch tensors in BCHW to this model, i.e. shape(1,3,224,224)."
                )
            elif model.pt and isinstance(model.model, SegmentationModel):
                LOGGER.warning(
                    "WARNING ⚠️ YOLOv5 SegmentationModel is not yet AutoShape compatible. "
                    "You will not be able to run inference with this model."
                )
            else:
                try:
                    model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
                except Exception as e:
                    LOGGER.warning(f"WARNING ⚠️ autoshape failed: {e}")
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ DetectMultiBackend failed: {e}")
        model = attempt_load(model_path, device=device, fuse=False)  # arbitrary model

    if not verbose:
        LOGGER.setLevel(logging.INFO)  # reset to default

    return model.to(device)
