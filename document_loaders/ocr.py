from typing import TYPE_CHECKING
from configs import logger

if TYPE_CHECKING:
    try:
        from rapidocr_paddle import RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR


def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda)
        logger.info('Using rapidocr_paddle')
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        logger.info('Using rapidocr_onnxruntime')
    return ocr
