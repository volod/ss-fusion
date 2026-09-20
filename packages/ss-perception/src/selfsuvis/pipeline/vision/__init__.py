"""Vision, captioning, and model-selection helpers."""

from importlib import import_module

_EXPORTS = {
    "ASRModel": (".asr", "ASRModel"),
    "DEFAULT_LABELS": (".labels", "DEFAULT_LABELS"),
    "DetectionModel": (".detection", "DetectionModel"),
    "DepthModel": (".depth", "DepthModel"),
    "FlorenceModel": (".florence", "FlorenceModel"),
    "OCRModel": (".ocr", "OCRModel"),
    "QwenModel": (".qwen", "QwenModel"),
    "RFDETRTracker": (".rfdetr", "RFDETRTracker"),
    "RFSignalAnalyzer": (".rf_analyzer", "RFSignalAnalyzer"),
    "SAMPredictor": (".sam", "SAMPredictor"),
    "UniDriveVLAModel": (".unidrive", "UniDriveVLAModel"),
    "WorldModel": (".world", "WorldModel"),
    "YOLODetector": (".yolo", "YOLODetector"),
    "auto_select": (".registry", "auto_select"),
    "build_vision_models": (".factory", "build_vision_models"),
    "detect_resources": (".registry", "detect_resources"),
    "load_labels": (".labels", "load_labels"),
    "normalize_model_id": (".registry", "normalize_model_id"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    return getattr(import_module(module_name, __name__), attr_name)
