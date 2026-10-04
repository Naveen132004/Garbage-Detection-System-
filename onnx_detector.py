"""Light YOLOv8 detector on ONNX Runtime (no PyTorch), used for small servers.

Create the .onnx file once from a trained model with:
    yolo export model=Weights/best.pt format=onnx imgsz=640
"""
import ast
import ctypes
import threading

import cv2
import numpy as np
import onnxruntime as ort

try:
    _libc = ctypes.CDLL("libc.so.6")  # Linux: lets us hand freed memory back to the OS
except OSError:
    _libc = None


class OnnxYoloDetector:
    def __init__(self, model_path, iou_threshold=0.45):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        # Keep memory low on small servers (Render free plan has 512 MB)
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.session = ort.InferenceSession(model_path, sess_options=options,
                                            providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        height, width = model_input.shape[2], model_input.shape[3]
        self.input_size = (int(width) if isinstance(width, int) else 640,
                           int(height) if isinstance(height, int) else 640)
        self.iou_threshold = iou_threshold
        self.lock = threading.Lock()  # one inference at a time keeps peak memory predictable

        # Ultralytics stores class names in the ONNX metadata, e.g. "{0: 'battery', ...}"
        meta = self.session.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(meta.get("names", "{}"))
        except (ValueError, SyntaxError):
            names = {}
        self.names = names

    def _letterbox(self, img):
        w, h = self.input_size
        scale = min(w / img.shape[1], h / img.shape[0])
        new_w, new_h = round(img.shape[1] * scale), round(img.shape[0] * scale)
        pad_x, pad_y = (w - new_w) / 2, (h - new_h) / 2
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
        left, right = round(pad_x - 0.1), round(pad_x + 0.1)
        canvas = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return canvas, scale, left, top

    def detect(self, img, conf_threshold=0.25):
        """Return a list of (x1, y1, x2, y2, confidence, class_id) in original image pixels."""
        canvas, scale, pad_x, pad_y = self._letterbox(img)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        with self.lock:
            output = self.session.run(None, {self.input_name: blob})[0][0]  # (4 + classes, anchors)
            if _libc is not None:
                _libc.malloc_trim(0)

        predictions = output.T
        scores = predictions[:, 4:]
        class_ids = scores.argmax(axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        keep = confidences > conf_threshold
        if not keep.any():
            return []

        boxes = predictions[keep, :4]
        confidences = confidences[keep]
        class_ids = class_ids[keep]

        # center x, center y, width, height -> corners in the original image
        x1 = (boxes[:, 0] - boxes[:, 2] / 2 - pad_x) / scale
        y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pad_y) / scale
        x2 = (boxes[:, 0] + boxes[:, 2] / 2 - pad_x) / scale
        y2 = (boxes[:, 1] + boxes[:, 3] / 2 - pad_y) / scale
        h, w = img.shape[:2]
        x1, x2 = np.clip(x1, 0, w), np.clip(x2, 0, w)
        y1, y2 = np.clip(y1, 0, h), np.clip(y2, 0, h)

        # Per-class NMS (offset boxes by class so different classes don't suppress each other)
        offset = class_ids.astype(np.float32) * 7680
        rects = np.stack([x1 + offset, y1 + offset, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(rects.tolist(), confidences.tolist(), conf_threshold, self.iou_threshold)
        indices = np.array(indices).flatten()[:300]

        order = indices[np.argsort(-confidences[indices])]
        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                 float(confidences[i]), int(class_ids[i])) for i in order]
