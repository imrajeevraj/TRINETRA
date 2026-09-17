import torch
import logging
import numpy as np
import threading

logger = logging.getLogger("OcrEngine")


class OcrEngine:
    def __init__(self, languages=["en"], use_gpu=None):
        self.languages = languages
        self.use_gpu = use_gpu
        self.reader = None
        self._init_lock = threading.Lock()
        self.initialized = False

    def _get_reader(self):
        if self.reader is None:
            with self._init_lock:
                if self.reader is None:
                    try:
                        import easyocr

                        gpu_flag = (
                            self.use_gpu
                            if self.use_gpu is not None
                            else torch.cuda.is_available()
                        )
                        logger.info(f"Lazy-initializing EasyOCR with GPU={gpu_flag}...")
                        self.reader = easyocr.Reader(self.languages, gpu=gpu_flag)
                        self.initialized = True
                        logger.info("EasyOCR initialized successfully.")
                    except Exception as e:
                        logger.error(f"Failed to initialize EasyOCR: {e}")
                        self.initialized = False
        return self.reader

    def extract_text(self, image: np.ndarray):
        """
        Extract text from an image crop (e.g., a vehicle bounding box).
        Returns a list of dictionaries with text and confidence.
        """
        if image is None or image.size == 0:
            return []

        reader = self._get_reader()
        if reader is None:
            return []

        try:
            results = reader.readtext(image)
            extracted = []
            for bbox, text, prob in results:
                extracted.append(
                    {"bbox": bbox, "text": text, "confidence": float(prob)}
                )
            return extracted
        except Exception as e:
            logger.error(f"Error extracting text with EasyOCR: {e}")
            return []


ocr_engine = OcrEngine(use_gpu=None)
