import pdfplumber           # PDF text extraction
import fitz                 # PyMuPDF for image extraction
import re
import uuid
import cv2          # Image processing
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import asdict
import pandas as pd
from tabulate import tabulate
import numpy as np


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
warnings.filterwarnings("ignore", message="CropBox missing")        # to deal with WARNING:pdfminer.pdfpage:CropBox missing from /Page, defaulting to MediaBox

import pytesseract          # Set up multiple OCR providers with fallbacks
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
TESSERACT_AVAILABLE = True          # fallbacks if services are unavailable

import boto3            # AWS Textract
AWS_TEXTRACT_AVAILABLE = True

# from google.cloud import vision         # Google Vision API
# GOOGLE_VISION_AVAILABLE = True


import os
# os.environ['AWS_ACCESS_KEY_ID'] = 'here'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'here'
# os.environ['AWS_SESSION_TOKEN'] = ('here')
logging.basicConfig(level=logging.INFO)         # Configures logging and SSL settings
logger = logging.getLogger(__name__)


@dataclass
class OCRMetrics:
    cer: float
    wer: float
    bleu_score: float
    rouge_l: float
    iou_scores: List[float]
    field_accuracy: Dict[str, float]


@dataclass
class BoundingBox:              # Store spatial coordinates for detected elements.
    x: float                    # Top-left corner coordinates
    y: float
    width: float                # Rectangle dimensions
    height: float
    confidence: float = 0.0         # detection accuracy (0.0-1.0)


@dataclass
class ExtractedElement:             # Represent any detected text/table/image with location data.
    text: str               # extracted content
    element_type: str  # content category ('text', 'table', 'image', 'drawing')
    bbox: Optional[BoundingBox] = None          # location on page
    page_number: int = 0                # source page
    confidence: float = 0.0             # extraction quality score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialTable:             # Structured table data with spatial information
    rows: List[List[str]]               # 2D array of cell contents
    headers: List[str]              # column names
    bbox: BoundingBox               # table location
    page_number: int
    confidence: float = 0.0         # table detection quality
    structure_type: str = "standard"  # table complexity ("standard", "complex", "drawing_table")


# Document -> ExtractedElements + SpatialTables -> ExtractionResult -> DocumentChunks for vector storage
@dataclass
class ExtractionResult:             # Main output containing all processed document data.

    document_id: str            # document identification
    filename: str
    total_pages: int                # page count
    processing_method: str          # OCR/extraction technique used
    extracted_text: str             # full document text
    structured_data: Dict[str, Any] = field(default_factory=dict)           # categorized extracted information
    tables: List[SpatialTable] = field(default_factory=list)            # all detected tables
    elements: List[ExtractedElement] = field(default_factory=list)          # all document components
    images: List[Dict] = field(default_factory=list)                    # extracted/referenced images
    spatial_analysis: Dict[str, Any] = field(default_factory=dict)          # layout analysis results
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0                     # overall extraction quality
    timestamp: datetime = field(default_factory=datetime.now)           # processing time


class ImageProcessor:

    @staticmethod
    def detect_tables_in_image(image: np.ndarray) -> List[BoundingBox]:         # morphological operations to detect table structures by finding intersecting lines.
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # convert to grayscale
        else:
            gray = image.copy()

        kernel_length = max(max(gray.shape) // 100, 20)     # calculate adaptive kernel size (minimum 20 pixels, scaled to image size)

        # create rectangular kernels to detect horizontal  and vertical lines.
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

        # Detect horizontal and vertical lines
        # MORPH_OPEN removes noise and isolates lines matching kernel shape
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

        # combine horizontal/vertical lines to find grid intersections indicating tables
        table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

        # Find table contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tables = []
        min_table_area = max(gray.shape[0] * gray.shape[1] * 0.001, 1000)           # minimum area: 0.1% of image or 1000 pixels common in contour filtering

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_table_area:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 50 and (w / h < 10 and h / w < 10):           # minimum dimensions: 50x50 pixels
                    tables.append(BoundingBox(x, y, w, h, confidence=0.7))          # return BoundingBox objects for detected table regions with 0.7 confidence

        return tables


    def detect_technical_objects(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:         # Detect engineering drawings elements like dimension lines and symbols.
        # self refers to the instance of the class
        # image is aNumPy array, standard format for images in OpenCV
        elements = []

        dimension_elements = self._detect_dimension_lines(image, page_num)              #  find measurement lines, arrows, dimension text
        elements.extend(dimension_elements)
        return elements


    def _detect_dimension_lines(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:      # Detect dimension lines and measurements
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))        # create 40x1 and 1x40 kernels to detect horizontal/vertical lines

        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)        # use MORPH_OPEN to isolate straight lines from noise

        contours_h, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_v, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)            # find contours in the resulting line masks

        elements = []
        for contour in contours_h + contours_v:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 or h > 50:  # Filter small lines
                bbox = BoundingBox(x=x, y=y, width=w, height=h, confidence=0.7)
                element = ExtractedElement(
                    text="[DIMENSION_LINE]",
                    element_type='dimension',
                    bbox=bbox,
                    page_number=page_num,
                    confidence=0.7,
                    metadata={'detection_method': 'dimension_line'}
                )
                elements.append(element)

        return elements


    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        # converts input array elements to absolute values and then scales the result by alpha and adds beta
        # output_pixel = saturate(input_pixel * alpha + beta)
        # alpha: contrast of the image    beta: brightness of the image


class OCREngine:

    def __init__(self, preferred_provider: str = "tesseract"):          # multiple OCR providers with fallback hierarchy
        self.preferred_provider = preferred_provider            # set preferred OCR provider (defaults to "tesseract")
        self.available_providers = self._check_available_providers()        # call _check_available_providers() to detect which OCR services are available

        # try:
        #     self.vision_client = vision.ImageAnnotatorClient()
        #     self.available_providers.append("google_vision")
        # except Exception as e:
        #     logger.warning(f"Google Vision API not available: {e}")
        #     self.vision_client = None

    def _check_available_providers(self) -> List[str]:
        providers = []

        if TESSERACT_AVAILABLE:
            try:
                pytesseract.get_tesseract_version()
                providers.append("tesseract")
            except:
                logger.warning("Tesseract installed but not accessible")

        try:
            import boto3
            boto3.client('textract', region_name='us-east-1')
            providers.append("aws_textract")
            logger.info("AWS Textract available")
        except Exception as e:
            logger.warning(f"AWS Textract not available: {e}")

        # if GOOGLE_VISION_AVAILABLE:
        #     providers.append("google_vision")

        return providers

    def _aws_textract_extract(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:
        try:
            import boto3

            textract = boto3.client('textract', region_name='us-east-1')

            # Validate and process image for Textract requirements
            processed_image = self._prepare_image_for_textract(image)
            if processed_image is None:
                logger.warning("Image processing failed for Textract")
                return self._basic_text_detection(image, page_num)

            # Encode image
            success, buffer = cv2.imencode('.png', processed_image)
            if not success:
                logger.warning("Failed to encode image for Textract")
                return self._basic_text_detection(image, page_num)

            image_bytes = buffer.tobytes()

            # Validate file size (Textract limit: 10MB)
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB
                logger.warning(f"Image too large for Textract: {len(image_bytes) / 1024 / 1024:.1f}MB")
                return self._basic_text_detection(image, page_num)

            # Call Textract
            response = textract.detect_document_text(
                Document={'Bytes': image_bytes}
            )

            elements = []
            for block in response['Blocks']:
                if block['BlockType'] == 'WORD':
                    bbox_data = block['Geometry']['BoundingBox']
                    height, width = processed_image.shape[:2]

                    bbox = BoundingBox(
                        x=int(bbox_data['Left'] * width),
                        y=int(bbox_data['Top'] * height),
                        width=int(bbox_data['Width'] * width),
                        height=int(bbox_data['Height'] * height),
                        confidence=block['Confidence'] / 100
                    )

                    element = ExtractedElement(
                        text=block['Text'],
                        element_type='text',
                        bbox=bbox,
                        page_number=page_num,
                        confidence=block['Confidence'] / 100,
                        metadata={
                            'provider': 'aws_textract',
                            'block_id': block['Id']
                        }
                    )
                    elements.append(element)

            logger.info(f"AWS Textract extracted {len(elements)} elements")
            return elements

        except Exception as e:
            logger.error(f"AWS Textract extraction failed: {e}")
            return self._basic_text_detection(image, page_num)

    def _prepare_image_for_textract(self, image: np.ndarray) -> np.ndarray:
        try:
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                # Convert grayscale to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Check dimensions (Textract limits: 50x50 min, 10000x10000 max)
            height, width = image_rgb.shape[:2]

            if height < 50 or width < 50:
                logger.warning(f"Image too small for Textract: {width}x{height}")
                return None

            if height > 10000 or width > 10000:
                scale = min(10000 / width, 10000 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")

            estimated_size = height * width * 3 * 2  # Conservative estimate
            if estimated_size > 8 * 1024 * 1024:  # 8MB threshold for safety
                scale = (8 * 1024 * 1024 / estimated_size) ** 0.5
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized for file size: {new_width}x{new_height}")

            return image_rgb

        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            return None

    def _tesseract_extract(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:
        try:
            configs = [
                '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}/-" ',
                '--psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}/-" ',
                '--psm 3',
                '--psm 11',
            ]

            best_elements = []
            best_confidence = 0

            for config in configs:
                try:
                    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
                    elements = self._process_tesseract_data(data, page_num)

                    if elements:
                        avg_conf = sum(e.confidence for e in elements) / len(elements)
                        if avg_conf > best_confidence:
                            best_confidence = avg_conf
                            best_elements = elements
                except Exception as e:
                    logger.debug(f"OCR config failed: {config}, error: {e}")
                    continue

            return best_elements

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return self._basic_text_detection(image, page_num)

    def _process_tesseract_data(self, data: dict, page_num: int) -> List[ExtractedElement]:
        elements = []

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])

            if (text and len(text) > 1 and conf > 30 and
                    not text.isspace() and text.isprintable() and
                    not all(c in '.-_|' for c in text)):

                bbox = BoundingBox(
                    x=data['left'][i],
                    y=data['top'][i],
                    width=data['width'][i],
                    height=data['height'][i],
                    confidence=conf / 100
                )

                element = ExtractedElement(
                    text=text,
                    element_type='text',
                    bbox=bbox,
                    page_number=page_num,
                    confidence=conf / 100,
                    metadata={
                        'word_num': data['word_num'][i],
                        'block_num': data['block_num'][i],
                        'line_num': data['line_num'][i]
                    }
                )
                elements.append(element)

        return elements

    def _basic_text_detection(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:
        elements = []

        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)

                if (w > 15 and h > 8 and 0.1 < aspect_ratio < 20 and area > 100):
                    try:
                        text_region = gray[y:y + h, x:x + w]

                        text = pytesseract.image_to_string(text_region, config='--psm 8').strip()

                        if text and len(text) > 1 and not text.isspace():
                            bbox = BoundingBox(x=x, y=y, width=w, height=h, confidence=0.6)

                            element = ExtractedElement(
                                text=text,
                                element_type='text',
                                bbox=bbox,
                                page_number=page_num,
                                confidence=0.6,
                                metadata={'detection_method': 'basic_with_ocr'}
                            )
                            elements.append(element)
                    except:
                        bbox = BoundingBox(x=x, y=y, width=w, height=h, confidence=0.4)
                        element = ExtractedElement(
                            text="[DETECTED_TEXT_REGION]",
                            element_type='text',
                            bbox=bbox,
                            page_number=page_num,
                            confidence=0.4,
                            metadata={'detection_method': 'basic_fallback'}
                        )
                        elements.append(element)

        except Exception as e:
            logger.warning(f"Enhanced basic text detection error: {e}")

        return elements


    # def _google_vision_extract(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:
    #     try:
    #         _, buffer = cv2.imencode('.png', image)
    #         image_content = buffer.tobytes()
    #
    #         vision_image = vision.Image(content=image_content)
    #
    #         response = self.vision_client.text_detection(image=vision_image)
    #         annotations = response.text_annotations
    #
    #         elements = []
    #         for annotation in annotations[1:]:
    #             vertices = annotation.bounding_poly.vertices
    #
    #             x_coords = [v.x for v in vertices]
    #             y_coords = [v.y for v in vertices]
    #
    #             bbox = BoundingBox(
    #                 x=min(x_coords),
    #                 y=min(y_coords),
    #                 width=max(x_coords) - min(x_coords),
    #                 height=max(y_coords) - min(y_coords),
    #                 confidence=0.9
    #             )
    #
    #             element = ExtractedElement(
    #                 text=annotation.description,
    #                 element_type='text',
    #                 bbox=bbox,
    #                 page_number=page_num,
    #                 confidence=0.9,
    #                 metadata={'provider': 'google_vision'}
    #             )
    #             elements.append(element)
    #
    #         return elements
    #
    #     except Exception as e:
    #         logger.error(f"Google Vision extraction failed: {e}")
    #         return self._tesseract_extract(image, page_num)


class OCREvaluator:
    def __init__(self):
        self.engineering_terms = {
            'building_codes': ['IBC', 'OBC', 'NFPA', 'ASTM', 'AISI'],
            'materials': ['Grade', 'ga', 'gauge', 'steel', 'concrete'],
            'dimensions': ['SF', 'ft', 'in', 'mm', 'psf', 'PSF'],
            'standards': ['ASCE', 'AWS', 'AISC', 'TMS', 'ANSI']
        }

    # 1. CHARACTER ERROR RATE (CER)
    def calculate_cer(self, ground_truth: str, predicted: str) -> float:
        if not ground_truth:
            return 100.0 if predicted else 0.0

        insertions, deletions, substitutions = self._levenshtein_operations(ground_truth, predicted)

        total_errors = insertions + deletions + substitutions
        cer = (total_errors / len(ground_truth)) * 100

        return min(cer, 100.0)

    def calculate_cer_carrasco(self, ground_truth: str, predicted: str) -> float:
        if not ground_truth:
            return 100.0 if predicted else 0.0

        i, d, s = self._levenshtein_operations(ground_truth, predicted)
        correct = len(ground_truth) - d - s

        carrasco_cer = (i + d + s) / (i + d + s + correct) * 100
        return carrasco_cer

    # 2. WORD ERROR RATE (WER)
    def calculate_wer(self, ground_truth: str, predicted: str) -> float:
        gt_words = ground_truth.split()
        pred_words = predicted.split()

        if not gt_words:
            return 100.0 if pred_words else 0.0

        i, d, s = self._levenshtein_operations_words(gt_words, pred_words)
        wer = (i + d + s) / len(gt_words) * 100

        return min(wer, 100.0)

    # 3. BLEU SCORE
    def calculate_bleu(self, ground_truth: str, predicted: str, n=4) -> float:
        gt_tokens = ground_truth.split()
        pred_tokens = predicted.split()

        if not pred_tokens:
            return 0.0

        precisions = []
        for i in range(1, n + 1):
            gt_ngrams = self._get_ngrams(gt_tokens, i)
            pred_ngrams = self._get_ngrams(pred_tokens, i)

            if not pred_ngrams:
                precisions.append(0.0)
                continue

            matches = sum(min(gt_ngrams.get(ngram, 0), count)
                          for ngram, count in pred_ngrams.items())
            precision = matches / len(pred_tokens) if len(pred_tokens) >= i else 0.0
            precisions.append(precision)

        bp = min(1.0, len(pred_tokens) / len(gt_tokens)) if gt_tokens else 0.0

        if all(p > 0 for p in precisions):
            bleu = bp * np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            bleu = 0.0

        return bleu * 100

    # 4. ROUGE-L SCORE
    def calculate_rouge_l(self, ground_truth: str, predicted: str, beta=1.0) -> float:
        gt_tokens = ground_truth.split()
        pred_tokens = predicted.split()

        if not gt_tokens or not pred_tokens:
            return 0.0

        lcs_length = self._lcs_length(gt_tokens, pred_tokens)

        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(gt_tokens)

        if precision + recall == 0:
            return 0.0

        f_score = ((1 + beta ** 2) * precision * recall) / (recall + beta ** 2 * precision)
        return f_score * 100

    # 5. IoU FOR SPATIAL ACCURACY
    def calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:

        x1_1, y1_1 = bbox1['x'], bbox1['y']
        x2_1, y2_1 = x1_1 + bbox1['width'], y1_1 + bbox1['height']

        x1_2, y1_2 = bbox2['x'], bbox2['y']
        x2_2, y2_2 = x1_2 + bbox2['width'], y1_2 + bbox2['height']

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    # 6. FIELD-LEVEL ACCURACY
    def calculate_field_accuracy(self, ground_truth_dict: Dict, predicted_dict: Dict) -> Dict[str, float]:
        field_accuracies = {}

        for field in ground_truth_dict:
            gt_values = set(ground_truth_dict.get(field, []))
            pred_values = set(predicted_dict.get(field, []))

            if not gt_values:
                accuracy = 100.0 if not pred_values else 0.0
            else:
                matches = len(gt_values.intersection(pred_values))
                accuracy = (matches / len(gt_values)) * 100

            field_accuracies[field] = accuracy

        return field_accuracies

    # 7. TECHNICAL TERMINOLOGY ACCURACY
    def calculate_technical_accuracy(self, ground_truth: str, predicted: str) -> Dict[str, float]:
        accuracies = {}

        for category, terms in self.engineering_terms.items():
            gt_found = set()
            pred_found = set()

            for term in terms:
                if re.search(rf'\b{re.escape(term)}\b', ground_truth, re.IGNORECASE):
                    gt_found.add(term.lower())
                if re.search(rf'\b{re.escape(term)}\b', predicted, re.IGNORECASE):
                    pred_found.add(term.lower())

            if gt_found:
                matches = len(gt_found.intersection(pred_found))
                accuracy = (matches / len(gt_found)) * 100
            else:
                accuracy = 100.0 if not pred_found else 0.0

            accuracies[category] = accuracy

        return accuracies

    # HELPER METHODS
    def _levenshtein_operations(self, s1: str, s2: str) -> Tuple[int, int, int]:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                       dp[i][j - 1],  # insertion
                                       dp[i - 1][j - 1])  # substitution

        # Backtrack to count operations
        i, j = m, n
        insertions = deletions = substitutions = 0

        while i > 0 or j > 0:
            if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                deletions += 1
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                insertions += 1
                j -= 1

        return insertions, deletions, substitutions

    def _levenshtein_operations_words(self, words1: List[str], words2: List[str]) -> Tuple[int, int, int]:
        return self._levenshtein_operations(' '.join(words1), ' '.join(words2))

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple, int]:
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def evaluate_complete(self, ground_truth: str, predicted: str,
                          gt_fields: Dict = None, pred_fields: Dict = None,
                          gt_bboxes: List[Dict] = None, pred_bboxes: List[Dict] = None) -> OCRMetrics:
        """Complete evaluation returning all metrics"""

        # Basic text metrics
        cer = self.calculate_cer(ground_truth, predicted)
        wer = self.calculate_wer(ground_truth, predicted)
        bleu = self.calculate_bleu(ground_truth, predicted)
        rouge_l = self.calculate_rouge_l(ground_truth, predicted)

        # Spatial metrics
        iou_scores = []
        if gt_bboxes and pred_bboxes:
            for gt_box in gt_bboxes:
                best_iou = max([self.calculate_iou(gt_box, pred_box)
                                for pred_box in pred_bboxes], default=0.0)
                iou_scores.append(best_iou)

        # Field-level accuracy
        field_accuracy = {}
        if gt_fields and pred_fields:
            field_accuracy = self.calculate_field_accuracy(gt_fields, pred_fields)

        return OCRMetrics(
            cer=cer,
            wer=wer,
            bleu_score=bleu,
            rouge_l=rouge_l,
            iou_scores=iou_scores,
            field_accuracy=field_accuracy
        )


class OCRResultEvaluator:

    def __init__(self):
        self.evaluator = OCREvaluator()

    def evaluate_extraction_result(self, result: ExtractionResult, ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:

        gt_text = ground_truth_data.get('text', '')
        predicted_text = result.extracted_text

        cer = self.evaluator.calculate_cer(gt_text, predicted_text)
        cer_carrasco = self.evaluator.calculate_cer_carrasco(gt_text, predicted_text)
        wer = self.evaluator.calculate_wer(gt_text, predicted_text)
        bleu = self.evaluator.calculate_bleu(gt_text, predicted_text)
        rouge_l = self.evaluator.calculate_rouge_l(gt_text, predicted_text)

        iou_scores = []
        if 'bboxes' in ground_truth_data and result.elements:
            gt_bboxes = ground_truth_data['bboxes']
            pred_bboxes = [
                {
                    'x': e.bbox.x, 'y': e.bbox.y,
                    'width': e.bbox.width, 'height': e.bbox.height
                }
                for e in result.elements if e.bbox
            ]

            for gt_box in gt_bboxes:
                if pred_bboxes:
                    best_iou = max([self.evaluator.calculate_iou(gt_box, pred_box)
                                    for pred_box in pred_bboxes], default=0.0)
                    iou_scores.append(best_iou)

        field_accuracy = {}
        if 'structured_fields' in ground_truth_data:
            gt_fields = ground_truth_data['structured_fields']
            pred_fields = result.structured_data
            field_accuracy = self.evaluator.calculate_field_accuracy(gt_fields, pred_fields)

        technical_accuracy = self.evaluator.calculate_technical_accuracy(gt_text, predicted_text)

        confidence_stats = self._analyze_confidence_distribution(result.elements)

        processing_metrics = {
            'total_elements': len(result.elements),
            'avg_confidence': sum([e.confidence for e in result.elements]) / len(
                result.elements) if result.elements else 0,
            'high_confidence_elements': len([e for e in result.elements if e.confidence > 0.7]),
            'low_confidence_elements': len([e for e in result.elements if e.confidence < 0.3]),
            'spatial_coverage': self._calculate_spatial_coverage(result.elements),
            'table_detection_rate': len(result.tables),
        }

        evaluation_results = {
            'text_accuracy_metrics': {
                'character_error_rate': cer,
                'character_error_rate_carrasco': cer_carrasco,
                'word_error_rate': wer,
                'bleu_score': bleu,
                'rouge_l_score': rouge_l,
                'accuracy_grade': self._grade_accuracy(cer, wer)
            },
            'spatial_accuracy_metrics': {
                'iou_scores': iou_scores,
                'mean_iou': sum(iou_scores) / len(iou_scores) if iou_scores else 0,
                'iou_at_50': len([s for s in iou_scores if s > 0.5]) / len(iou_scores) if iou_scores else 0,
                'iou_at_75': len([s for s in iou_scores if s > 0.75]) / len(iou_scores) if iou_scores else 0,
            },
            'field_accuracy_metrics': field_accuracy,
            'technical_accuracy_metrics': technical_accuracy,
            'confidence_metrics': confidence_stats,
            'processing_performance_metrics': processing_metrics,
            'overall_score': self._calculate_overall_score(cer, wer, field_accuracy, confidence_stats),
            'recommendations': self._generate_recommendations(cer, wer, confidence_stats, technical_accuracy)
        }

        return evaluation_results

    def _analyze_confidence_distribution(self, elements: List[ExtractedElement]) -> Dict[str, float]:
        if not elements:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'q25': 0, 'q75': 0, 'high_confidence_ratio': 0}

        confidences = [e.confidence for e in elements]

        return {
            'mean': sum(confidences) / len(confidences),
            'std': np.std(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'q25': np.percentile(confidences, 25),
            'q75': np.percentile(confidences, 75),
            'high_confidence_ratio': len([c for c in confidences if c > 0.7]) / len(confidences)
        }

    def _calculate_spatial_coverage(self, elements: List[ExtractedElement]) -> float:
        if not elements:
            return 0.0

        # standard page size (8.5" x 11" = 612 x 792 points)
        page_area = 612 * 792

        total_element_area = 0
        for element in elements:
            if element.bbox:
                element_area = element.bbox.width * element.bbox.height
                total_element_area += element_area

        return min(total_element_area / page_area, 1.0)

    def _grade_accuracy(self, cer: float, wer: float) -> str:
        if cer <= 2 and wer <= 5:
            return "EXCELLENT (Industry Standard)"
        elif cer <= 5 and wer <= 10:
            return "GOOD"
        elif cer <= 10 and wer <= 20:
            return "MODERATE"
        else:
            return "POOR - Needs Improvement"

    def _calculate_overall_score(self, cer: float, wer: float, field_accuracy: Dict, confidence_stats: Dict) -> float:

        # Text accuracy score (40% weight)
        text_score = max(0, 100 - (cer + wer) / 2)

        # Field accuracy score (30% weight)
        field_score = 0
        if field_accuracy:
            field_score = sum(field_accuracy.values()) / len(field_accuracy)

        # Confidence score (20% weight)
        confidence_score = confidence_stats.get('mean', 0) * 100

        # Coverage score (10% weight)
        coverage_score = confidence_stats.get('high_confidence_ratio', 0) * 100

        overall = (text_score * 0.4 + field_score * 0.3 + confidence_score * 0.2 + coverage_score * 0.1)
        return round(overall, 2)

    def _generate_recommendations(self, cer: float, wer: float, confidence_stats: Dict, technical_accuracy: Dict) -> \
    List[str]:
        recommendations = []

        if cer > 5:
            recommendations.append("HIGH CER: Consider image preprocessing improvements or OCR engine tuning")

        if wer > 15:
            recommendations.append("HIGH WER: Implement post-processing spell correction for technical terms")

        if confidence_stats.get('mean', 0) < 0.6:
            recommendations.append("LOW CONFIDENCE: Review image quality and OCR parameter settings")

        low_technical_categories = [cat for cat, acc in technical_accuracy.items() if acc < 70]
        if low_technical_categories:
            recommendations.append(f"TECHNICAL TERMS: Improve recognition for: {', '.join(low_technical_categories)}")

        if confidence_stats.get('high_confidence_ratio', 0) < 0.5:
            recommendations.append(
                "QUALITY CONTROL: Less than 50% elements have high confidence - consider manual review workflow")

        return recommendations


class PDFProcessor:

    def __init__(self, ocr_provider: str = "tesseract"):                # Initialize with regex patterns for engineering data extraction
        self.extraction_patterns = self._load_extraction_patterns()
        self.image_processor = ImageProcessor()
        self.ocr_engine = OCREngine(ocr_provider)

    def _load_extraction_patterns(self) -> Dict[str, List[str]]:            # regex patterns I found for extracting engineering parameters by the help of llms
        return {
            # parameter 1
            'building_codes': [
                r'(?:IBC|International Building Code)\s*(\d{4})',
                r'(?:OBC|Ohio Building Code)\s*(\d{4})',
                r'(?:AISI|ASTM|AWS|AISC|TMS)\s*([A-Z]?\s*\d+(?:[/-][A-Z]?\d+)*)',
                r'(?:NFPA|NEC)\s*(\d+)',
                r'(?:ANSI)\s*([A-Z]\d+(?:\.\d+)*)',
                r'(?:IECC|IFGC)\s*(?:(\d{4})\s*)?',
                r'Construction Type[:\s]*([A-Z0-9-]+)',
                r'Occupancy Classification[:\s]*([A-Z0-9-,\s]+)',
            ],
            # parameter 2
            'material_specs': [
                r'(\d+)\s*(?:ga|gauge|gage)\b',
                r'(\d+(?:\.\d+)?)\s*(?:inch|in\.?|")\s*thick',
                r'Grade\s*([A-Z]?\d+)',
                r'Type\s*([A-Z]?\d+[A-Z]*)',
                r'ASTM\s*([A-Z]\s*\d+(?:[/-][A-Z]?\d+)*)',
                r'Thickness[:\s]*(\d+(?:\.\d+)?)\s*(?:inch|in\.?|mm)',
                r'Species[:\s]*([^\n,]+)',
                r'Temper\s*([A-Z]\d+)',
            ],
            # parameter 3
            'project_info': [
                r'Project[:\s]+([^\n]+)',
                r'(?:Location|Address)[:\s]+([^\n]+)',
                r'Date[:\s]+([^\n]+)',
                r'(?:Section|SECTION)\s+(\d+(?:\.\d+)*)\s*[-–]\s*([^\n]+)',
                r'Division\s+(\d+)\s*[-–]\s*([^\n]+)',
                r'Building\s*(?:Height|Area)[:\s]*([^\n]+)',
                r'Shell Permit[:\s]*([^\n]+)',
                r'(?:Tenant\s*Space|TENANT\s*SPACE)\s*([A-Z0-9]+)',
            ],
            # parameter 4
            'dimensions': [
                r'(\d+(?:\.\d+)?)\s*(?:SF|sq\.?\s*ft\.?)',
                r'(\d+(?:\.\d+)?)\s*(?:ft|feet|\')\s*(?:x|\*|by)\s*(\d+(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:inches?|in\.?|")\s*(?:high|width|length|thick)',
                r'(\d+(?:\.\d+)?)\s*(?:mm)',
                r'Building\s*Area[:\s]*(\d+(?:,\d+)?)\s*SF',
                r'(\d+(?:\.\d+)?)\s*(?:inches?|in\.?|")\s*O\.?C\.?',
            ],
            # parameter 5
            'fire_protection': [
                r'Fire\s*(?:Rating|Resistance)[:\s]*(\d+(?:\.\d+)?)\s*(?:hour|hr)',
                r'Fire\s*Separation[:\s]*([^\n]+)',
                r'(?:Sprinkler|NFPA\s*13)[:\s]*([^\n]+)',
                r'Fire\s*Alarm[:\s]*([^\n]+)',
                r'Fire\s*Extinguisher[:\s]*([^\n]+)',
            ],
            # parameter 6
            'environmental_conditions': [
                r'(?:Dead|Live|Wind)\s*Load[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
                r'(?:Temperature|Humidity)[:\s]*([^\n]+)',
                r'Deflection[:\s]*L\s*/\s*(\d+)',
                r'Environmental\s*conditions[:\s]*([^\n]+)',
            ],
            # parameter 7
            'manufacturers': [
                r'ClarkDietrich',
                r'COLLABORATIVE DESIGN',
                r'(?:Manufacturer|MANUFACTURER)[:\s]*([^\n]+)',
                r'Product\s*Data[:\s]*([^\n]+)',
            ],
            # parameter 8
            'specification_structure': [
                r'DIVISION\s+(\d+)\s*[-–—]\s*([^\n\r]+)',
                r'SECTION\s+(\d+(?:\s+\d+)*)\s*[-–—]\s*([^\n\r]+)',
                r'PART\s+([123])[:\s]*([^\n\r]+)',
                r'(?:SUMMARY|REFERENCES|SUBMITTALS|QUALITY ASSURANCE|PRODUCTS|EXECUTION)',
                r'(\d{2}\s+\d{2}\s+\d{2})\s*[-–—]\s*([^\n\r]+)',  # CSI format
            ],
            # parameter 9
            'quality_standards': [
                r'Quality\s*Assurance[:\s]*([^\n]+)',
                r'Comply\s*with[:\s]*([^\n]+)',
                r'Standard[:\s]*([^\n]+)',
                r'Warranty[:\s]*([^\n]+)',
                r'(?:Installation|INSTALLATION)\s*(?:method|instruction)[s]?[:\s]*([^\n]+)',
            ]
        }

    def process_document(self, file_path: Union[str, Path], use_ocr: bool = True) -> ExtractionResult:          # Main entry point for PDF processing
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        document_id = str(uuid.uuid4())
        logger.info(f"Processing document: {file_path.name}")

        if file_path.suffix.lower() == '.pdf':
            return self._process_pdf(file_path, document_id, use_ocr)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def _process_pdf(self, pdf_path: Path, document_id: str, use_ocr: bool) -> ExtractionResult:
        all_text = ""
        all_elements = []
        all_tables = []
        all_images = []

        try:
            # Extract text using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                page_texts = []

                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    all_text += page_text + "\n"
                    page_texts.append({
                        'page_number': page_num,
                        'text': page_text,
                        'char_count': len(page_text)
                    })
                    page_tables = page.extract_tables()
                    for table_idx, table_data in enumerate(page_tables):
                        if table_data and any(any(cell for cell in row if cell) for row in table_data):
                            spatial_table = SpatialTable(
                                rows=table_data,
                                headers=table_data[0] if table_data else [],
                                bbox=BoundingBox(0, 0, page.width, page.height, 0.8),
                                page_number=page_num,
                                confidence=0.8,
                                structure_type="standard"
                            )
                            all_tables.append(spatial_table)
            if use_ocr:
                ocr_elements, ocr_images = self._extract_with_ocr(pdf_path, total_pages)
                all_elements.extend(ocr_elements)
                all_images.extend(ocr_images)

            spatial_data = self._perform_spatial_analysis(all_elements, all_tables, total_pages)


            structured_data = self._extract_structured_data(all_text)

            confidence = self._calculate_confidence(structured_data, all_text, all_elements)

            metadata = {
                'page_texts': page_texts,
                'total_characters': len(all_text),
                'table_count': len(all_tables),
                'image_count': len(all_images),
                'ocr_elements_count': len(all_elements),
                'extraction_timestamp': datetime.now().isoformat(),
                'ocr_enabled': use_ocr,
                'available_ocr_providers': self.ocr_engine.available_providers,
                'patterns_matched': sum(len(v) for v in structured_data.values() if isinstance(v, list))
            }

            result = ExtractionResult(
                document_id=document_id,
                filename=pdf_path.name,
                total_pages=total_pages,
                processing_method="pdfplumber_ocr",
                extracted_text=all_text,
                structured_data=structured_data,
                tables=all_tables,
                elements=all_elements,
                images=all_images,
                spatial_analysis=spatial_data,
                metadata=metadata,
                confidence=confidence
            )

            logger.info(f"Successfully processed {pdf_path.name}: {total_pages} pages, confidence: {confidence:.2f}")       # ext -> tables -> OCR -> pattern matching -> confidence scoring
            return result

        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
            raise

    def _extract_with_ocr(self, pdf_path: Path, total_pages: int) -> Tuple[List[ExtractedElement], List[Dict]]:
        elements = []
        images = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(total_pages):
                page = doc[page_num]

                # Convert page to image with reduced scaling
                mat = fitz.Matrix(2.0, 2.0)  # Reduced from 3.0 to 2.0
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                # Convert to OpenCV format
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Log image dimensions for debugging
                logger.info(f"Page {page_num + 1} image size: {img.shape[1]}x{img.shape[0]}")

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                images.append({
                    'page_number': page_num + 1,
                    'format': 'png',
                    'size': pix.width * pix.height,
                    'dimensions': (pix.width, pix.height)
                })

                enhanced_img = self.image_processor.enhance_image(img)

                ocr_elements = self._ocr_with_fallbacks(enhanced_img, page_num + 1)
                elements.extend(ocr_elements)

                table_regions = self._detect_tables(img, page_num + 1)
                elements.extend(table_regions)

            doc.close()

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")

        return elements, images

    def _refine_table_bbox(self, gray_image: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        try:
            table_region = gray_image[y:y + h, x:x + w]

            rows_with_content = np.where(np.sum(table_region < 200, axis=1) > 5)[0]
            cols_with_content = np.where(np.sum(table_region < 200, axis=0) > 5)[0]

            if len(rows_with_content) > 0 and len(cols_with_content) > 0:
                new_y = y + rows_with_content[0]
                new_x = x + cols_with_content[0]
                new_h = rows_with_content[-1] - rows_with_content[0]
                new_w = cols_with_content[-1] - cols_with_content[0]

                return (new_x, new_y, new_w, new_h)

        except Exception:
            pass

        return (x, y, w, h)


    def _ocr_with_fallbacks(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:
        elements = []

        # Try Tesseract first
        if TESSERACT_AVAILABLE:
            try:
                elements = self._tesseract_extract(image, page_num)
                if elements:
                    return elements
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")

        try:
            elements = self._opencv_text_detection(image, page_num)
            if elements:
                return elements
        except Exception as e:
            logger.warning(f"OpenCV text detection failed: {e}")

        try:
            elements = self._basic_text_detection(image, page_num)
        except Exception as e:
            logger.warning(f"Basic text detection failed: {e}")

        return elements

    def _opencv_text_detection(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:
        elements = []

        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)

            for region in regions:
                x, y, w, h = cv2.boundingRect(region)

                if w > 10 and h > 5 and w < image.shape[1] * 0.8:
                    bbox = BoundingBox(x=x, y=y, width=w, height=h, confidence=0.6)

                    element = ExtractedElement(
                        text="[TEXT_REGION]",
                        element_type='text',
                        bbox=bbox,
                        page_number=page_num,
                        confidence=0.6,
                        metadata={'detection_method': 'opencv_mser'}
                    )
                    elements.append(element)

        except Exception as e:
            logger.warning(f"OpenCV text detection error: {e}")

        return elements

    def _detect_tables(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:
        elements = []

        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:
                    x, y, w, h = cv2.boundingRect(contour)

                    precise_bbox = self._refine_table_bbox(gray, x, y, w, h)

                    if precise_bbox:
                        bbox = BoundingBox(
                            x=precise_bbox[0],
                            y=precise_bbox[1],
                            width=precise_bbox[2],
                            height=precise_bbox[3],
                            confidence=0.8
                        )

                        element = ExtractedElement(
                            text="[TABLE_REGION]",
                            element_type='table',
                            bbox=bbox,
                            page_number=page_num,
                            confidence=0.8,
                            metadata={'table_area': area, 'detection_method': 'line_detection'}
                        )
                        elements.append(element)

        except Exception as e:
            logger.warning(f"Table detection error: {e}")

        return elements

    def _perform_spatial_analysis(self, elements: List[ExtractedElement], tables: List[SpatialTable], total_pages: int) -> Dict[str, Any]:
        analysis = {
            'page_layout_analysis': [],
            'text_regions': [],
            'table_regions': [],
            'drawing_regions': [],
            'spatial_relationships': []
        }

        for page_num in range(1, total_pages + 1):
            page_elements = [e for e in elements if e.page_number == page_num]
            page_tables = [t for t in tables if t.page_number == page_num]

            text_regions = self._analyze_text_regions(page_elements)
            analysis['text_regions'].extend(text_regions)

            for table in page_tables:
                analysis['table_regions'].append({
                    'page': page_num,
                    'bbox': asdict(table.bbox),
                    'structure_type': table.structure_type,
                    'row_count': len(table.rows),
                    'confidence': table.confidence
                })

            layout_info = {
                'page': page_num,
                'element_count': len(page_elements),
                'table_count': len(page_tables),
                'text_coverage': self._calculate_text_coverage(page_elements),
                'dominant_regions': self._identify_dominant_regions(page_elements)
            }
            analysis['page_layout_analysis'].append(layout_info)

        return analysis

    def _analyze_text_regions(self, elements: List[ExtractedElement]) -> List[Dict]:        # group nearby text elements into logical regions
        regions = []
        if not elements:
            return regions

        text_elements = [e for e in elements if e.element_type == 'text' and e.bbox]

        sorted_elements = sorted(text_elements, key=lambda e: e.bbox.y)

        current_region = []
        last_y = 0

        for element in sorted_elements:
            if abs(element.bbox.y - last_y) > 20:  # New region threshold
                if current_region:
                    region_text = " ".join([e.text for e in current_region])
                    regions.append({
                        'text': region_text,
                        'element_count': len(current_region),
                        'avg_confidence': sum(e.confidence for e in current_region) / len(current_region),
                        'bbox': self._calculate_region_bbox(current_region)
                    })
                current_region = [element]
            else:
                current_region.append(element)
            last_y = element.bbox.y

        if current_region:
            region_text = " ".join([e.text for e in current_region])
            regions.append({
                'text': region_text,
                'element_count': len(current_region),
                'avg_confidence': sum(e.confidence for e in current_region) / len(current_region),
                'bbox': self._calculate_region_bbox(current_region)
            })

        return regions

    def _calculate_region_bbox(self, elements: List[ExtractedElement]) -> Dict:     # compute bounding box for grouped text elements
        if not elements:
            return {}

        min_x = min(e.bbox.x for e in elements)
        min_y = min(e.bbox.y for e in elements)
        max_x = max(e.bbox.x + e.bbox.width for e in elements)
        max_y = max(e.bbox.y + e.bbox.height for e in elements)

        return {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }

    def _calculate_text_coverage(self, elements: List[ExtractedElement]) -> float:      # measure percentage of page covered by text
        if not elements:
            return 0.0

        text_elements = [e for e in elements if e.element_type == 'text' and e.bbox]
        if not text_elements:
            return 0.0

        total_text_area = sum(e.bbox.width * e.bbox.height for e in text_elements)
        page_area = 612 * 792

        return min(total_text_area / page_area, 1.0)

    def _identify_dominant_regions(self, elements: List[ExtractedElement]) -> List[str]:        # find most common element types per page
        if not elements:
            return []

        type_counts = {}
        for element in elements:
            type_counts[element.element_type] = type_counts.get(element.element_type, 0) + 1

        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_types[:3]]

    def _extract_structured_data(self, text: str) -> Dict[str, Any]:            # apply regex patterns to extract categorized information
        results = {}

        for category, patterns in self.extraction_patterns.items():
            category_results = []

            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if matches:
                        if isinstance(matches[0], tuple):
                            category_results.extend([' '.join(match).strip() for match in matches])
                        else:
                            category_results.extend([match.strip() for match in matches])
                except re.error as e:
                    logger.warning(f"Regex error in pattern '{pattern}': {e}")
                    continue

            if category_results:
                cleaned_results = list(set([
                    result for result in category_results
                    if result and len(result) > 1
                ]))
                if cleaned_results:
                    results[category] = cleaned_results

        results.update(self._extract_contextual_data(text))

        return results

    def _extract_contextual_data(self, text: str) -> Dict[str, Any]:            # add document type detection and domain-specific extraction
        context_data = {}

        doc_type = self._detect_document_type(text)
        context_data['document_type'] = doc_type
        context_data['document_characteristics'] = self._analyze_document_characteristics(text)

        if 'specification' in doc_type.lower():
            context_data.update(self._extract_specification_data(text))
        elif 'design' in doc_type.lower() or 'engineering' in doc_type.lower():
            context_data.update(self._extract_engineering_data(text))
        elif 'plan' in doc_type.lower() or 'drawing' in doc_type.lower():
            context_data.update(self._extract_drawing_data(text))

        return context_data

    def _analyze_document_characteristics(self, text: str) -> Dict[str, Any]:           # detect technical drawings, specs, codes, materials
        text_lower = text.lower()

        characteristics = {
            'has_technical_drawings': bool(re.search(r'elevation|plan|section|detail', text_lower)),
            'has_specifications': bool(re.search(r'specification|astm|section \d+', text_lower)),
            'has_building_codes': bool(re.search(r'building code|ibc|obc', text_lower)),
            'has_dimensions': bool(re.search(r'\d+(?:\.\d+)?\s*(?:ft|in|mm)', text_lower)),
            'has_materials': bool(re.search(r'steel|concrete|wood|masonry', text_lower)),
            'has_fire_protection': bool(re.search(r'fire|sprinkler|nfpa', text_lower)),
            'document_complexity': self._assess_complexity(text),
            'language_indicators': self._detect_language_patterns(text)
        }

        return characteristics

    def _assess_complexity(self, text: str) -> str:
        indicators = {
            'simple': ['summary', 'general', 'basic'],
            'moderate': ['specification', 'detail', 'requirements'],
            'complex': ['engineering', 'structural', 'technical', 'analysis'],
            'very_complex': ['seismic', 'load analysis', 'finite element', 'computational']
        }

        text_lower = text.lower()
        scores = {}

        for level, terms in indicators.items():
            score = sum(1 for term in terms if term in text_lower)
            scores[level] = score

        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        return 'moderate'

    def _detect_language_patterns(self, text: str) -> List[str]:            # identify regulatory, technical, procedural language
        patterns = {
            'regulatory': ['shall', 'comply', 'conform', 'required', 'mandatory'],
            'technical': ['specification', 'standard', 'grade', 'type', 'class'],
            'procedural': ['install', 'apply', 'prepare', 'clean', 'provide'],
            'quality': ['quality', 'assurance', 'control', 'inspection', 'testing']
        }

        text_lower = text.lower()
        detected = []

        for pattern_type, keywords in patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append(pattern_type)

        return detected

    def _extract_drawing_data(self, text: str) -> Dict[str, Any]:           # detect figure references, scales, drawing types
        drawing_data = {}

        drawing_refs = re.findall(r'(?:Figure|Fig\.?)\s+(\d+(?:\.\d+)*)', text, re.IGNORECASE)
        if drawing_refs:
            drawing_data['drawing_references'] = drawing_refs

        scales = re.findall(r'(\d+(?:/\d+)?"\s*=\s*\d+(?:\'\-?\d+"?)?)', text)
        if scales:
            drawing_data['scales'] = scales

        drawing_types = re.findall(r'(elevation|plan|section|detail|profile)', text, re.IGNORECASE)
        if drawing_types:
            drawing_data['drawing_types'] = list(set(drawing_types))

        return drawing_data

    def _detect_document_type(self, text: str) -> str:      # classify document (architectural plan, specification, etc.)
        text_lower = text.lower()

        type_indicators = {
            'architectural_plan': ['floor plan', 'elevation', 'building section', 'key plan', 'ada'],
            'construction_specification': ['section', 'division', 'astm', 'part 1', 'part 2', 'part 3',
                                           'specification'],
            'engineering_design': ['structural', 'foundation', 'framing', 'load', 'design criteria'],
            'building_code': ['building code', 'fire code', 'occupancy', 'construction type'],
            'contract_document': ['contract', 'proposal', 'bid', 'general conditions'],
            'technical_manual': ['manual', 'installation', 'procedure', 'maintenance'],
            'fire_protection': ['fire protection', 'sprinkler', 'nfpa', 'fire alarm'],
            'accessibility': ['ada', 'accessibility', 'ansi a117', 'barrier'],
            'material_specification': ['material', 'product data', 'manufacturer', 'grade', 'type']
        }

        scores = {}
        for doc_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            weighted_score = score * (
                10 if any(key in text_lower for key in ['section', 'division', 'specification']) else 1)
            if weighted_score > 0:
                scores[doc_type] = weighted_score

        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        return 'general_construction_document'

    def _extract_specification_data(self, text: str) -> Dict[str, Any]:         # extract CSI sections, divisions, parts from specs
        spec_data = {}

        section_patterns = [
            r'(?:SECTION|Section)\s+(\d+(?:\s+\d+)*)\s*[-–]\s*([^\n]+)',
            r'(\d{2}\s+\d{2}\s+\d{2})\s*[-–]\s*([^\n]+)'  # CSI format
        ]

        sections = []
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            sections.extend([(num.strip(), title.strip()) for num, title in matches])

        if sections:
            spec_data['sections'] = [{'number': num, 'title': title} for num, title in sections]

        division_pattern = r'(?:DIVISION|Division)\s+(\d+)\s*[-–]\s*([^\n]+)'
        divisions = re.findall(division_pattern, text, re.IGNORECASE)
        if divisions:
            spec_data['divisions'] = [{'number': num, 'title': title.strip()} for num, title in divisions]

        part_pattern = r'PART\s+(\d+)[:\s]*([^\n]+)'
        parts = re.findall(part_pattern, text, re.IGNORECASE)
        if parts:
            spec_data['parts'] = [{'number': num, 'title': title.strip()} for num, title in parts]

        return spec_data

    def _extract_engineering_data(self, text: str) -> Dict[str, Any]:           # find design criteria, loads, structural requirements
        eng_data = {}

        criteria_patterns = [
            r'(?:Design|design)\s+(?:criteria|requirements)[:\s]*([^\n]+)',
            r'(?:Performance|performance)\s+(?:criteria|requirements)[:\s]*([^\n]+)',
            r'(?:Load|LOAD)\s+(?:criteria|requirements)[:\s]*([^\n]+)',
        ]

        criteria = []
        for pattern in criteria_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            criteria.extend(matches)

        if criteria:
            eng_data['design_criteria'] = [c.strip() for c in criteria]

        load_patterns = [
            r'(?:Dead|Live|Wind|Snow)\s*Load[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
            r'(?:Seismic|SEISMIC)[:\s]*([^\n]+)',
            r'(?:Deflection|DEFLECTION)[:\s]*([^\n]+)',
        ]

        loads = []
        for pattern in load_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            loads.extend(matches)

        if loads:
            eng_data['structural_loads'] = loads

        return eng_data

    def _calculate_confidence(self, structured_data: Dict[str, Any], text: str, elements: List[ExtractedElement]) -> float:
        total_patterns = sum(len(patterns) for patterns in self.extraction_patterns.values())
        matched_patterns = sum(len(v) for v in structured_data.values() if isinstance(v, list))
        pattern_confidence = min(matched_patterns / max(total_patterns * 0.1, 1), 1.0)

        text_length_factor = min(len(text) / 10000, 1.0)

        ocr_confidence = 0.8
        if elements:
            ocr_elements = [e for e in elements if e.confidence > 0]
            if ocr_elements:
                ocr_confidence = sum(e.confidence for e in ocr_elements) / len(ocr_elements)

        technical_indicators = [
            'astm', 'aisi', 'code', 'specification', 'grade', 'psf', 'gauge',
            'section', 'division', 'nfpa', 'ansi', 'fire', 'structural'
        ]
        technical_score = sum(1 for indicator in technical_indicators if indicator in text.lower())
        technical_factor = min(technical_score / len(technical_indicators), 1.0)

        structure_indicators = ['part 1', 'part 2', 'part 3', 'general', 'products', 'execution']
        structure_score = sum(1 for indicator in structure_indicators if indicator in text.lower())
        structure_factor = min(structure_score / 3, 1.0)  # At least 3 parts expected

        weights = {
            'pattern': 0.3,
            'text_length': 0.1,
            'ocr': 0.2,
            'technical': 0.2,
            'structure': 0.2
        }

        final_confidence = (
                pattern_confidence * weights['pattern'] +
                text_length_factor * weights['text_length'] +
                ocr_confidence * weights['ocr'] +
                technical_factor * weights['technical'] +
                structure_factor * weights['structure']
        )

        return round(final_confidence, 2)

    def export_to_structured_table(self, result: ExtractionResult) -> Dict[str, Any]:   # convert results to organized JSON format for downstream use
        structured_output = {
            'document_metadata': {
                'id': result.document_id,
                'filename': result.filename,
                'pages': result.total_pages,
                'processing_date': result.timestamp.isoformat(),
                'confidence': result.confidence,
                'document_type': result.structured_data.get('document_type', 'unknown')
            },
            'extracted_data_tables': {},
            'spatial_data': result.spatial_analysis,
            'quality_metrics': {
                'ocr_elements': len(result.elements),
                'tables_found': len(result.tables),
                'images_processed': len(result.images),
                'patterns_matched': sum(len(v) for v in result.structured_data.values() if isinstance(v, list))
            }
        }

        for category, data in result.structured_data.items():
            if isinstance(data, list) and data:
                structured_output['extracted_data_tables'][category] = [
                    {'value': item, 'confidence': 0.8, 'source': 'regex_pattern'}
                    for item in data
                ]
            elif isinstance(data, dict):
                structured_output['extracted_data_tables'][category] = data

        if result.tables:
            table_data = []
            for i, table in enumerate(result.tables):
                table_data.append({
                    'table_id': i,
                    'page': table.page_number,
                    'headers': table.headers,
                    'rows': table.rows,
                    'confidence': table.confidence,
                    'bbox': asdict(table.bbox) if table.bbox else None
                })
            structured_output['extracted_data_tables']['tables'] = table_data

        return structured_output

    def save_results(self, result: ExtractionResult, output_path: Union[str, Path], format_type: str = "json") -> None:        # save extraction results in JSON or structured table format
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "json":
            serializable_data = asdict(result)
            serializable_data['timestamp'] = result.timestamp.isoformat()

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)

        elif format_type == "structured_table":
            structured_data = self.export_to_structured_table(result)
            output_path = output_path.with_suffix('.json')

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Results saved to: {output_path}")

    def evaluate_with_ground_truth(self, file_path: Union[str, Path], ground_truth_file: Union[str, Path]) -> Dict[str, Any]:

        result = self.process_document(file_path, use_ocr=True)

        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        evaluator = OCRResultEvaluator()
        evaluation_metrics = evaluator.evaluate_extraction_result(result, ground_truth)

        complete_results = {
            'extraction_result': asdict(result),
            'evaluation_metrics': evaluation_metrics,
            'processing_timestamp': datetime.now().isoformat()
        }

        return complete_results

    def _tesseract_extract(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:
        return self.ocr_engine._tesseract_extract(image, page_num)

    def _basic_text_detection(self, image: np.ndarray, page_num: int) -> List[ExtractedElement]:
        return self.ocr_engine._basic_text_detection(image, page_num)

    def evaluate_ocr_only(self, file_path: Union[str, Path], ground_truth_file: Union[str, Path],ocr_engine: str = "aws_textract") -> Dict[str, Any]:

        file_path = Path(file_path)

        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        doc = fitz.open(file_path)
        page = doc[0]
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        doc.close()

        if ocr_engine == "aws_textract":
            ocr_elements = self.ocr_engine._aws_textract_extract(img, 1)
        elif ocr_engine == "tesseract":
            ocr_elements = self.ocr_engine._tesseract_extract(img, 1)
        else:
            raise ValueError(f"Unsupported OCR engine: {ocr_engine}")

        ocr_text = " ".join([e.text for e in ocr_elements if e.text.strip()])

        print(f"\n=== {ocr_engine.upper()} OCR EVALUATION ===")
        print(f"OCR Elements: {len(ocr_elements)}")
        print(f"OCR Text Length: {len(ocr_text)} chars")
        print(f"Sample: {ocr_text[:200]}...")

        mock_result = ExtractionResult(
            document_id="ocr_test",
            filename=file_path.name,
            total_pages=1,
            processing_method=f"ocr_{ocr_engine}",
            extracted_text=ocr_text,  # USE OCR TEXT, NOT PDF TEXT
            elements=ocr_elements
        )

        evaluator = OCRResultEvaluator()
        evaluation_metrics = evaluator.evaluate_extraction_result(mock_result, ground_truth)

        metrics = evaluation_metrics
        print(f"{ocr_engine.upper()} RESULTS:")
        print(f"   CER: {metrics['text_accuracy_metrics']['character_error_rate']:.2f}%")
        print(f"   WER: {metrics['text_accuracy_metrics']['word_error_rate']:.2f}%")
        print(f"   Score: {metrics['overall_score']:.1f}/100")
        print(f"   Grade: {metrics['text_accuracy_metrics']['accuracy_grade']}")

        return {
            'ocr_engine': ocr_engine,
            'ocr_text': ocr_text,
            'ocr_elements_count': len(ocr_elements),
            'evaluation_metrics': evaluation_metrics
        }

    def compare_ocr_engines(self, file_path: Union[str, Path], ground_truth_file: Union[str, Path]) -> Dict[str, Any]:
        print("=== OCR ENGINE COMPARISON ===")
        results = {}

        # Test AWS Textract
        if "aws_textract" in self.ocr_engine.available_providers:
            try:
                results['aws_textract'] = self.evaluate_ocr_only(file_path, ground_truth_file, "aws_textract")
            except Exception as e:
                print(f"AWS Textract evaluation failed: {e}")

        # Test Tesseract
        if "tesseract" in self.ocr_engine.available_providers:
            try:
                results['tesseract'] = self.evaluate_ocr_only(file_path, ground_truth_file, "tesseract")
            except Exception as e:
                print(f"Tesseract evaluation failed: {e}")

        if len(results) >= 2:
            print("\n=== COMPARISON ===")
            for engine, result in results.items():
                metrics = result['evaluation_metrics']
                print(
                    f"{engine.upper()}: CER {metrics['text_accuracy_metrics']['character_error_rate']:.1f}%, Score {metrics['overall_score']:.1f}/100")

            best = min(results.keys(),
                       key=lambda x: results[x]['evaluation_metrics']['text_accuracy_metrics']['character_error_rate'])
            print(f"Best: {best.upper()}")

        return results


def create_ground_truth_template(pdf_path: str, output_path: str):

    template = {
        "document_info": {
            "source_file": pdf_path,
            "creation_date": datetime.now().isoformat(),
            "annotator": "YOUR_NAME",
            "notes": "Add any relevant notes about this document"
        },
        "text": "PASTE_REFERENCE_TEXT_HERE - This should be the exact text content you expect",  # for case 2
        "structured_fields": {
            "building_codes": ["IBC 2018", "ASTM A36"],  # Expected building codes
            "material_specs": ["Grade 50", "16 gauge"],  # Expected material specifications
            "dimensions": ["100 SF", "12 ft"],  # Expected dimensions
            "project_info": ["Project Name", "Location"],  # Expected project information
            "fire_protection": ["1 hour rating"],  # Expected fire protection info
            "environmental_conditions": ["50 PSF live load"],  # Expected environmental conditions
            "manufacturers": ["ClarkDietrich"],  # Expected manufacturer names
            "quality_standards": ["ASTM compliance"]  # Expected quality standards
        },
        "bboxes": [
            # Format: {"x": left, "y": top, "width": width, "height": height}
            {"x": 100, "y": 50, "width": 200, "height": 30, "text": "Sample text region 1"},
            {"x": 150, "y": 100, "width": 180, "height": 25, "text": "Sample text region 2"}
        ],
        "evaluation_focus": {
            "prioritize_text_accuracy": True,
            "prioritize_spatial_accuracy": False,
            "prioritize_field_extraction": True,
            "ignore_formatting": True
        }
    }

    # Save template
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"Ground truth template created: {output_path}")

    return template


def test_ocr_engines():

    processor = PDFProcessor(ocr_provider="aws_textract")  # Set preferred

    pdf_file = "Arch+Strl_Thru Bulletin 1_CHS-1-20.pdf"
    gt_file = "Arch+Strl_Thru Bulletin 1_CHS-1-20_ground_truth.json"

    print("Testing AWS Textract")
    aws_results = processor.evaluate_ocr_only(pdf_file, gt_file, "aws_textract")

    print("\nTesting Tesseract")
    tesseract_results = processor.evaluate_ocr_only(pdf_file, gt_file, "tesseract")

    comparison = processor.compare_ocr_engines(pdf_file, gt_file)

    return comparison


def create_ground_truth_from_ocr():
    # reference_text = """Michigan State Multicultural Center FARM & SHAW LN MICHIGAN STATE UNIVERSITY MICHIGAN STATE UNIVERSITY 426 AUDITORIUM RD EAST LANSING, MI 48824 517.355.1855 msu.edu SMITHGROUP 500 GRISWOLD SUITE 1700 DETROIT, MI 48226 313.442.8844 smithgroup.com 4/26/2023 11:25:36 AM Plot Date:"""
    reference_text = "Michigan State Multicultural Center FARM SHAW LN MICHIGAN STATE UNIVERSITY 426 AUDITORIUM RD EAST LANSING MI 48824 517.355.1855 msu.edu SMITHGROUP 500 GRISWOLD SUITE 1700 DETROIT MI 48226 313.442.8844 smithgroup.com Plot Date"
    ground_truth = {
        "document_info": {
            "source_file": "Arch+Strl_Thru Bulletin 1_CHS-1-20.pdf",
            "creation_date": "2024-01-01",
            "annotator": "AUTO_GENERATED",
            "notes": "Generated from OCR output for testing"
        },
        "text": reference_text,
        "structured_fields": {
            "building_codes": [],
            "material_specs": [],
            "dimensions": [],
            "project_info": ["Michigan State Multicultural Center"],
            "fire_protection": [],
            "environmental_conditions": [],
            "manufacturers": ["SMITHGROUP"],
            "quality_standards": []
        },
        "bboxes": [],
        "evaluation_focus": {
            "prioritize_text_accuracy": True,
            "prioritize_spatial_accuracy": False,
            "prioritize_field_extraction": False,
            "ignore_formatting": True
        }
    }

    with open("Arch+Strl_Thru Bulletin 1_CHS-1-20_ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


def test_with_ground_truth():
    processor = PDFProcessor(ocr_provider="aws_textract")

    pdf_file = "Arch+Strl_Thru Bulletin 1_CHS-1-20.pdf"
    gt_file = "Arch+Strl_Thru Bulletin 1_CHS-1-20_ground_truth.json"

    # Test AWS Textract
    aws_results = processor.evaluate_ocr_only(pdf_file, gt_file, "aws_textract")
    print(f"AWS CER: {aws_results['evaluation_metrics']['text_accuracy_metrics']['character_error_rate']:.1f}%")

    # Test Tesseract
    tesseract_results = processor.evaluate_ocr_only(pdf_file, gt_file, "tesseract")
    print(
        f"Tesseract CER: {tesseract_results['evaluation_metrics']['text_accuracy_metrics']['character_error_rate']:.1f}%")


def evaluate_all_ocr_methods(processor, pdf_file, gt_file):

    doc = fitz.open(pdf_file)
    page = doc[0]
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    doc.close()

    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)

    results = []

    ocr_methods = [
        ("AWS Textract", "aws_textract"),
        ("Tesseract", "tesseract"),
        ("CDOCR method", "basic"),
        ("OpenCV MSER", "opencv")
    ]

    for method_name, method_type in ocr_methods:
        try:
            if method_type == "aws_textract" and "aws_textract" in processor.ocr_engine.available_providers:
                elements = processor.ocr_engine._aws_textract_extract(img, 1)
            elif method_type == "tesseract" and "tesseract" in processor.ocr_engine.available_providers:
                elements = processor.ocr_engine._tesseract_extract(img, 1)
            elif method_type == "basic":
                elements = processor.ocr_engine._basic_text_detection(img, 1)
            elif method_type == "opencv":
                elements = processor._opencv_text_detection(img, 1)
            else:
                continue

            extracted_text = " ".join(
                [e.text for e in elements if e.text.strip() and not e.text.startswith("[DETECTED")])

            if extracted_text:
                mock_result = ExtractionResult(
                    document_id="test",
                    filename=pdf_file,
                    total_pages=1,
                    processing_method=method_type,
                    extracted_text=extracted_text,
                    elements=elements
                )

                evaluator = OCRResultEvaluator()
                metrics = evaluator.evaluate_extraction_result(mock_result, ground_truth)

                results.append({
                    'Method': method_name,
                    'Elements': len(elements),
                    'Text Length': len(extracted_text),
                    'CER (%)': f"{metrics['text_accuracy_metrics']['character_error_rate']:.1f}",
                    'WER (%)': f"{metrics['text_accuracy_metrics']['word_error_rate']:.1f}",
                    'BLEU': f"{metrics['text_accuracy_metrics']['bleu_score']:.1f}",
                    'ROUGE-L': f"{metrics['text_accuracy_metrics']['rouge_l_score']:.1f}",
                    'Overall Score': f"{metrics['overall_score']:.1f}",
                    'Grade': metrics['text_accuracy_metrics']['accuracy_grade'],
                    'Avg Confidence': f"{sum(e.confidence for e in elements) / len(elements) * 100:.1f}" if elements else "0",
                    'Sample Text': extracted_text[:50] + "..." if len(extracted_text) > 50 else extracted_text
                })
            else:
                results.append({
                    'Method': method_name,
                    'Elements': len(elements),
                    'Text Length': 0,
                    'CER (%)': "N/A",
                    'WER (%)': "N/A",
                    'BLEU': "N/A",
                    'ROUGE-L': "N/A",
                    'Overall Score': "0",
                    'Grade': "NO TEXT",
                    'Avg Confidence': f"{sum(e.confidence for e in elements) / len(elements) * 100:.1f}" if elements else "0",
                    'Sample Text': "No readable text extracted"
                })

        except Exception as e:
            results.append({
                'Method': method_name,
                'Elements': 0,
                'Text Length': 0,
                'CER (%)': "ERROR",
                'WER (%)': "ERROR",
                'BLEU': "ERROR",
                'ROUGE-L': "ERROR",
                'Overall Score': "0",
                'Grade': "FAILED",
                'Avg Confidence': "0",
                'Sample Text': f"Error: {str(e)[:30]}..."
            })

    return pd.DataFrame(results)


def run_comprehensive_evaluation():

    processor = PDFProcessor(ocr_provider="aws_textract")
    pdf_file = "Arch+Strl_Thru Bulletin 1_CHS-1-20.pdf"
    gt_file = "Arch+Strl_Thru Bulletin 1_CHS-1-20_ground_truth.json"

    if not Path(gt_file).exists():
        create_ground_truth_from_ocr()
        print(f"✓ Created ground truth file: {gt_file}")

    print("=" * 80)
    print("COMPREHENSIVE OCR EVALUATION")
    print("=" * 80)

    results_df = evaluate_all_ocr_methods(processor, pdf_file, gt_file)

    main_cols = ['Method', 'Elements', 'CER (%)', 'WER (%)', 'Overall Score', 'Grade', 'Avg Confidence']
    print("\nMAIN RESULTS:")
    print(tabulate(results_df[main_cols], headers='keys', tablefmt='grid', showindex=False))

    detail_cols = ['Method', 'BLEU', 'ROUGE-L', 'Text Length', 'Sample Text']
    print("\nDETAILED METRICS:")
    print(tabulate(results_df[detail_cols], headers='keys', tablefmt='grid', showindex=False))

    numeric_results = results_df[results_df['CER (%)'] != 'ERROR'][results_df['CER (%)'] != 'N/A']
    if not numeric_results.empty:
        numeric_results['CER_numeric'] = pd.to_numeric(numeric_results['CER (%)'])
        best_method = numeric_results.loc[numeric_results['CER_numeric'].idxmin(), 'Method']
        best_cer = numeric_results.loc[numeric_results['CER_numeric'].idxmin(), 'CER (%)']

        print(f"\nBEST PERFORMER: {best_method} (CER: {best_cer}%)")

        working_methods = len(numeric_results)
        avg_cer = numeric_results['CER_numeric'].mean()

        print(f"\nSUMMARY:")
        print(f"   Working Methods: {working_methods}/4")
        print(f"   Average CER: {avg_cer:.1f}%")

        if avg_cer < 5:
            print("   System Grade: EXCELLENT")
        elif avg_cer < 10:
            print("   System Grade: GOOD")
        elif avg_cer < 20:
            print("   System Grade: MODERATE")
        else:
            print("   System Grade: POOR")

    return results_df


def get_ground_truth(pdf = None):
    processor = PDFProcessor()
    result = processor.process_document(pdf)
    print("ACTUAL TEXT:")
    print(result.extracted_text)


def main():
    test_ocr_engines()
    # create_ground_truth_from_ocr()
    create_ground_truth_template("Arch+Strl_Thru Bulletin 1_CHS.pdf", "ground_truth.json")
    test_with_ground_truth()
    results = run_comprehensive_evaluation()
    results.to_csv("comprehensive_ocr_results.csv", index=False)

# if __name__ == "__main__":
    get_ground_truth("Arch+Strl_Thru Bulletin 1_CHS.pdf")
    # main()