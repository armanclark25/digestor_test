"""
Main PDF Processor
"""

import time
import uuid
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import numpy as np

# Core imports
from core.models import ExtractionResult, ProcessingMethod, DocumentType, ExtractedElement, SpatialTable, ImageInfo, ProcessingMetrics
from core.exceptions import PDFProcessorError, FileNotFoundError, UnsupportedFileTypeError, ProcessingError, ValidationError
from core.interfaces import ProcessingPipeline

# Component imports
from config.settings import get_config, initialize_config
from ocr.base import get_ocr_registry
from ocr.aws_textract import create_aws_textract_engine
from ocr.tesseract import create_tesseract_engine
from ocr.opencv import create_opencv_engine
from processors.image import create_image_processor
from processors.pattern import create_pattern_processor
from extractors.pdf_extractor import create_pdf_extractor
from extractors.table_extractor import create_table_extractor
from processors.spatial_analyzer import create_spatial_analyzer
from processors.document_classifier import create_document_classifier
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

# $env:AZURE_DOCUMENT_KEY="key"
# $env:AZURE_DOCUMENT_ENDPOINT="https://cognitiveservices.azure.com/"
# $env:MISTRAL_API_KEY="key"

class PDFProcessor(ProcessingPipeline):
    
    def __init__(self, config_path: Optional[Path] = None):

        self.config = initialize_config(config_path)

        setup_logging(self.config.config.logging)

        self._initialize_components()

        self._stages = []
        self._setup_default_pipeline()
        
        logger.info("PDF Processor initialized successfully")
    
    def _initialize_components(self) -> None:
        try:

            self.ocr_registry = get_ocr_registry()
            self._register_ocr_engines()

            self.image_processor = create_image_processor()
            self.pattern_processor = create_pattern_processor()
            self.pdf_extractor = create_pdf_extractor()
            self.table_extractor = create_table_extractor()
            self.spatial_analyzer = create_spatial_analyzer()
            self.document_classifier = create_document_classifier()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise PDFProcessorError(f"Failed to initialize components: {e}")
    
    def _register_ocr_engines(self) -> None:
        engines_to_register = []

        # AWS Textract
        try:
            from ocr.aws_textract import create_aws_textract_engine
            aws_engine = create_aws_textract_engine()
            engines_to_register.append(aws_engine)
        except Exception as e:
            logger.warning(f"AWS Textract not available: {e}")

        # Azure
        try:
            from ocr.azure import create_azure_ocr_engine
            azure_engine = create_azure_ocr_engine()
            engines_to_register.append(azure_engine)
        except Exception as e:
            logger.warning(f"Azure OCR not available: {e}")

        # Mistral AI
        try:
            from ocr.mistral import create_mistral_ocr_engine
            mistral_engine = create_mistral_ocr_engine()
            engines_to_register.append(mistral_engine)
        except Exception as e:
            logger.warning(f"Mistral OCR not available: {e}")

        # Tesseract
        try:
            from ocr.tesseract import create_tesseract_engine
            tesseract_engine = create_tesseract_engine()
            engines_to_register.append(tesseract_engine)
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")

        # OpenCV fallback
        try:
            from ocr.opencv import create_opencv_engine
            opencv_engine = create_opencv_engine()
            engines_to_register.append(opencv_engine)
        except Exception as e:
            logger.warning(f"OpenCV not available: {e}")

        # Register engines
        for engine in engines_to_register:
            self.ocr_registry.register(engine)

        available_engines = self.ocr_registry.get_available_engines()
        logger.info(f"Registered {len(available_engines)} OCR engines")

    
    def _setup_default_pipeline(self) -> None:
        from stages.base import PDFTextStage, OCRStage, TableStage, PatternStage, SpatialStage, ClassificationStage, FinalizationStage

        self._stages = [
            PDFTextStage(self.pdf_extractor),
            OCRStage(self.ocr_registry, self.image_processor),
            TableStage(self.table_extractor, self.image_processor),
            PatternStage(self.pattern_processor),
            SpatialStage(self.spatial_analyzer),
            ClassificationStage(self.document_classifier),
            FinalizationStage()
        ]
    
    def process(self, input_path: Path, **kwargs) -> ExtractionResult:
        """
        Process a PDF document through the complete pipeline.
        
        Args:
            input_path: Path to PDF file
            **kwargs: Additional processing options
                - use_ocr: Enable OCR processing (default: True)
                - ocr_engine: Preferred OCR engine name
                - extract_tables: Enable table extraction (default: True)
                - extract_patterns: Enable pattern extraction (default: True)
                - enhance_images: Enable image enhancement (default: True)
        
        Returns:
            ExtractionResult with all extracted data
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            UnsupportedFileTypeError: If file type is not supported
            ProcessingError: If processing fails
        """
        input_path = Path(input_path)
        self._validate_input(input_path)

        document_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"Starting processing: {input_path.name} (ID: {document_id})")
        
        try:

            result = ExtractionResult(
                document_id=document_id,
                filename=input_path.name,
                total_pages=0,
                processing_method=ProcessingMethod.HYBRID
            )

            context = {
                'input_path': input_path,
                'result': result,
                'start_time': start_time,
                'config': self.config,
                **kwargs
            }

            for stage in self._stages:
                try:
                    logger.debug(f"Executing stage: {stage.name}")
                    stage_start = time.time()

                    if not stage.validate_input(context):
                        logger.warning(f"Stage {stage.name} input validation failed")
                        continue

                    context = stage.process(context)
                    
                    stage_time = time.time() - stage_start
                    logger.debug(f"Stage {stage.name} completed in {stage_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Stage {stage.name} failed: {e}")
                    if stage.name in ['pdf_text', 'finalization']:
                        raise ProcessingError(stage.name, document_id, str(e))
                    else:
                        logger.warning(f"Continuing processing despite {stage.name} failure")
                        continue

            result = context['result']

            processing_time = time.time() - start_time
            result.processing_metrics.processing_time = processing_time

            result.confidence = self._calculate_overall_confidence(result)
            
            logger.info(
                f"Processing completed: {input_path.name} "
                f"({processing_time:.2f}s, confidence: {result.confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Processing failed for {input_path.name} after {processing_time:.2f}s: {e}")
            raise ProcessingError("pipeline", document_id, str(e))
    
    def _validate_input(self, input_path: Path) -> None:

        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        
        if not input_path.is_file():
            raise FileNotFoundError(f"Path is not a file: {input_path}")

        allowed_extensions = self.config.get('security.allowed_file_extensions', ['.pdf'])
        if input_path.suffix.lower() not in allowed_extensions:
            raise UnsupportedFileTypeError(
                str(input_path), 
                input_path.suffix, 
                allowed_extensions
            )

        max_size_mb = self.config.get('security.max_file_size_mb', 100)
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValidationError(f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)")
    
    def _calculate_overall_confidence(self, result: ExtractionResult) -> float:
        if not result.elements:
            return 0.0

        # Element confidence (40% weight)
        element_confidences = [e.confidence for e in result.elements if e.confidence > 0]
        avg_element_confidence = sum(element_confidences) / len(element_confidences) if element_confidences else 0.0
        
        # Pattern extraction success (30% weight)
        pattern_categories = len(result.structured_data)
        total_patterns = len(self.pattern_processor._compiled_patterns)
        pattern_score = min(pattern_categories / max(total_patterns * 0.2, 1), 1.0)
        
        # Table detection (20% weight)
        table_score = min(len(result.tables) / 5, 1.0)  # Up to 5 tables = full score
        
        # Text coverage (10% weight)
        text_length = len(result.extracted_text)
        coverage_score = min(text_length / 5000, 1.0)  # 5000 chars = full score

        confidence = (
            avg_element_confidence * 0.4 +
            pattern_score * 0.3 +
            table_score * 0.2 +
            coverage_score * 0.1
        )
        
        return round(confidence, 3)
    
    def process_batch(self, input_dir: Path, output_dir: Path, 
                     file_pattern: str = "*.pdf", **kwargs) -> Dict[str, Any]:
        """
        Process multiple PDF files in batch.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for output files
            file_pattern: File pattern to match (default: "*.pdf")
            **kwargs: Additional processing options
        
        Returns:
            Dict with batch processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(input_dir.glob(file_pattern))
        
        logger.info(f"Starting batch processing: {len(pdf_files)} files")
        
        results = {
            'total_files': len(pdf_files),
            'successful': 0,
            'failed': 0,
            'results': [],
            'errors': []
        }
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing batch file: {pdf_file.name}")

                result = self.process(pdf_file, **kwargs)
                self.save_result(result, "output.json")

                output_file = output_dir / f"{pdf_file.stem}_result.json"
                self.save_result(result, output_file)
                
                results['successful'] += 1
                results['results'].append({
                    'file': pdf_file.name,
                    'status': 'success',
                    'output': output_file.name,
                    'confidence': result.confidence,
                    'pages': result.total_pages,
                    'elements': len(result.elements)
                })
                
            except Exception as e:
                logger.error(f"Batch processing failed for {pdf_file.name}: {e}")
                results['failed'] += 1
                results['errors'].append({
                    'file': pdf_file.name,
                    'error': str(e)
                })
        
        logger.info(f"Batch processing completed: {results['successful']}/{len(pdf_files)} successful")
        return results
    
    def save_result(self, result: ExtractionResult, output_path: Path) -> None:
        from exporters.json_exporter import create_json_exporter
        
        exporter = create_json_exporter()
        exporter.export(result, output_path)
        logger.debug(f"Result saved to: {output_path}")
    
    def get_system_info(self) -> Dict[str, Any]:
        return {
            'processor_version': '2.0.0',
            'config': self.config.to_dict(),
            'ocr_engines': self.ocr_registry.get_engine_info(),
            'available_stages': [stage.name for stage in self._stages],
            'component_status': {
                'image_processor': bool(self.image_processor),
                'pattern_processor': bool(self.pattern_processor),
                'pdf_extractor': bool(self.pdf_extractor),
                'table_extractor': bool(self.table_extractor),
                'spatial_analyzer': bool(self.spatial_analyzer),
                'document_classifier': bool(self.document_classifier)
            }
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        validation_results = {
            'config_valid': True,
            'config_issues': [],
            'component_issues': {},
            'ocr_issues': {},
            'recommendations': []
        }
        
        try:
            self.config.validate_config()
        except Exception as e:
            validation_results['config_valid'] = False
            validation_results['config_issues'].append(str(e))
        
        ocr_issues = self.ocr_registry.validate_configuration()
        validation_results['ocr_issues'] = ocr_issues
        
        components = {
            'image_processor': self.image_processor,
            'pattern_processor': self.pattern_processor,
            'pdf_extractor': self.pdf_extractor
        }
        
        for name, component in components.items():
            if component is None:
                validation_results['component_issues'][name] = "Component not initialized"

        if ocr_issues:
            validation_results['recommendations'].append(
                "Configure at least one OCR engine for optimal results"
            )
        
        if not self.ocr_registry.get_available_engines():
            validation_results['recommendations'].append(
                "No OCR engines available - only PDF text extraction will work"
            )
        
        return validation_results
    
    def add_stage(self, stage, position: Optional[int] = None) -> None:
        if position is None:
            self._stages.append(stage)
        else:
            self._stages.insert(position, stage)
        
        logger.info(f"Added stage: {stage.name}")
    
    def remove_stage(self, stage_name: str) -> None:
        self._stages = [stage for stage in self._stages if stage.name != stage_name]
        logger.info(f"Removed stage: {stage_name}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        return {
            'total_stages': len(self._stages),
            'stages': [
                {
                    'name': stage.name,
                    'dependencies': getattr(stage, 'dependencies', []),
                    'optional': getattr(stage, 'optional', False)
                }
                for stage in self._stages
            ]
        }



def create_pdf_processor(config_path: Optional[Path] = None) -> PDFProcessor:
    return PDFProcessor(config_path)


def process_document(pdf_path: Union[str, Path], **kwargs) -> ExtractionResult:
    processor = create_pdf_processor()
    result = processor.process(Path(pdf_path))
    processor.save_result(result, Path("output.json"))
    print("Structured data found:")
    for category, items in result.structured_data.items():
        print(f"  {category}: {items}")
    return processor.process(Path(pdf_path), **kwargs)


def process_batch_documents(input_dir: Union[str, Path], 
                          output_dir: Union[str, Path], 
                          **kwargs) -> Dict[str, Any]:

    processor = create_pdf_processor()
    result = processor.process(Path(pdf_path))
    processor.save_result(result, Path("output.json"))
    print("Structured data found:")
    for category, items in result.structured_data.items():
        print(f"  {category}: {items}")

    return processor.process_batch(Path(input_dir), Path(output_dir), **kwargs)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        result = process_document(pdf_path)
        print(f"Processing completed successfully!")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Pages: {result.total_pages}")
        print(f"Elements: {len(result.elements)}")
        print(f"Tables: {len(result.tables)}")
        print(f"Structured data categories: {len(result.structured_data)}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        sys.exit(1)