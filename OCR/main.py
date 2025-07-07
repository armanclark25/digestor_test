import os
import sys
import time
import uuid
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import numpy as np

from core.models import ExtractionResult, ProcessingMethod, DocumentType, ExtractedElement, SpatialTable, ImageInfo, ProcessingMetrics
from core.exceptions import PDFProcessorError, FileNotFoundError, UnsupportedFileTypeError, ProcessingError, ValidationError
from core.interfaces import ProcessingPipeline

from config.settings import get_config, initialize_config
from ocr_engine.base import get_ocr_registry
from processors.image import create_image_processor
from processors.pattern import create_pattern_processor
from extractors.pdf_extractor import create_pdf_extractor
from extractors.table_extractor import create_table_extractor
from processors.spatial_analyzer import create_spatial_analyzer
from processors.document_classifier import create_document_classifier
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

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

        # Try to register all available OCR engines
        ocr_engines = [
            ('ocr.aws_textract', 'create_aws_textract_engine'),
            ('ocr.azure', 'create_azure_ocr_engine'),
            ('ocr.mistral', 'create_mistral_ocr_engine'),
            ('ocr.tesseract', 'create_tesseract_engine'),
            ('ocr.opencv', 'create_opencv_engine')
        ]

        for module_name, function_name in ocr_engines:
            try:
                module = __import__(module_name, fromlist=[function_name])
                create_engine = getattr(module, function_name)
                engine = create_engine()
                engines_to_register.append(engine)
            except Exception as e:
                logger.warning(f"{module_name} not available: {e}")

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

            logger.info(f"Processing completed: {input_path.name} ({processing_time:.2f}s, confidence: {result.confidence:.2f})")
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
            raise UnsupportedFileTypeError(str(input_path), input_path.suffix, allowed_extensions)

        max_size_mb = self.config.get('security.max_file_size_mb', 100)
        file_size_mb = input_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValidationError(f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)")

    def _calculate_overall_confidence(self, result: ExtractionResult) -> float:
        if not result.elements:
            return 0.0

        element_confidences = [e.confidence for e in result.elements if e.confidence > 0]
        avg_element_confidence = sum(element_confidences) / len(element_confidences) if element_confidences else 0.0

        pattern_categories = len(result.structured_data)
        total_patterns = getattr(self.pattern_processor, '_compiled_patterns', {})
        total_patterns = len(total_patterns) if hasattr(total_patterns, '__len__') else 10
        pattern_score = min(pattern_categories / max(total_patterns * 0.2, 1), 1.0)

        table_score = min(len(result.tables) / 5, 1.0)
        text_length = len(result.extracted_text)
        coverage_score = min(text_length / 5000, 1.0)

        confidence = (avg_element_confidence * 0.4 + pattern_score * 0.3 + table_score * 0.2 + coverage_score * 0.1)
        return round(confidence, 3)

    def process_batch(self, input_dir: Path, output_dir: Path, file_pattern: str = "*.pdf", **kwargs) -> Dict[str, Any]:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = list(input_dir.glob(file_pattern))

        logger.info(f"Starting batch processing: {len(pdf_files)} files")

        results = {'total_files': len(pdf_files), 'successful': 0, 'failed': 0, 'results': [], 'errors': []}

        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing batch file: {pdf_file.name}")
                result = self.process(pdf_file, **kwargs)
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
                results['errors'].append({'file': pdf_file.name, 'error': str(e)})

        logger.info(f"Batch processing completed: {results['successful']}/{len(pdf_files)} successful")
        return results

    def save_result(self, result: ExtractionResult, output_path: Path) -> None:
        from exporters.json_exporter import create_json_exporter
        exporter = create_json_exporter()
        exporter.export(result, output_path)
        logger.debug(f"Result saved to: {output_path}")

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

    def get_system_info(self) -> Dict[str, Any]:
        return {
            'processor_version': '2.0.0',
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


class MenuInterface:
    def __init__(self):
        self.processor = None
        self.current_pdf = None

    def display_banner(self):
        print("\n" + "="*60)
        print("           CDE OCR v2.0")
        print("           Let's Begin")
        print("="*60)

    def display_main_menu(self):
        print("\nMAIN MENU")
        print("─" * 40)
        print("1. Upload/Select PDF File")
        print("2. Full Document Processing")
        print("3. OCR Text Extraction Only")
        print("4. Table Extraction Only")
        print("5. Pattern Extraction Only")
        print("6. Batch Processing")
        print("7. System Information")
        print("8. Validate Configuration")
        print("9. Exit")
        print("─" * 40)

    def get_user_choice(self, prompt="Enter your choice: ", max_choice=9):
        while True:
            try:
                user_input = input(prompt).strip()
                choice = int(user_input)
                if 1 <= choice <= max_choice:
                    return choice
                else:
                    print(f"Please enter a number between 1 and {max_choice}")
            except ValueError:
                if user_input.endswith('.pdf') and ('\\' in user_input or '/' in user_input):
                    print("💡 It looks like you entered a file path. Please select option 1 first, then enter the path.")
                else:
                    print("Please enter a valid number")

    def select_pdf_file(self):
        print("\nPDF FILE SELECTION")
        print("─" * 40)
        print("1. Enter file path manually")
        print("2. Browse current directory")
        print("3. Go back to main menu")

        choice = self.get_user_choice("Select option: ", 3)

        if choice == 1:
            file_path = input("Enter PDF file path: ").strip().strip('"\'')
            if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
                self.current_pdf = Path(file_path)
                print(f"Selected: {self.current_pdf.name}")
                return True
            else:
                print("File not found or not a PDF file")
                return False

        elif choice == 2:
            current_dir = Path.cwd()
            pdf_files = list(current_dir.glob("*.pdf"))

            if not pdf_files:
                print("No PDF files found in current directory")
                return False

            print(f"\nPDF files in {current_dir}:")
            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"{i}. {pdf_file.name}")

            if len(pdf_files) == 1:
                file_choice = 1
            else:
                file_choice = self.get_user_choice("Select file: ", len(pdf_files))

            self.current_pdf = pdf_files[file_choice - 1]
            print(f"Selected: {self.current_pdf.name}")
            return True

        return False

    def initialize_processor(self):
        if self.processor is None:
            print("Initializing PDF Processor")
            try:
                self.processor = PDFProcessor()
                print("Processor initialized successfully")
                return True
            except Exception as e:
                print(f"Failed to initialize processor: {e}")
                return False
        return True

    def check_pdf_selected(self):
        if self.current_pdf is None:
            print("No PDF file selected. Please select a file first.")
            return False
        return True

    def full_processing(self):
        if not self.check_pdf_selected() or not self.initialize_processor():
            return

        print(f"\n🔍 FULL PROCESSING: {self.current_pdf.name}")
        print("─" * 50)

        print("Processing Options:")
        use_ocr = input("Enable OCR? (Y/n): ").strip().lower() != 'n'
        extract_tables = input("Extract tables? (Y/n): ").strip().lower() != 'n'
        extract_patterns = input("Extract patterns? (Y/n): ").strip().lower() != 'n'
        enhance_images = input("Enhance images? (Y/n): ").strip().lower() != 'n'

        try:
            print("\nProcessing document")
            start_time = time.time()

            result = self.processor.process(
                self.current_pdf,
                use_ocr=use_ocr,
                extract_tables=extract_tables,
                extract_patterns=extract_patterns,
                enhance_images=enhance_images
            )

            elapsed_time = time.time() - start_time

            # Display results
            print(f"\nPROCESSING COMPLETED ({elapsed_time:.2f}s)")
            print("─" * 50)
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Pages: {result.total_pages}")
            print(f"Text Elements: {len(result.elements)}")
            print(f"Tables: {len(result.tables)}")
            print(f"Pattern Categories: {len(result.structured_data)}")

            if result.structured_data:
                print(f"\nStructured Data Found:")
                try:
                    for category, items in result.structured_data.items():
                        if hasattr(items, '__len__'):
                            print(f"  • {category}: {len(items)} items")
                        else:
                            print(f"  • {category}: {items}")
                except Exception as display_error:
                    print(f"Error displaying structured data: {display_error}")

            try:
                output_file = self.current_pdf.parent / f"{self.current_pdf.stem}_result.json"
                self.processor.save_result(result, output_file)
                print(f"Results saved to: {output_file.name}")
            except Exception as save_error:
                print(f"Warning: Could not save result file: {save_error}")
                print("Processing completed successfully (save failed)")

        except Exception as e:
            print(f"Processing failed: {e}")

    def ocr_only(self):
        if not self.check_pdf_selected() or not self.initialize_processor():
            return

        print(f"\nOCR EXTRACTION: {self.current_pdf.name}")
        print("─" * 50)

        try:
            print("Extracting text with OCR...")
            result = self.processor.process(
                self.current_pdf,
                use_ocr=True,
                extract_tables=False,
                extract_patterns=False
            )

            print(f"\nOCR COMPLETED")
            print("─" * 30)
            print(f"Text Elements: {len(result.elements)}")
            print(f"Text Length: {len(result.extracted_text)} characters")

            # Show preview
            if result.extracted_text:
                preview = result.extracted_text[:500]
                print(f"\n📄 Text Preview:")
                print(f"{preview}{'...' if len(result.extracted_text) > 500 else ''}")

        except Exception as e:
            print(f"OCR failed: {e}")

    def table_extraction_only(self):
        if not self.check_pdf_selected() or not self.initialize_processor():
            return

        print(f"\nTABLE EXTRACTION: {self.current_pdf.name}")
        print("─" * 50)

        try:
            print("Extracting tables...")
            result = self.processor.process(
                self.current_pdf,
                use_ocr=True,
                extract_tables=True,
                extract_patterns=False
            )

            print(f"\nTABLE EXTRACTION COMPLETED")
            print("─" * 35)
            print(f"Tables Found: {len(result.tables)}")

            if result.tables:
                for i, table in enumerate(result.tables, 1):
                    print(f"  Table {i}: {len(table.get('rows', []))} rows x {len(table.get('rows', [[]])[0]) if table.get('rows') else 0} columns")

        except Exception as e:
            print(f"Table extraction failed: {e}")

    def pattern_extraction_only(self):
        if not self.check_pdf_selected() or not self.initialize_processor():
            return

        print(f"\nPATTERN EXTRACTION: {self.current_pdf.name}")
        print("─" * 50)

        try:
            print("Extracting patterns")
            result = self.processor.process(
                self.current_pdf,
                use_ocr=False,
                extract_tables=False,
                extract_patterns=True
            )

            print(f"\nPATTERN EXTRACTION COMPLETED")
            print("─" * 40)
            print(f"Categories Found: {len(result.structured_data)}")

            if result.structured_data:
                for category, items in result.structured_data.items():
                    print(f"  • {category}: {items}")

        except Exception as e:
            print(f"Pattern extraction failed: {e}")

    def batch_processing(self):
        print("\nBATCH PROCESSING")
        print("─" * 40)

        input_dir = input("Enter input directory path: ").strip().strip('"\'')
        if not os.path.exists(input_dir):
            print("Input directory not found")
            return

        output_dir = input("Enter output directory path: ").strip().strip('"\'')

        if not self.initialize_processor():
            return

        try:
            print("Starting batch processing...")
            results = self.processor.process_batch(Path(input_dir), Path(output_dir))

            print(f"\nBATCH PROCESSING COMPLETED")
            print("─" * 40)
            print(f"Total Files: {results['total_files']}")
            print(f"Successful: {results['successful']}")
            print(f"Failed: {results['failed']}")

            if results['errors']:
                print(f"\nErrors:")
                for error in results['errors']:
                    print(f"  • {error['file']}: {error['error']}")

        except Exception as e:
            print(f"Batch processing failed: {e}")

    def show_system_info(self):
        if not self.initialize_processor():
            return

        print("\nSYSTEM INFORMATION")
        print("─" * 50)

        info = self.processor.get_system_info()

        print(f"Processor Version: {info['processor_version']}")
        print(f"\nOCR Engines:")
        for name, engine_info in info['ocr_engines'].items():
            status = "Available" if engine_info['available'] else "Unavailable"
            print(f"  • {name}: {status} (Priority: {engine_info['priority']})")

        print(f"\nProcessing Stages:")
        for stage in info['available_stages']:
            print(f"  • {stage}")

        print(f"\nComponents:")
        for component, status in info['component_status'].items():
            status_icon = "Yes" if status else "No"
            print(f"  • {component}: {status_icon}")

    def validate_configuration(self):
        if not self.initialize_processor():
            return

        print("\nCONFIGURATION VALIDATION")
        print("─" * 50)

        try:
            validation = self.processor.validate_configuration()

            if validation['config_valid']:
                print("Configuration is valid")
            else:
                print("Configuration issues found:")
                for issue in validation['config_issues']:
                    print(f"  • {issue}")

            if validation['ocr_issues']:
                print(f"\nOCR Engine Issues:")
                for engine, issues in validation['ocr_issues'].items():
                    print(f"  • {engine}:")
                    for issue in issues:
                        print(f"    - {issue}")

            if validation['recommendations']:
                print(f"\n💡 Recommendations:")
                for rec in validation['recommendations']:
                    print(f"  • {rec}")

        except Exception as e:
            print(f"Validation failed: {e}")

    def run(self):
        self.display_banner()

        while True:
            self.display_main_menu()

            if self.current_pdf:
                print(f"Current PDF: {self.current_pdf.name}")
            else:
                print("No PDF selected")

            choice = self.get_user_choice()

            if choice == 1:
                self.select_pdf_file()
            elif choice == 2:
                self.full_processing()
            elif choice == 3:
                self.ocr_only()
            elif choice == 4:
                self.table_extraction_only()
            elif choice == 5:
                self.pattern_extraction_only()
            elif choice == 6:
                self.batch_processing()
            elif choice == 7:
                self.show_system_info()
            elif choice == 8:
                self.validate_configuration()
            elif choice == 9:
                print("\nThank you for your attention to this matter!")
                sys.exit(0)

            input("\nPress Enter to continue")


if __name__ == "__main__":
    interface = MenuInterface()
    interface.run()