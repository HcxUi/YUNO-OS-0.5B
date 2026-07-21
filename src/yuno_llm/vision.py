"""
YUNO Vision Module
===================
Provides image understanding, optical character recognition (OCR), desktop
screenshot analysis, and PDF document text extraction for YUNO OS.

Dependencies:
    - Pillow (PIL) for image loading and desktop screenshot capture
    - pytesseract for OCR text extraction (with clean fallback if Tesseract binary is uninstalled)
    - PyMuPDF (fitz) for PDF document text & image rendering

Usage:
    from yuno_llm.vision import YunoVision
    vision = YunoVision(config)

    # Analyze local image
    result = vision.analyze_image("sample.png")
    print(result.extracted_text)

    # Capture and analyze screen
    screenshot_result = vision.analyze_screenshot()

    # Document OCR
    doc_text = vision.extract_document_ocr("paper.pdf")
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

logger = logging.getLogger("yuno_llm.vision")


# ── Vision Analysis Result Dataclass ──────────────────────────────────────────

@dataclass
class VisionAnalysisResult:
    """Structured container for vision analysis results."""
    source_path: str
    width: int
    height: int
    format: str
    mode: str
    file_size_bytes: int
    extracted_text: str = ""
    confidence: float = 0.0
    has_text: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary block for prompt context injection."""
        lines = [
            f"[Vision Analysis Result]",
            f"  Source     : {self.source_path}",
            f"  Dimensions : {self.width}x{self.height} ({self.format}, {self.mode})",
            f"  File Size  : {self.file_size_bytes / 1024:.1f} KB",
        ]
        if self.has_text:
            lines.append(f"  Extracted Text (OCR):\n{self.extracted_text.strip()}")
        else:
            lines.append("  Text Status: No readable text detected via OCR.")
        return "\n".join(lines)


# ── YunoVision Engine Class ───────────────────────────────────────────────────

class YunoVision:
    """
    Core YUNO OS Vision engine.

    Handles image file inspection, OCR text extraction, desktop screenshot
    captures, and multi-page PDF document OCR.
    """

    def __init__(self, config=None):
        self.config = config
        self.enabled: bool = True
        self.max_dim: int = 1024
        self.ocr_engine: str = "tesseract"
        self.screenshots_dir: Path = Path("workspace/screenshots")

        if config and hasattr(config, "vision"):
            v = config.vision
            self.enabled = getattr(v, "enabled", True)
            self.max_dim = getattr(v, "max_image_dim", 1024)
            self.ocr_engine = getattr(v, "ocr_engine", "tesseract")

        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._tesseract_available: Optional[bool] = None

    # ── 1. Image File Analysis ────────────────────────────────────────────────

    def analyze_image(self, image_path: str) -> VisionAnalysisResult:
        """
        Loads an image file, extracts metadata, resizes if needed, and runs OCR.

        Args:
            image_path: Path to local image file (.png, .jpg, .webp, .bmp).

        Returns:
            VisionAnalysisResult object containing metadata and OCR text.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            from PIL import Image
        except ImportError:
            return VisionAnalysisResult(
                source_path=str(path),
                width=0, height=0, format="UNKNOWN", mode="UNKNOWN",
                file_size_bytes=path.stat().st_size,
                extracted_text="[ERROR: Pillow library not installed]",
            )

        file_size = path.stat().st_size
        try:
            with Image.open(path) as img:
                w, h = img.size
                fmt = img.format or path.suffix[1:].upper()
                mode = img.mode

                # Run OCR extraction
                extracted_text = self._run_ocr(img)
                has_text = bool(extracted_text.strip())

                return VisionAnalysisResult(
                    source_path=str(path.resolve()),
                    width=w,
                    height=h,
                    format=fmt,
                    mode=mode,
                    file_size_bytes=file_size,
                    extracted_text=extracted_text,
                    has_text=has_text,
                )
        except Exception as e:
            logger.error(f"[Vision] Image analysis failed for {image_path}: {e}")
            return VisionAnalysisResult(
                source_path=str(path),
                width=0, height=0, format="ERROR", mode="ERROR",
                file_size_bytes=file_size,
                extracted_text=f"[ERROR reading image: {e}]",
            )

    # ── 2. Desktop Screenshot Capture & Analysis ───────────────────────────────

    def analyze_screenshot(self, save: bool = True) -> VisionAnalysisResult:
        """
        Captures a real-time desktop screenshot, saves it to workspace/screenshots,
        and performs full OCR text analysis.

        Returns:
            VisionAnalysisResult for the desktop screenshot.
        """
        try:
            from PIL import ImageGrab
        except ImportError:
            return VisionAnalysisResult(
                source_path="screenshot",
                width=0, height=0, format="ERROR", mode="ERROR",
                file_size_bytes=0,
                extracted_text="[ERROR: Pillow ImageGrab not supported on this OS]",
            )

        try:
            logger.info("[Vision] Capturing desktop screenshot...")
            screenshot = ImageGrab.grab()
            w, h = screenshot.size
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.screenshots_dir / f"screenshot_{timestamp}.png"

            if save:
                screenshot.save(save_path, format="PNG")
                logger.info(f"[Vision] Screenshot saved to {save_path}")

            extracted_text = self._run_ocr(screenshot)
            has_text = bool(extracted_text.strip())

            return VisionAnalysisResult(
                source_path=str(save_path.resolve()) if save else "desktop_screenshot",
                width=w,
                height=h,
                format="PNG",
                mode=screenshot.mode,
                file_size_bytes=save_path.stat().st_size if save else 0,
                extracted_text=extracted_text,
                has_text=has_text,
            )
        except Exception as e:
            logger.error(f"[Vision] Screenshot capture failed: {e}")
            return VisionAnalysisResult(
                source_path="screenshot",
                width=0, height=0, format="ERROR", mode="ERROR",
                file_size_bytes=0,
                extracted_text=f"[ERROR capturing screenshot: {e}]",
            )

    # ── 3. Document PDF OCR ───────────────────────────────────────────────────

    def extract_document_ocr(self, pdf_path: str, max_pages: int = 10) -> str:
        """
        Extracts text from a PDF document using PyMuPDF (fitz) text extraction.
        Falls back to image rendering + OCR for scanned PDF pages.

        Args:
            pdf_path: Path to local PDF file.
            max_pages: Maximum pages to process.

        Returns:
            Extracted text content from the PDF document.
        """
        path = Path(pdf_path)
        if not path.exists():
            return f"[ERROR] PDF document not found: {pdf_path}"

        try:
            import fitz  # PyMuPDF
        except ImportError:
            return f"[ERROR] PyMuPDF (fitz) library not installed."

        try:
            doc = fitz.open(str(path))
            pages_text = []
            total_pages = min(len(doc), max_pages)

            for i in range(total_pages):
                page = doc[i]
                text = page.get_text().strip()

                # If page text is empty (scanned image page), render and OCR
                if not text:
                    pix = page.get_pixmap()
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = self._run_ocr(img)

                pages_text.append(f"--- Page {i+1} ---\n{text if text else '[No text on page]'}")

            doc.close()
            return f"Document: {path.name} ({total_pages} pages processed)\n\n" + "\n\n".join(pages_text)

        except Exception as e:
            logger.error(f"[Vision] PDF document OCR error: {e}")
            return f"[ERROR processing PDF document: {e}]"

    # ── Internal OCR Runner ───────────────────────────────────────────────────

    def _run_ocr(self, pil_img) -> str:
        """
        Internal OCR execution via pytesseract with graceful fallback.
        """
        try:
            import pytesseract
            # Test if tesseract binary is in PATH
            text = pytesseract.image_to_string(pil_img)
            return text.strip()
        except Exception as e:
            logger.debug(f"[Vision] Pytesseract execution note: {e}")
            # Intelligent fallback: basic visual analysis summary if Tesseract binary is uninstalled
            w, h = pil_img.size
            return f"[Image dimensions: {w}x{h}. Install Tesseract-OCR binary for full optical character recognition.]"

    def __repr__(self) -> str:
        return f"YunoVision(enabled={self.enabled}, engine={self.ocr_engine!r})"
