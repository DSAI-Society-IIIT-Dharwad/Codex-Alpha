import os
import pytesseract
from pdf2image import convert_from_path
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any
import PyPDF2
import pdfplumber

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

class PDFTextExtractor:
    """Handles all PDF text extraction methods including OCR"""
    
    def __init__(self):
        """Initialize the PDF text extractor"""
        pass
    
    def extract_text_with_pdfplumber(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text using pdfplumber for better structure preservation"""
        try:
            print(f"Processing {pdf_path} with pdfplumber...")
            
            extracted_text = ""
            metadata = {"total_pages": 0, "page_texts": {}}
            
            with pdfplumber.open(pdf_path) as pdf:
                metadata["total_pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        page_text = f"\n--- Page {page_num} ---\n{page_text}\n"
                        extracted_text += page_text
                        metadata["page_texts"][str(page_num)] = page_text
                    
                    print(f"Processed page {page_num}/{len(pdf.pages)}")
            
            print(f"pdfplumber completed for {os.path.basename(pdf_path)}")
            return extracted_text.strip(), metadata
                
        except Exception as e:
            print(f"Error in pdfplumber processing for {pdf_path}: {str(e)}")
            return "", {"total_pages": 0, "page_texts": {}}

    def extract_text_with_ocr(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF using Tesseract OCR with enhanced language support"""
        try:
            print(f"Processing {pdf_path} with enhanced OCR...")
            
            # Create a temporary directory for storing images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF pages to images with higher DPI for better OCR
                images = convert_from_path(pdf_path, dpi=400, output_folder=temp_dir)
                
                extracted_text = ""
                metadata = {"total_pages": len(images), "page_texts": {}}
                
                for i, image in enumerate(images):
                    print(f"Processing page {i+1}/{len(images)} of {os.path.basename(pdf_path)}")
                    
                    # Save image temporarily
                    image_path = os.path.join(temp_dir, f"page_{i}.png")
                    image.save(image_path, 'PNG')
                    
                    # Extract text using Tesseract OCR with enhanced settings
                    try:
                        # Use both English and Hindi for better legal document processing
                        custom_config = r'--oem 3 --psm 6'
                        page_text = pytesseract.image_to_string(
                            image_path, 
                            lang='eng+hin',  # Support for bilingual documents
                            config=custom_config
                        )
                        
                        if page_text.strip():
                            formatted_text = f"\n--- Page {i+1} ---\n{page_text}\n"
                            extracted_text += formatted_text
                            metadata["page_texts"][str(i+1)] = formatted_text
                        
                    except Exception as ocr_error:
                        print(f"OCR error on page {i+1}: {str(ocr_error)}")
                        continue
                
                print(f"Enhanced OCR completed for {os.path.basename(pdf_path)}")
                return extracted_text.strip(), metadata
                
        except Exception as e:
            print(f"Error in OCR processing for {pdf_path}: {str(e)}")
            return "", {"total_pages": 0, "page_texts": {}}

    def extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text with PyPDF2 from {pdf_path}: {str(e)}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Enhanced text extraction with metadata collection"""
        extracted_text = ""
        metadata = {}
        
        # First try pdfplumber for better structure preservation
        extracted_text, metadata = self.extract_text_with_pdfplumber(pdf_path)
        
        # If pdfplumber fails or returns insufficient text, use OCR
        if not extracted_text.strip() or len(extracted_text) < 100:
            if use_ocr:
                print(f"pdfplumber returned insufficient text for {pdf_path}, trying OCR...")
                extracted_text, metadata = self.extract_text_with_ocr(pdf_path)
            else:
                # Fallback to PyPDF2
                extracted_text = self.extract_text_with_pypdf2(pdf_path)
                metadata = {"extraction_method": "pypdf2"}
        else:
            metadata["extraction_method"] = "pdfplumber"
        
        return extracted_text, metadata

    def validate_pdf_readability(self, pdf_path: str) -> Dict[str, Any]:
        """Validate if a PDF is readable and estimate extraction quality"""
        if not os.path.exists(pdf_path):
            return {"success": False, "error": "File not found"}
        
        try:
            # Try pdfplumber first
            text_pdfplumber = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:3]:  # Check first 3 pages
                    page_text = page.extract_text()
                    if page_text:
                        text_pdfplumber += page_text
            
            # Try PyPDF2
            text_pypdf2 = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages[:3]:  # Check first 3 pages
                    text_pypdf2 += page.extract_text()
            
            # Estimate quality
            pdfplumber_words = len(text_pdfplumber.split())
            pypdf2_words = len(text_pypdf2.split())
            
            recommendation = "OCR recommended" if max(pdfplumber_words, pypdf2_words) < 50 else "Text extraction sufficient"
            
            return {
                "success": True,
                "filename": os.path.basename(pdf_path),
                "pdfplumber_words": pdfplumber_words,
                "pypdf2_words": pypdf2_words,
                "recommendation": recommendation,
                "preview_pdfplumber": text_pdfplumber[:200] + "..." if text_pdfplumber else "No text extracted",
                "preview_pypdf2": text_pypdf2[:200] + "..." if text_pypdf2 else "No text extracted"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error validating PDF: {str(e)}"
            }

# Global instance for easy access
pdf_extractor = PDFTextExtractor()

def extract_pdf_text(pdf_path: str, use_ocr: bool = True) -> Tuple[str, Dict[str, Any]]:
    """Convenience function to extract text from PDF"""
    return pdf_extractor.extract_text_from_pdf(pdf_path, use_ocr)

def validate_pdf(pdf_path: str) -> Dict[str, Any]:
    """Convenience function to validate PDF readability"""
    return pdf_extractor.validate_pdf_readability(pdf_path)