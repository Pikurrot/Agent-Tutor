from __future__ import annotations
from docling.document_converter import DocumentConverter


class Processor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.doc_converter = DocumentConverter()

    def process_pdf(
        self,
        file_path: str,
        output_path: str
    ):
        doc = self.doc_converter.convert(file_path)
        
