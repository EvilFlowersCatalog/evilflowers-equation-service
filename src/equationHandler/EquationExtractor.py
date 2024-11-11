import os
from pdf2image import convert_from_path
from PIL import Image

from domain.extractors.Nougat import Nougat
from domain.base.ExtractorInterface import ExtractorInterface
from config.Config import Config

class EquationExtractor:
    
    config: Config = Config()
    _latex_extractor: ExtractorInterface

    def __init__(self, document_path: str):
        self._validate_document_path(document_path)
        self.document_path = document_path
        self._latex_extractor = self._load_latex_conversion_model()

    def extract_equations_images_from_page(self, page: Image.Image):
        """TODO: Implement equation extraction"""
        return page

    def extract_equations(self):
        equations_list = []
        
        pages = self.extract_pages()
        for i, page in enumerate(pages[1:2]):  # Remove after testing
            equations = self.extract_equations_images_from_page(page)
            latex_equations = self._latex_extractor.extract_latex_equation(equations)
            equations_list.append(latex_equations)

        return equations_list

    ##
    # Private functions
    def extract_pages(self):
        return convert_from_path(self.document_path, dpi=300)

    def _validate_document_path(self, document_path: str):
        assert os.path.exists(
            document_path
        ), f"Document path did not found: {document_path}"

    def _load_latex_conversion_model(self):
        if self.config.get_config()['LATEX_CONVERSION_MODEL'] == 'Nougat':
            latex_extractor = Nougat()
        else:
            raise ValueError(f"Model type {self.config.get_config()['LATEX_CONVERSION_MODEL']} not supported")
        return latex_extractor
    
    def _load_equation_extraction_model(self):
        if self.config.get_config()['EQUATION_EXTRACTION_MODEL'] == 'Nougat':
            # TODO: Implement equation extraction model
            model = None
            pass
        else:
            raise ValueError(f"Model type {self.config.get_config()['EQUATION_EXTRACTION_MODEL']} not supported")
        return model
