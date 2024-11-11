from equationHandler.EquationExtractor import EquationExtractor
from equationHandler.EquationProcessor import EquationProcessor

class EquationHandler:

    _instance = None
    equation_extractor: EquationExtractor
    equation_processor: EquationProcessor

    def __init__(self, document_path: str):
        if not hasattr(self, "initialized"):
            self.equation_extractor = EquationExtractor(document_path)
            self.equation_processor = EquationProcessor()
            self.initialized = True
        else:
            return self._instance

    def extract_equations_from_latex(self):
        """Pipeline for processing equations, when we have extracted latex"""
        pass

    def extract_equations_from_image(self, python_code: bool = True):
        """Pipeline for processing equations, when we have extracted equations TODO: equation extraction is not implemented yet"""
        latex_equations = self.equation_extractor.extract_equations()
        if python_code:
            python_equations = self.equation_processor.convert_latex_to_python_code(latex_equations)
        return latex_equations, python_equations