from PIL import Image
from typing import List
from abc import ABC, abstractmethod

class ExtractorInterface(ABC):

    @abstractmethod
    def extract_latex_equation(self, image: Image.Image) -> List[str]:
        pass

    @abstractmethod
    def filter_equations(self, latex: str) -> List[str]:
        pass

    @abstractmethod
    def convert_image_to_latex(self, image: Image.Image) -> str:
        pass
