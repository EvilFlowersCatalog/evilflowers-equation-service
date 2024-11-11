from PIL import Image
import torch
from typing import List
from transformers import NougatProcessor, VisionEncoderDecoderModel
import re

from domain.base.ExtractorInterface import ExtractorInterface

class Nougat(ExtractorInterface):

    def __init__(self, max_new_tokens: int = 2048):
        self.max_new_tokens = max_new_tokens

    def extract_latex_equation(self, image: Image.Image) -> List[str]:
        image_latex = self.convert_image_to_latex(image)
        equations = self.filter_equations(image_latex)
        return equations

    def convert_image_to_latex(self, image: Image.Image) -> str:
        processor = NougatProcessor.from_pretrained("facebook/nougat-base")
        model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        pixel_values = processor(image, return_tensors="pt").pixel_values

        try:
            outputs = model.generate(
                pixel_values.to(device),
                min_length=1,
                max_new_tokens=self.max_new_tokens,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
            )
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")

        sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        latex_code = processor.post_process_generation(sequence, fix_markdown=False)

        return latex_code

    def filter_equations(self, latex: str) -> List[str]:
        
        # formatted_string = latex.replace("\\\\", "\\")

        # Detect mathematical expressions
        # - $$ ... $$ for display mode
        # - \( ... \) for inline mode
        # - \[ ... \] for display mode (another format)
        math_pattern = r"\$\$(.*?)\$\$|\\\((.*?)\\\)|\\\[(.*?)\\\]"

        # Find all matches
        matches = re.findall(math_pattern, latex)
        
        # Extract the matched groups into a list of equations
        equations = []
        for match in matches:
            # Each match is a tuple with groups; only one group will contain content per match
            equation = next(group for group in match if group)
            equations.append(equation)


        return equations
