from sympy.parsing.latex import parse_latex
from sympy import Eq, And
import re


class EquationProcessor:
    def __init__(self):
        pass

    def convert_latex_to_python_code(self, equations_pages):
        python_code_equation_pages = []
        for page in equations_pages:
            python_code_equation_list = []
            for latex_equation in page:
                parsed_expr = self._fix_grouping_and_formatting(latex_equation)
                python_code_equation_list.append(parsed_expr)
            python_code_equation_pages.append(python_code_equation_list)
        return python_code_equation_pages


    def _fix_grouping_and_formatting(self, equation: str) -> str:
        equation = re.sub(r'\\begin\{align\}|\\end\{align\}', '', equation)
        equation = re.sub(r'\\left\{|\\right\}', '', equation)
        
        # Step 2: Split the LaTeX string by lines or individual equations
        equations = re.split(r'\\\\|\\&', equation)  # Splits on LaTeX newline or alignment operators
        
        # Step 3: Parse each equation individually
        parsed_equations = []
        for eq_str in equations:
            eq_str = eq_str.strip()  # Remove whitespace
            if "=" in eq_str:  # Check if it's a valid equation format
                try:
                    parsed_eq = parse_latex(eq_str)
                    if isinstance(parsed_eq, Eq):  # Ensure it's a proper equation
                        parsed_equations.append(parsed_eq)
                except Exception as e:
                    print(f"Error parsing equation '{eq_str}': {e}")

        # Step 4: Combine parsed equations using And for a logical set of equations
        if parsed_equations:
            return And(*parsed_equations)
        else:
            return None