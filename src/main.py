from equationHandler.EquationService import EquationHandler


def main():
    document_path = "test_data/doc1.pdf"
    equation_handler = EquationHandler(document_path)
    return equation_handler.extract_equations_from_image()

if __name__ == "__main__":
    latex_equations, python_equations = main()
    print("latex_equations", latex_equations)
    print("python_equations", python_equations)
