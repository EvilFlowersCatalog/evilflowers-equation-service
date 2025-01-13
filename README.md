# LaTeX Equation Extractor

A Python application that extracts equations from PDF documents and converts them to both LaTeX and Python code.


# Installation

### Install dependencies
```
make install
```
or

```
pip install -r requirements.txt 
```

# Usage

### Run with Makefile
```
make run
```

### Run directly
```
python src/main.py
```

# Project Structure

```
.
├── Makefile
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── main.py
    ├── config/
    │ └── Config.py
    ├── domain/
    │ ├── base/
    │ │ └── ExtractorInterface.py
    │ └── extractors/
    │ └── Nougat.py
    └── equationHandler/
        ├── EquationExtractor.py
        ├── EquationProcessor.py
        └── EquationService.py
```

# File Structure Explanation

## Root Directory Files

1. **Makefile**: Contains commands for installing dependencies and running the application
   - `make install`: Installs dependencies
   - `make run`: Runs the application

2. **README.md**: Basic documentation with installation and running instructions

3. **requirements.txt**: Lists all Python dependencies including:
   - PDF processing: pdf2image
   - Machine learning: torch, transformers
   - Equation processing: sympy
   - Environment management: python-dotenv

4. **.gitignore**: Standard Git ignore file for Python projects

## Source Code (`src/`)

### Main Application

- **main.py**: Entry point of the application that initializes the equation handling pipeline

### Configuration (`config/`)

- **Config.py**: Singleton configuration class that loads environment variables for:
  - LATEX_CONVERSION_MODEL
  - EQUATION_EXTRACTION_MODEL

### Domain (`domain/`)

1. **base/**
   - **ExtractorInterface.py**: Abstract base class defining the interface for equation extractors with methods:
     - extract_latex_equation()
     - filter_equations()
     - convert_image_to_latex()

2. **extractors/**
   - **Nougat.py**: Implementation of ExtractorInterface using the Nougat model for LaTeX extraction

### Equation Handler (`equationHandler/`)

1. **EquationService.py**: Main service class that orchestrates the equation extraction pipeline
   - Coordinates between extraction and processing
   - Implements singleton pattern

2. **EquationExtractor.py**: Handles PDF to image conversion and equation extraction
   - Converts PDF pages to images
   - Extracts equations from images
   - Uses Nougat model for LaTeX conversion

3. **EquationProcessor.py**: Processes extracted equations
   - Converts LaTeX equations to Python code
   - Handles equation formatting and grouping
   - Uses sympy for LaTeX parsing

# Architecture Flow

1. User provides a PDF document path
2. EquationHandler initializes the pipeline
3. EquationExtractor converts PDF to images and extracts equations
4. Nougat model converts images to LaTeX
5. EquationProcessor converts LaTeX to Python code
6. Final output includes both LaTeX and Python representations of equations