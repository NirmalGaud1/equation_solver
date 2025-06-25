# Requires Python 3.8+, install dependencies via requirements.txt
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import re
import io
import base64
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Streamlit app configuration
st.set_page_config(page_title="Maths OCR Solver", layout="wide")
st.title("Maths OCR Solver")
st.markdown("Upload an image of a mathematical equation (e.g., 2x + 3 = 5 or x^2 - 4 = 0) to recognize, solve, and visualize it.")

# Initialize SymPy symbol
x = sp.Symbol('x')

def preprocess_image(image):
    """Preprocess image for better OCR results."""
    try:
        # Convert PIL image to OpenCV format
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to enhance contrast
        _, img_binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Resize for better OCR accuracy
        img_resized = cv2.resize(img_binary, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        
        return img_resized
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        logging.error(f"Error preprocessing image: {e}")
        return None

def parse_text_to_sympy(text):
    """Convert extracted text to SymPy expression."""
    try:
        logging.debug(f"Extracted text: {text}")
        # Clean text: remove newlines, extra spaces
        text = text.strip().replace('\n', ' ').replace('\r', '')
        
        # Replace common OCR mistakes (e.g., 'O' with '0')
        text = re.sub(r'\bO\b', '0', text)
        text = re.sub(r'\^', '**', text)  # Convert ^ to **
        text = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', text)  # Add * for implicit multiplication (e.g., 2x -> 2*x)
        
        # Split equation at '='
        if '=' not in text:
            st.error("No valid equation found (missing '=')")
            logging.error("No valid equation found (missing '=')")
            return None
        
        left, right = text.split('=')
        left = left.strip()
        right = right.strip()
        
        # Parse to SymPy expression
        expr = sp.sympify(left + '- (' + right + ')')
        logging.debug(f"Parsed expression: {expr}")
        return expr
    except Exception as e:
        st.error(f"Error parsing equation: {e}")
        logging.error(f"Error parsing equation: {e}")
        return None

def solve_equation(expr):
    """Solve the equation using SymPy."""
    if expr is None:
        return None
    try:
        solutions = sp.solve(expr, x)
        logging.debug(f"Solutions: {solutions}")
        return solutions
    except Exception as e:
        st.error(f"Error solving equation: {e}")
        logging.error(f"Error solving equation: {e}")
        return None

def plot_graph(expr, solutions):
    """Plot the graph of the equation and return as base64 string."""
    try:
        # Convert expression to a function for plotting
        f = sp.lambdify(x, expr, modules=['numpy'])
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_vals, y_vals, label=str(expr))
        if solutions:
            for sol in solutions:
                if sol.is_real:
                    ax.scatter([float(sol)], [0], color='red', label=f'x = {sol}')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True)
        ax.legend()
        ax.set_title(f"Graph of {expr} = 0")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Save plot to buffer and encode to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        logging.debug("Graph generated successfully")
        return img_base64
    except Exception as e:
        st.error(f"Error plotting graph: {e}")
        logging.error(f"Error plotting graph: {e}")
        return None

def process_image(uploaded_file):
    """Process uploaded image to extract, solve, and graph equation."""
    try:
        # Load image
        img = Image.open(uploaded_file)
        logging.debug("Image loaded successfully")
        
        # Display uploaded image
        st.image(img, caption="Uploaded Equation Image", width=300)
        
        # Preprocess image
        img_processed = preprocess_image(img)
        if img_processed is None:
            return
        
        # Extract text using Tesseract
        text = pytesseract.image_to_string(img_processed, config='--psm 6')
        st.subheader("Recognized Text")
        st.write(text)
        logging.debug(f"Recognized text: {text}")
        
        # Parse text to SymPy
        expr = parse_text_to_sympy(text)
        if expr is None:
            return
        
        st.subheader("Parsed Equation")
        st.latex(f"{expr} = 0")
        
        # Solve equation
        solutions = solve_equation(expr)
        st.subheader("Solutions")
        if solutions:
            st.write(f"Solutions: {solutions}")
            for i, sol in enumerate(solutions, 1):
                st.latex(f"x_{i} = {sol}")
        else:
            st.write("No solutions found or error in solving.")
        
        # Plot graph
        st.subheader("Graph")
        img_base64 = plot_graph(expr, solutions)
        if img_base64:
            st.image(f"data:image/png;base64,{img_base64}", caption="Graph of the Equation")
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
        logging.error(f"Error processing image: {e}")

# File uploader
uploaded_file = st.file_uploader("Upload an image of a mathematical equation", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    process_image(uploaded_file)
else:
    st.info("Please upload an image to get started.")
