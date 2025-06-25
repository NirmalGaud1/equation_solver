# Requires Python 3.8+, install dependencies:
# pip install streamlit pix2tex Pillow sympy numpy nest-asyncio
# Ensure pix2tex model is set up (follow https://github.com/lukas-blecher/LaTeX-OCR)
import streamlit as st
from PIL import Image
from pix2tex.cli import LatexOCR
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import re
import nest_asyncio
import io
import base64

# Apply nest_asyncio to handle any async issues in pix2tex
nest_asyncio.apply()

# Streamlit app configuration
st.set_page_config(page_title="Maths OCR Solver", layout="wide")
st.title("Maths OCR Solver")
st.markdown("Upload an image of a mathematical equation to recognize, solve, and visualize it.")

# Initialize SymPy symbol
x = sp.Symbol('x')

def parse_latex_to_sympy(latex_str):
    """Convert LaTeX equation to SymPy expression."""
    # Remove LaTeX-specific commands and simplify
    latex_str = latex_str.replace(r'\frac', 'frac').replace(r'{', '(').replace(r'}', ')')
    latex_str = latex_str.replace(r'^', '**')
    
    # Handle fractions
    def replace_frac(match):
        num, denom = match.groups()
        return f"({num})/({denom})"
    latex_str = re.sub(r'frac\((.*?)\)\((.*?)\)', replace_frac, latex_str)
    
    # Split equation at '='
    if '=' in latex_str:
        left, right = latex_str.split('=')
        left = left.strip()
        right = right.strip()
        # Move all terms to one side (e.g., ax^2 + bx + c = 0)
        try:
            expr = sp.sympify(left + '- (' + right + ')')
            return expr
        except Exception as e:
            st.error(f"Error parsing equation: {e}")
            return None
    st.error("No valid equation found (missing '=')")
    return None

def solve_equation(expr):
    """Solve the equation using SymPy."""
    if expr is None:
        return None
    try:
        solutions = sp.solve(expr, x)
        return solutions
    except Exception as e:
        st.error(f"Error solving equation: {e}")
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
        return img_base64
    except Exception as e:
        st.error(f"Error plotting graph: {e}")
        return None

def process_image(uploaded_file):
    """Process uploaded image to extract, solve, and graph equation."""
    try:
        # Load image
        img = Image.open(uploaded_file)
        
        # Display uploaded image
        st.image(img, caption="Uploaded Equation Image", width=300)
        
        # Initialize pix2tex model
        model = LatexOCR()
        # Extract LaTeX from image
        latex = model(img)
        st.subheader("Recognized LaTeX")
        st.latex(latex)
        
        # Parse LaTeX to SymPy
        expr = parse_latex_to_sympy(latex)
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

# File uploader
uploaded_file = st.file_uploader("Upload an image of a mathematical equation", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    process_image(uploaded_file)
else:
    st.info("Please upload an image to get started.")
