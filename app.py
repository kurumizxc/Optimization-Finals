import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

"""Members"
       - John Paul Sapasap
       - Kyle Eron Hallares
       - Gem Win Canete
       - Lord Patrick Raizen Togonon"""


st.set_page_config(
    page_title="Newton-Raphson Optimizer",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Newton-Raphson Method for Optimization")
st.markdown("Finds local **minima** or **maxima** where $f'(x) = 0$.")

# Sidebar
st.sidebar.header("Input Parameters")
st.sidebar.info("💡 **Syntax Help:**\n- Power: `x**2` or `x^2`\n- Trig: `sin(x)`, `cos(x)`\n- Euler's #: `exp(x)`")

func_input = st.sidebar.text_input("Enter Function f(x):", value="sin(x) - x**2/2")
x0 = st.sidebar.number_input("Initial Guess ($x_0$):", value=1.0, step=0.1)
tol = st.sidebar.number_input("Tolerance:", value=0.0001, format="%.6f")
max_iter = st.sidebar.number_input("Max Iterations:", value=20, step=1, min_value=1)

#Main Logic 
if func_input:
    try:
        # Symbolic Math
        x = sp.symbols('x')
        f_expr = sp.sympify(func_input.replace('^', '**'))
        f_prime_expr = sp.diff(f_expr, x)
        f_double_prime_expr = sp.diff(f_prime_expr, x)

        # Show formulas in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Calculated Derivatives:**")
        st.sidebar.latex(f"f'(x) = {sp.latex(f_prime_expr)}")
        st.sidebar.latex(f"f''(x) = {sp.latex(f_double_prime_expr)}")
        
        # Lambda functions
        f = sp.lambdify(x, f_expr, 'numpy')
        f_prime = sp.lambdify(x, f_prime_expr, 'numpy')
        f_double_prime = sp.lambdify(x, f_double_prime_expr, 'numpy')

        #Iteration Loop
        data = []
        x_curr = float(x0)
        tol_header = f"{tol}" 
        
        for i in range(int(max_iter)):
            fpx = float(f_prime(x_curr))
            fppx = float(f_double_prime(x_curr))
            
            # Safety check
            if abs(fppx) < 1e-15:
                st.error("Error: Second derivative is zero. Stopping.")
                break

            x_next = x_curr - (fpx / fppx)
            error = abs(x_next - x_curr)
            is_converged = error < tol
            
            # Logic for TRUE/FALSE text
            converged_text = "TRUE" if is_converged else "FALSE"
            data.append({
                "x": x_curr,
                "f'(x)": fpx,
                "f''(x)": fppx,
                "x_i+1": x_next,
                "|x_i+1 - x|": error,
                tol_header: converged_text 
            })
            
            x_curr = x_next
            
            if is_converged:
                break 

        #Results Table
        st.header("Results")
        
        if data:
            df = pd.DataFrame(data)
            
            # Function to highlight rows where the tolerance column is "TRUE"
            def highlight_true(row):
                if row[tol_header] == "TRUE":
                    return ['background-color: yellow; color: black'] * len(row)
                return [''] * len(row)
            st.dataframe(
                df.style
                .apply(highlight_true, axis=1)
                .format({
                    "x": "{:.9f}",
                    "f'(x)": "{:.9f}",
                    "f''(x)": "{:.9f}",
                    "x_i+1": "{:.9f}",
                    "|x_i+1 - x|": "{:.9f}"
                }), 
                use_container_width=True
            )

        # Graph ---
        st.markdown("---")
        st.header("Graphical Output")
        
        # Plot setup
        all_x = [d['x'] for d in data] + [x_curr]
        x_min, x_max = min(all_x) - 1, max(all_x) + 1
        x_vals = np.linspace(x_min, x_max, 400)
        y_vals = f(x_vals)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x_vals, y_vals, label="$f(x)$", color='blue')
        
        # Plot steps
        iter_x = [d['x'] for d in data]
        iter_y = [f(ix) for ix in iter_x]
        ax.plot(iter_x, iter_y, 'ro--', label='Path', alpha=0.6)
        
        # Markers
        ax.scatter(iter_x[0], iter_y[0], color='green', s=100, label='Start')
        if data[-1][tol_header] == "TRUE":
            ax.scatter(iter_x[-1], iter_y[-1], color='yellow', edgecolor='black', s=150, marker='*', label='Converged')

        ax.set_title("Optimization Path")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please enter a function.")