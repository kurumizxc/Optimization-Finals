import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
import io

st.set_page_config(
    page_title="Newton-Raphson Optimizer",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Newton-Raphson Method for Optimization")
st.markdown("""
This tool finds local **minima** or **maxima** where $f'(x) = 0$.  
It supports polynomials (e.g., `x^3 - 2*x`), trigonometry (e.g., `sin(x)`), and exponentials (e.g., `exp(x)`).
""")

#Sidebar
st.sidebar.header("Input Parameters")


with st.sidebar.expander("ℹ️ Syntax Guide"):
    st.write("""
    - **Variable:** Use `x` only.
    - **Multiply:** `2*x` (not `2x`).
    - **Power:** `x**2` or `x^2`.
    - **Functions:** `sin(x)`, `cos(x)`, `tan(x)`, `exp(x)`, `log(x)`, `sqrt(x)`.
    """)

func_input = st.sidebar.text_input("Enter Function f(x):", value="sin(x) - x^2/2")
x0 = st.sidebar.number_input("Initial Guess ($x_0$):", value=1.0, step=0.1)
tol = st.sidebar.number_input("Tolerance:", value=0.0001, format="%.6f")
max_iter = st.sidebar.number_input("Max Iterations:", value=20, step=1, min_value=1)

#Main Logi

if func_input:
    try:
        x = sp.symbols('x')
        cleaned_input = func_input.replace('^', '**')
        try:
            f_expr = sp.sympify(cleaned_input)
        except sp.SympifyError:
            st.error("**Syntax Error:** Could not understand the function. Please check for unbalanced parentheses or invalid characters.")
            st.stop()

        #Check for invalid variables
        if not f_expr.free_symbols.issubset({x}):
            st.error("**Variable Error:** Please use only 'x' as the variable.")
            st.stop()
            
        # Calculate Derivatives
        f_prime_expr = sp.diff(f_expr, x)
        f_double_prime_expr = sp.diff(f_prime_expr, x)

        # Show formulas in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Calculated Derivatives:**")
        st.sidebar.latex(f"f'(x) = {sp.latex(f_prime_expr)}")
        st.sidebar.latex(f"f''(x) = {sp.latex(f_double_prime_expr)}")
        
        # Create lambda functions for calculation
        try:
            f = sp.lambdify(x, f_expr, 'numpy')
            f_prime = sp.lambdify(x, f_prime_expr, 'numpy')
            f_double_prime = sp.lambdify(x, f_double_prime_expr, 'numpy')
        except Exception as e:
             st.error(f"**Conversion Error:** Could not convert function to math logic. Details: {e}")
             st.stop()

        #Iteration Loop 
        data = []
        x_curr = float(x0)
        tol_header = f"{tol}" 
        converged_index = None
        
        for i in range(int(max_iter)):
            try:
                # Calculate derivatives
                fpx = float(f_prime(x_curr))
                fppx = float(f_double_prime(x_curr))
                
                # Check for Zero Division 
                if abs(fppx) < 1e-15:
                    st.error(f"**Math Error:** Second derivative is zero at x={x_curr:.4f}. Division by zero prevented.")
                    break

                # Newton Step
                x_next = x_curr - (fpx / fppx)
                error = abs(x_next - x_curr)
                is_converged = error < tol
                
                converged_text = "TRUE" if is_converged else "FALSE"
                
                data.append({
                    "x": x_curr,
                    "f'(x)": fpx,
                    "f''(x)": fppx,
                    "x_i+1": x_next,
                    "|x_i+1 - x|": error,
                    tol_header: converged_text 
                })
                
                # Track convergence
                if is_converged:
                    converged_index = i
                    break 
                
                x_curr = x_next
                
            except Exception as e:
                st.error(f"**Calculation Error:** An error occurred at iteration {i+1}: {e}")
                break

        # Display Result
        st.header("Results")
        
        if data:
            df = pd.DataFrame(data)
            
            # Highlight function
            def highlight_true(row):
                if row[tol_header] == "TRUE":
                    return ['background-color: yellow; color: black'] * len(row)
                return [''] * len(row)

            # Display Table
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
            
            #Download Button 
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Iterations')
                
            st.download_button(
                label="📥 Download Results as Excel",
                data=buffer.getvalue(),
                file_name="newton_raphson_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Convergence Status
            if converged_index is not None:
                st.success(f"Converged at x ≈ {data[-1]['x_i+1']:.6f} after {len(data)} iterations.")
            else:
                st.warning("⚠️ Max iterations reached without meeting tolerance.")

        # Graphical Output
        st.markdown("---")
        st.header("Graphical Output")
        
        if data:
            # Dynamic Plot Range
            all_x = [d['x'] for d in data] + [x_curr]
            x_min, x_max = min(all_x) - 1, max(all_x) + 1
            x_vals = np.linspace(x_min, x_max, 400)
            
            try:
                y_vals = f(x_vals)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(x_vals, y_vals, label=f"$f(x)$", color='blue')
                
                # Plot Iterations
                iter_x = [d['x'] for d in data]
                iter_y = [f(ix) for ix in iter_x]
                
                ax.plot(iter_x, iter_y, 'ro--', label='Iteration Path', alpha=0.6)
                ax.scatter(iter_x[0], iter_y[0], color='green', s=100, zorder=5, label='Start')
                
                # Mark End Point
                if data[-1][tol_header] == "TRUE":
                    final_x = data[-1]['x_i+1']
                    ax.scatter(final_x, f(final_x), color='yellow', edgecolor='black', s=150, marker='*', zorder=6, label='Optimal')

                ax.set_title("Optimization Path")
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.legend()
                
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not render graph: {e}")

    except Exception as global_e:
        st.error(f"An unexpected error occurred: {global_e}")

else:
    st.info("Please enter a function in the sidebar to begin.")