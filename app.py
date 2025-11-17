import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------
# Streamlit Page Config
# ------------------------------------
st.set_page_config(page_title="Root Finding Methods Calculator", layout="wide")
st.title("🔍 Root Finding Methods Calculator")
st.write("Enter function, pick a method and tolerance. Table at left, compact plot at right.")

# ------------------------------------
# User Input
# ------------------------------------
f_input = st.text_input("Enter f(x):", "x**3 - x - 1")

method = st.selectbox(
    "Select Method:",
    ["Bisection", "False Position", "Newton-Raphson", "Secant", "Fixed Point"]
)

x = sp.symbols('x')

# Parse function
try:
    f_expr = sp.sympify(f_input)
    f = sp.lambdify(x, f_expr, "numpy")
except Exception:
    st.error("Invalid f(x). Please fix syntax.")
    st.stop()

col1, col2 = st.columns(2)

# Method Inputs
if method in ["Bisection", "False Position"]:
    a = col1.number_input("Initial a:", value=0.0, step=0.1)
    b = col2.number_input("Initial b:", value=2.0, step=0.1)

elif method == "Newton-Raphson":
    df_input = st.text_input("Enter f'(x):", str(sp.diff(f_expr, x)))
    x0 = col1.number_input("Initial x0:", value=1.0, step=0.1)
    try:
        df_expr = sp.sympify(df_input)
        df = sp.lambdify(x, df_expr, "numpy")
    except:
        st.error("Invalid f'(x).")
        st.stop()

elif method == "Secant":
    x0 = col1.number_input("x0:", value=0.0, step=0.1)
    x1 = col2.number_input("x1:", value=1.0, step=0.1)

elif method == "Fixed Point":
    g_input = st.text_input("Enter g(x):", "(x+1)**(1/3)")
    x0 = col1.number_input("Initial x0:", value=0.5, step=0.1)
    try:
        g_expr = sp.sympify(g_input)
        g = sp.lambdify(x, g_expr, "numpy")
    except:
        st.error("Invalid g(x).")
        st.stop()

tol = st.number_input("Tolerance:", value=1e-4, format="%.6f")

MAX_ITERS = 500

# ------------------------------------
# Error Function
# ------------------------------------
def compute_error(prev, current):
    if prev is None:
        return None
    return abs(current - prev)

# ------------------------------------
# Animated Button Style
# ------------------------------------
st.markdown("""
<style>
div.stButton > button:first-child {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    padding: 0.7rem 2rem;
    border-radius: 12px;
    border: none;
    font-size: 18px;
    font-weight: bold;
    cursor: pointer;
    transition: 0.4s;
    animation: pulse 2s infinite;
}
div.stButton > button:first-child:hover {
    transform: scale(1.07);
    box-shadow: 0 0 18px rgba(0, 153, 255, 0.7);
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05);}
    100% { transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------
# Start Calculation
# ------------------------------------
if st.button("Start Calculation 🚀"):

    iterations = []
    roots = []

    # -----------------------------------------------------
    # 1) Bisection
    # -----------------------------------------------------
    if method == "Bisection":

        # Initial root check
        if abs(f(a)) < 1e-14:
            st.success(f"Initial a is already a root: {a}")
            st.stop()
        if abs(f(b)) < 1e-14:
            st.success(f"Initial b is already a root: {b}")
            st.stop()

        if f(a) * f(b) > 0:
            st.error("f(a) and f(b) must have opposite signs.")
            st.stop()

        header = ["Iter", "a", "b", "c", "Error"]
        prev = None

        for i in range(MAX_ITERS):
            c = (a + b) / 2
            err = compute_error(prev, c)

            iterations.append([i, float(a), float(b), float(c), err])
            roots.append(c)

            if err is not None and err < tol:
                break

            fc = f(c)

            if f(a) * fc < 0:
                b = c
            else:
                a = c

            prev = c

    # -----------------------------------------------------
    # 2) False Position
    # -----------------------------------------------------
    elif method == "False Position":

        if abs(f(a)) < 1e-14:
            st.success(f"Initial a is already a root: {a}")
            st.stop()
        if abs(f(b)) < 1e-14:
            st.success(f"Initial b is already a root: {b}")
            st.stop()

        if f(a) * f(b) > 0:
            st.error("f(a) and f(b) must have opposite signs.")
            st.stop()

        header = ["Iter", "a", "b", "c", "Error"]
        prev = None

        for i in range(MAX_ITERS):
            fa, fb = f(a), f(b)
            c = (a*fb - b*fa) / (fb - fa)
            err = compute_error(prev, c)

            iterations.append([i, float(a), float(b), float(c), err])
            roots.append(c)

            if err is not None and err < tol:
                break

            fc = f(c)

            if fa * fc < 0:
                b = c
            else:
                a = c

            prev = c

    # -----------------------------------------------------
    # 3) Newton-Raphson
    # -----------------------------------------------------
    elif method == "Newton-Raphson":

        if abs(f(x0)) < 1e-14:
            st.success(f"x0 is already a root: {x0}")
            st.stop()

        header = ["Iter", "xn", "f(xn)", "f'(xn)", "xn+1", "Error"]
        xn = x0

        for i in range(MAX_ITERS):
            fx, dfx = f(xn), df(xn)

            if dfx == 0:
                st.error("Derivative is zero → cannot continue.")
                st.stop()

            xn1 = xn - fx/dfx
            err = abs(xn1 - xn)

            iterations.append([i, float(xn), float(fx), float(dfx), float(xn1), err])
            roots.append(xn1)

            if err < tol:
                break

            xn = xn1

    # -----------------------------------------------------
    # 4) Secant
    # -----------------------------------------------------
    elif method == "Secant":

        if abs(f(x0)) < 1e-14:
            st.success(f"x0 is already a root: {x0}")
            st.stop()
        if abs(f(x1)) < 1e-14:
            st.success(f"x1 is already a root: {x1}")
            st.stop()

        header = ["Iter", "x0", "x1", "x2", "Error"]
        prev = None

        for i in range(MAX_ITERS):
            fx0, fx1 = f(x0), f(x1)
            denom = fx1 - fx0

            if denom == 0:
                st.error("Division by zero in Secant method.")
                st.stop()

            x2 = x1 - fx1*(x1 - x0)/denom
            err = compute_error(prev, x2)

            iterations.append([i, float(x0), float(x1), float(x2), err])
            roots.append(x2)

            if err is not None and err < tol:
                break

            prev = x2
            x0, x1 = x1, x2

    # -----------------------------------------------------
    # 5) Fixed Point
    # -----------------------------------------------------
    elif method == "Fixed Point":

        if abs(f(x0)) < 1e-14:
            st.success(f"x0 is already a root: {x0}")
            st.stop()

        header = ["Iter", "x_old", "x_new", "Error"]
        xn = x0
        prev = None

        for i in range(MAX_ITERS):
            x_new = g(xn)
            err = compute_error(prev, x_new)

            iterations.append([i, float(xn), float(x_new), err])
            roots.append(x_new)

            if err is not None and err < tol:
                break

            prev = x_new
            xn = x_new

    # ------------------------------------
    # Present Results
    # ------------------------------------
    left_col, right_col = st.columns([3, 1])

    df = pd.DataFrame(iterations)
    df.columns = header

    # Highlight last row
    def highlight_last(row):
        if row.name == df.index[-1]:
            return ['background-color: #003366; color: white'] * len(row)
        return [''] * len(row)

    left_col.subheader("📑 Iteration Table")
    left_col.dataframe(df.style.apply(highlight_last, axis=1), use_container_width=True)

    final_root = roots[-1]
    left_col.markdown(f"**Final approximation:** `{final_root:.6g}`")
    left_col.markdown(f"**f(root):** `{float(f(final_root)):.6g}`")

    # ------------------------------------
    # PLOT
    # ------------------------------------
    right_col.subheader("📉 Compact Plot")

    rmin, rmax = min(roots), max(roots)
    span = max(1.0, (rmax - rmin) * 1.5)
    center = (rmin + rmax) / 2
    x_min, x_max = center - span, center + span

    xs = np.linspace(x_min, x_max, 400)
    ys = f(xs)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(xs, ys, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.plot([final_root], [f(final_root)], "ro")
    ax.axvline(final_root, color="green", linestyle=":")

    right_col.pyplot(fig)

    st.success("✔ Calculation completed successfully!")
