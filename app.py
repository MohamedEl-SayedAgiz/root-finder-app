import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

# ---------- page ----------
st.set_page_config(page_title="Root Finding Methods Calculator", layout="wide")
st.title("🔍 Root Finding Methods Calculator")
st.write("Enter function, pick a method and tolerance. Table at left, compact plot at right.")

# ---------- user inputs ----------
f_input = st.text_input("Enter f(x):", "x**3 - x - 1")

method = st.selectbox("Select Method:",
                      ["Bisection", "False Position", "Newton-Raphson", "Secant", "Fixed Point"])

x = sp.symbols('x')

# try parse f(x)
try:
    f_expr = sp.sympify(f_input)
    f = sp.lambdify(x, f_expr, "numpy")
except Exception:
    st.error("Invalid f(x). Fix expression (use python syntax, e.g. x**3, exp(x)).")
    st.stop()

# dynamic inputs per method
col1, col2 = st.columns(2)

if method in ["Bisection", "False Position"]:
    a = col1.number_input("Initial a:", value=0.0, step=0.1)
    b = col2.number_input("Initial b:", value=2.0, step=0.1)

elif method == "Newton-Raphson":
    df_input = st.text_input("Enter f'(x):", str(sp.diff(f_expr, x)))
    x0 = col1.number_input("Initial x0:", value=1.0, step=0.1)
    try:
        df_expr = sp.sympify(df_input)
        df = sp.lambdify(x, df_expr, "numpy")
    except Exception:
        st.error("Invalid f'(x).")
        st.stop()

elif method == "Secant":
    x0 = col1.number_input("x0:", value=0.0, step=0.1)
    x1 = col2.number_input("x1:", value=1.0, step=0.1)

elif method == "Fixed Point":
    g_input = st.text_input("Enter g(x) (for fixed-point):", "(x+1)**(1/3)")
    x0 = col1.number_input("Initial x0:", value=0.5, step=0.1)
    try:
        g_expr = sp.sympify(g_input)
        g = sp.lambdify(x, g_expr, "numpy")
    except Exception:
        st.error("Invalid g(x).")
        st.stop()

tol = st.number_input("Tolerance (stopping criterion):", value=1e-4, format="%.6f")

MAX_ITERS = 500

# ---------- animated button CSS ----------
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

# ---------- compute ----------
if st.button("Start Calculation 🚀"):

    iterations = []
    roots = []
    header = []

    # ---------- BISECTION ----------
    if method == "Bisection":
        if f(a) * f(b) > 0:
            st.error("f(a) and f(b) must have opposite signs for Bisection / False Position.")
            st.stop()

        prev = None
        iter_count = 0
        header = ["Iter", "a", "b", "c", "Error"]

        while True:
            c = (a + b) / 2
            err = np.nan if prev is None else abs(c - prev)

            iterations.append([iter_count, float(a), float(b), float(c), err])
            roots.append(c)

            if (prev is not None and err < tol) or iter_count >= MAX_ITERS:
                break

            fc = f(c)
            if f(a) * fc < 0:
                b = c
            else:
                a = c

            prev = c
            iter_count += 1

    # ---------- FALSE POSITION ----------
    elif method == "False Position":
        if f(a) * f(b) > 0:
            st.error("f(a) and f(b) must have opposite signs.")
            st.stop()

        prev = None
        iter_count = 0
        header = ["Iter", "a", "b", "c", "Error"]

        while True:
            fa = f(a)
            fb = f(b)
            c = (a * fb - b * fa) / (fb - fa)
            err = np.nan if prev is None else abs(c - prev)

            iterations.append([iter_count, float(a), float(b), float(c), err])
            roots.append(c)

            if (prev is not None and err < tol) or iter_count >= MAX_ITERS:
                break

            fc = f(c)
            if fa * fc < 0:
                b = c
            else:
                a = c

            prev = c
            iter_count += 1

    # ---------- NEWTON ----------
    elif method == "Newton-Raphson":
        iter_count = 0
        xn = x0
        header = ["Iter", "xn", "f(xn)", "f'(xn)", "xn+1", "Error"]

        while True:
            fx = f(xn)
            dfx = df(xn)
            if dfx == 0:
                st.error("Derivative is zero → cannot continue.")
                st.stop()

            xn1 = xn - fx / dfx
            err = abs(xn1 - xn)

            iterations.append([iter_count, float(xn), float(fx), float(dfx), float(xn1), float(err)])
            roots.append(xn1)

            if err < tol or iter_count >= MAX_ITERS:
                break

            xn = xn1
            iter_count += 1

    # ---------- SECANT ----------
    elif method == "Secant":
        iter_count = 0
        prev_x = None
        header = ["Iter", "x0", "x1", "x2", "Error"]

        while True:
            fx0 = f(x0)
            fx1 = f(x1)
            denom = fx1 - fx0
            if denom == 0:
                st.error("Division by zero in Secant method.")
                st.stop()

            x2 = x1 - fx1 * (x1 - x0) / denom
            err = np.nan if prev_x is None else abs(x2 - prev_x)

            iterations.append([iter_count, float(x0), float(x1), float(x2), err])
            roots.append(x2)

            if (prev_x is not None and err < tol) or iter_count >= MAX_ITERS:
                break

            prev_x = x2
            x0, x1 = x1, x2
            iter_count += 1

    # ---------- FIXED POINT ----------
    elif method == "Fixed Point":
        iter_count = 0
        xn = x0
        prev_x = None
        header = ["Iter", "x_old", "x_new", "Error"]

        while True:
            x_new = g(xn)
            err = np.nan if prev_x is None else abs(x_new - prev_x)

            iterations.append([iter_count, float(xn), float(x_new), err])
            roots.append(x_new)

            if (prev_x is not None and err < tol) or iter_count >= MAX_ITERS:
                break

            prev_x = x_new
            xn = x_new
            iter_count += 1

    # ---------- PRESENT RESULTS ----------
    left_col, right_col = st.columns([3, 1])

    left_col.subheader("📑 Iteration Table")
    df = pd.DataFrame(iterations)
    df.columns = header
    left_col.dataframe(df, use_container_width=True)

    if len(roots) > 0:
        final_root = roots[-1]
        left_col.markdown(f"**Final approximation:** `{final_root:.6g}`")
        left_col.markdown(f"**f(root):** `{float(f(final_root)):.6g}`")

    # ---------- PLOT ----------
    right_col.subheader("📉 Compact Plot")

    if len(roots) == 0:
        x_min, x_max = -5, 5
    else:
        rmin, rmax = np.min(roots), np.max(roots)
        span = max(1.0, (rmax - rmin) * 1.5)
        center = (rmin + rmax) / 2
        x_min, x_max = center - span, center + span

    xs = np.linspace(x_min, x_max, 400)

    try:
        ys = f(xs)

        # -------- CLEAN INVALID VALUES --------
        mask = np.isfinite(ys)
        xs = xs[mask]
        ys = ys[mask]

        if len(xs) == 0:
            right_col.warning("Plot cannot be displayed due to invalid values.")
            st.stop()

    except Exception:
        ys = np.array([float(f_expr.subs(x, xi)) for xi in xs])

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.plot(xs, ys, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)

    if len(roots) > 0:
        last_root = final_root
        y_at_root = float(f(last_root))
        ax.plot([last_root], [y_at_root], marker="o", color="red")
        ax.axvline(last_root, color="green", linestyle=":")

    ax.set_xlim(x_min, x_max)

    right_col.pyplot(fig)

    st.success("Calculation completed ✔")
