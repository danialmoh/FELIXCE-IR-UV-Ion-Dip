# pages/Custom_Spectrum.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import plotly.graph_objs as go
from amespahdbpythonsuite.amespahdb import AmesPAHdb

st.title("Custom Spectrum")

# --- UPLOAD OR PASTE DATA ---
st.markdown("## 1. Load your spectrum")
col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader(
        "Upload file (.csv, .xlsx, .jdx)", 
        type=["csv","xlsx","jdx"]
    )
with col2:
    raw_text = st.text_area(
        "…or paste wavenumber ⏤ intensity text (two columns)",
        height=150, placeholder="1000 0.12\n995 0.15\n…"
    )

df = None
# CSV / Excel
if uploaded is not None and uploaded.name.endswith((".csv",".xlsx")):
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
# JDX (placeholder – need format spec)
elif uploaded is not None and uploaded.name.endswith(".jdx"):
    st.info("Parsing JDX… (please send me a sample so I can implement this)")
# Pasted text
elif raw_text:
    try:
        df = pd.read_csv(
            pd.io.common.StringIO(raw_text), 
            sep=r"\s+", engine="python", 
            names=["wavenumber","intensity"], 
            header=None
        )
    except Exception as e:
        st.error(f"Could not parse pasted data: {e}")

if df is None:
    st.stop()

# --- CLEAN & NORMALIZE ---
st.subheader("2. Preprocess")
# try to find the right columns
wcol, icol = None, None
for c in df.columns:
    if "wave" in c.lower(): wcol = c
    if "intens" in c.lower(): icol = c
if wcol is None or icol is None:
    st.error("Could not auto-detect columns. Please rename to contain “wavenumber” and “intensity”.")
    st.stop()

# drop NaNs, sort by wavenumber descending (IR convention)
df = df[[wcol, icol]].dropna()
df = df.sort_values(by=wcol, ascending=False).reset_index(drop=True)
df.columns = ["wavenumber","intensity"]

# normalize
df["norm_intensity"] = df["intensity"] / df["intensity"].max()

# smoothing?
apply_smooth = st.checkbox("Apply Savitzky–Golay smoothing", value=False)
if apply_smooth:
    win = st.slider("Window size (odd)", 5, 51, 9, 2)
    poly = st.slider("Polynomial order", 1, 5, 2)
    sm = savgol_filter(df["norm_intensity"], win, poly)
    df["norm_intensity"] = sm

st.dataframe(df.head())

# store for later pages
st.session_state["custom_df"] = df

# --- PLOT CUSTOM (and overlay) ---
st.subheader("3. Plot")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["wavenumber"], y=df["norm_intensity"],
    mode="lines", name="Custom", line=dict(width=2)
))

# Optional overlay
if st.checkbox("Overlay theoretical from AmesPAHdb", value=False):
    with st.expander("Load theoretical spectrum"):
        xml_path = st.text_input("XML file path", "")
        uid_input = st.text_input("UID", "")
        conv = st.selectbox("Convolution", ["Gaussian","Lorentzian"])
        fwhm = st.number_input("FWHM", 15.0)
        if st.button("Load theory"):
            try:
                pahdb = AmesPAHdb(filename=xml_path, check=False, cache=True)
                uid = int(uid_input)
                trans = pahdb.gettransitionsbyuid([uid]).get()
                grid = trans["grid"]; data = trans["data"][uid]
                data = np.array(data)/np.max(data)
                theory = pd.DataFrame({
                    "wavenumber": grid, "norm_intensity": data
                }).sort_values("wavenumber", ascending=False)
                # shift slider
                shift = st.slider("Shift (cm⁻¹)", -50, 50, 0)
                theory["wavenumber"] += shift
                fig.add_trace(go.Scatter(
                    x=theory["wavenumber"], y=theory["norm_intensity"],
                    mode="lines", name="Theory"
                ))
            except Exception as e:
                st.error(f"Theory load failed: {e}")

fig.update_layout(
    xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Normalized intensity",
    xaxis=dict(autorange="reversed")
)
st.plotly_chart(fig, use_container_width=True)
