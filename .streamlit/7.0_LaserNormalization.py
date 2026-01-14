# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# from io import BytesIO

# st.title("Ion Yield Normalization (Full Depletion Data)")

# # ------------------------------------------------------------------
# # 1. Upload CSV Files
# # ------------------------------------------------------------------
# st.header("1. Upload CSV Files")

# # Full depletion data (ion signals)
# ion_file = st.file_uploader(
#     "Upload the 'full depletion data' CSV (columns: wavenumber, sum_withoutIR, sum_withIR, depletion, -ln(depletion))",
#     type=["csv"]
# )

# # FELIX Power Scan
# power_file = st.file_uploader(
#     "Upload the FELIX Power Scan CSV (columns: undulator wavelength (µm), mean power (mJ), ...)",
#     type=["csv"]
# )

# if ion_file is not None:
#     st.subheader("Specify Unit for Full Depletion Data 'wavenumber'")
#     unit = st.radio(
#         "Is the wavenumber in cm⁻¹ or do you already have it in µm?",
#         options=["cm⁻¹", "µm"],
#         index=0
#     )
# else:
#     unit = "cm⁻¹"  # default if no file

# # ------------------------------------------------------------------
# # 2. Process Data Only if Both Files Are Uploaded
# # ------------------------------------------------------------------
# # 2. Process Data Only if Both Files Are Uploaded
# # ------------------------------------------------------------------
# if ion_file is not None and power_file is not None:
#     # Read the CSV files
#     try:
#         ion_df = pd.read_csv(ion_file, encoding="latin-1")
#     except Exception as e:
#         st.error(f"Error reading Full Depletion CSV: {e}")
#         st.stop()

#     try:
#         # Use the uploaded file object directly (power_file), 
#         # not "power_file.csv"
#         power_df = pd.read_csv(power_file, sep=";", encoding="latin-1")
#     except Exception as e:
#         st.error(f"Error reading FELIX Power Scan CSV: {e}")
#         st.stop()

#     st.subheader("Raw Full Depletion Data")
#     st.dataframe(ion_df.head())

#     st.subheader("Raw FELIX Power Scan Data")
#     st.dataframe(power_df.head())

#     # ------------------------------------------------------------------
#     # 3. Data Preparation & Renaming
#     # ------------------------------------------------------------------
#     # For the FELIX Power Scan, your columns are:
#     # "undulator wavelength (µm)", "mean power (mJ)", "standard deviation", "spectrum analyzer mean wavelength (µm)"
#     # We rename them to "undulator_wavelength" and "mean_power" for internal consistency:
#     rename_cols = {
#         "undulator wavelength (µm)": "undulator_wavelength",
#         "mean power (mJ)": "mean_power"
#     }
#     power_df.rename(columns=rename_cols, inplace=True)

#     # Now check if the rename was successful
#     if "undulator_wavelength" not in power_df.columns or "mean_power" not in power_df.columns:
#         st.error("FELIX Power Scan CSV must have 'undulator wavelength (µm)' and 'mean power (mJ)' columns.")
#         st.stop()

#     # The user’s Full Depletion CSV columns: wavenumber, sum_withoutIR, sum_withIR, depletion, -ln(depletion)
#     if not all(col in ion_df.columns for col in ["wavenumber", "sum_withoutIR", "sum_withIR"]):
#         st.error("The full depletion CSV must have columns: wavenumber, sum_withoutIR, sum_withIR.")
#         st.stop()

#     # Convert wavenumber to wavelength (µm) if user says the data is in cm⁻¹
#     if unit == "cm⁻¹":
#         ion_df["wavelength"] = 1e4 / ion_df["wavenumber"]
#     else:
#         # If it's already in µm, rename for clarity
#         ion_df.rename(columns={"wavenumber": "wavelength"}, inplace=True)

#     # Sort the power scan by undulator_wavelength for interpolation
#     power_df.sort_values("undulator_wavelength", inplace=True)

#     # ------------------------------------------------------------------
#     # 4. Interpolate mean_power
#     # ------------------------------------------------------------------
#     ion_df["mean_power"] = np.interp(
#         ion_df["wavelength"],
#         power_df["undulator_wavelength"],
#         power_df["mean_power"]
#     )

#     # Warn if out-of-range
#     if (ion_df["wavelength"].min() < power_df["undulator_wavelength"].min() or
#         ion_df["wavelength"].max() > power_df["undulator_wavelength"].max()):
#         st.warning("Some wavelengths are outside the FELIX Power Scan range. Interpolation may be inaccurate.")

#     # ------------------------------------------------------------------
#     # 5. Compute Normalized Ion Yield
#     # ------------------------------------------------------------------
#     # Corrected formula: Y_IR(ω) = (1 / E_IR(ω)) * -ln( (sum_withIR) / (sum_withoutIR) )
#     valid_mask = (
#         (ion_df["sum_withIR"] > 0) &
#         (ion_df["sum_withoutIR"] > 0) &
#         (ion_df["mean_power"] > 0)
#     )

#     if not valid_mask.all():
#         st.warning("Some rows have non-positive values for sum_withIR, sum_withoutIR, or mean_power. Those rows → NaN.")

#     ion_df["ion_yield"] = np.where(
#         valid_mask,
#         -np.log(ion_df["sum_withIR"] / ion_df["sum_withoutIR"]) / ion_df["mean_power"],
#         np.nan
#     )

#     # ------------------------------------------------------------------
#     # 6. Assemble Processed DataFrame
#     # ------------------------------------------------------------------
#     processed_df = pd.DataFrame()

#     if unit == "cm⁻¹":
#         processed_df["wavenumber"] = ion_df["wavenumber"]
#     else:
#         processed_df["wavenumber"] = 1e4 / ion_df["wavelength"]

#     processed_df["sum_withoutIR"] = ion_df["sum_withoutIR"]
#     processed_df["sum_withIR"]    = ion_df["sum_withIR"]
#     processed_df["mean_power"]    = ion_df["mean_power"]
#     processed_df["ion_yield"]     = ion_df["ion_yield"]

#     # Keep the original depletion columns if you want them
#     if "depletion" in ion_df.columns:
#         processed_df["depletion"] = ion_df["depletion"]
#     if "-ln(depletion)" in ion_df.columns:
#         processed_df["-ln(depletion)"] = ion_df["-ln(depletion)"]

#     st.subheader("Processed Data")
#     st.dataframe(processed_df.head())

#     # Download button for processed CSV
#     st.download_button(
#         label="Download Processed Data as CSV",
#         data=processed_df.to_csv(index=False),
#         file_name="processed_data.csv",
#         mime="text/csv"
#     )

#     # ------------------------------------------------------------------
#     # 7. Plotting
#     # ------------------------------------------------------------------
#     st.subheader("Static Plot: Ion Yield vs. Wavenumber")
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.plot(processed_df["wavenumber"], processed_df["ion_yield"],
#             marker="o", linestyle="-", label="Ion Yield")
#     ax.set_xlabel("Wavenumber (cm⁻¹)")
#     ax.set_ylabel("Ion Yield")
#     ax.set_title("Ion Yield vs. Wavenumber")
#     ax.grid(True)
#     ax.legend()
#     st.pyplot(fig)

#     buf = BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)
#     st.download_button(
#         label="Download Static Plot as PNG",
#         data=buf,
#         file_name="static_plot.png",
#         mime="image/png"
#     )

#     st.subheader("Interactive Plot: Ion Yield vs. Wavenumber")
#     fig_plotly = px.line(
#         processed_df,
#         x="wavenumber",
#         y="ion_yield",
#         title="Ion Yield vs. Wavenumber",
#         markers=True,
#         labels={"ion_yield": "Ion Yield", "wavenumber": "Wavenumber (cm⁻¹)"}
#     )
#     st.plotly_chart(fig_plotly)

#     html_str = fig_plotly.to_html()
#     st.download_button(
#         label="Download Interactive Plot as HTML",
#         data=html_str,
#         file_name="interactive_plot.html",
#         mime="text/html"
#     )
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO

# ------------------------------------------------------------------
# Caching for File Reading
# ------------------------------------------------------------------
@st.cache_data
def read_csv_cached(file, sep=",", encoding="latin-1"):
    """
    Read a CSV file with caching.

    Args:
        file (UploadedFile): File uploaded via Streamlit.
        sep (str): CSV separator.
        encoding (str): File encoding.

    Returns:
        pd.DataFrame or None: DataFrame if successful; otherwise, None.
    """
    try:
        df = pd.read_csv(file, sep=sep, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

# ------------------------------------------------------------------
# File Upload & Validation Functions
# ------------------------------------------------------------------
def validate_ion_data(df):
    """
    Validate that the ion data file contains the required columns.

    Args:
        df (pd.DataFrame): Ion data DataFrame.

    Returns:
        bool: True if valid, False otherwise.
    """
    required_columns = ["wavenumber", "sum_withoutIR", "sum_withIR"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.error(f"The ion data file is missing required columns: {', '.join(missing)}. "
                 "Ensure your CSV has 'wavenumber', 'sum_withoutIR', and 'sum_withIR'.")
        return False
    return True

def validate_and_process_power_data(df):
    """
    Validate and process the FELIX Power Scan data.
    Renames required columns for consistency and sorts by 'undulator_wavelength'.

    Expected original columns:
        - "undulator wavelength (µm)"
        - "mean power (mJ)"

    Args:
        df (pd.DataFrame): Power scan DataFrame.

    Returns:
        pd.DataFrame or None: Processed DataFrame if successful; otherwise, None.
    """
    rename_cols = {
        "undulator wavelength (µm)": "undulator_wavelength",
        "mean power (mJ)": "mean_power"
    }
    df.rename(columns=rename_cols, inplace=True)
    if "undulator_wavelength" not in df.columns or "mean_power" not in df.columns:
        st.error("FELIX Power Scan CSV must contain 'undulator wavelength (µm)' and 'mean power (mJ)' columns.")
        return None
    df.sort_values("undulator_wavelength", inplace=True)
    return df

# ------------------------------------------------------------------
# Data Processing & Unit Conversion
# ------------------------------------------------------------------
@st.cache_data
def process_data(ion_df, power_df, unit):
    """
    Process the ion and power scan data:
      - Converts ion data from wavenumber to wavelength if needed.
      - Interpolates the mean power from the power scan onto the ion data.

    Args:
        ion_df (pd.DataFrame): Ion data.
        power_df (pd.DataFrame): Processed power scan data.
        unit (str): "cm⁻¹" if ion data is in wavenumber, "µm" if already in wavelength.

    Returns:
        pd.DataFrame: Ion data with an interpolated 'mean_power' column.
    """
    ion_df = ion_df.copy()
    power_df = power_df.copy()

    # Convert ion data: if unit is cm⁻¹, convert to wavelength in µm.
    if unit == "cm⁻¹":
        ion_df["wavelength"] = 1e4 / ion_df["wavenumber"]
    else:
        # If already in µm, rename column for clarity.
        ion_df.rename(columns={"wavenumber": "wavelength"}, inplace=True)

    # Interpolate mean power based on wavelength.
    ion_df["mean_power"] = np.interp(
        ion_df["wavelength"],
        power_df["undulator_wavelength"],
        power_df["mean_power"]
    )

    # Warn if some wavelengths fall outside the power scan range.
    if (ion_df["wavelength"].min() < power_df["undulator_wavelength"].min() or
        ion_df["wavelength"].max() > power_df["undulator_wavelength"].max()):
        st.warning("Some wavelengths are outside the FELIX Power Scan range. Interpolation may be inaccurate.")

    return ion_df

# ------------------------------------------------------------------
# Calculation of Normalized Ion Yield
# ------------------------------------------------------------------
def calculate_ion_yield(ion_df):
    """
    Calculate the normalized ion yield using the formula:
      ion_yield = -ln(sum_withIR / sum_withoutIR) / mean_power
    Rows with non-positive values for sum_withIR, sum_withoutIR, or mean_power are set to NaN.

    Args:
        ion_df (pd.DataFrame): Ion data with required columns.

    Returns:
        pd.DataFrame: Ion data with a new 'ion_yield' column.
    """
    valid_mask = (
        (ion_df["sum_withIR"] > 0) &
        (ion_df["sum_withoutIR"] > 0) &
        (ion_df["mean_power"] > 0)
    )
    if not valid_mask.all():
        st.warning("Some rows have non-positive values in sum_withIR, sum_withoutIR, or mean_power. Those rows will be set to NaN.")
    ion_df["ion_yield"] = np.where(
        valid_mask,
        -np.log(ion_df["sum_withIR"] / ion_df["sum_withoutIR"]) / ion_df["mean_power"],
        np.nan
    )
    return ion_df

# ------------------------------------------------------------------
# Plotting Functions
# ------------------------------------------------------------------
def plot_static(processed_df):
    """
    Create a static matplotlib plot of Ion Yield vs. Wavenumber.

    Args:
        processed_df (pd.DataFrame): Processed DataFrame with 'wavenumber' and 'ion_yield'.

    Returns:
        tuple: (matplotlib.figure.Figure, BytesIO) containing the figure and PNG data.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(processed_df["wavenumber"], processed_df["ion_yield"],
            marker="o", linestyle="-", label="Ion Yield")
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Ion Yield")
    ax.set_title("Ion Yield vs. Wavenumber")
    ax.grid(True)
    ax.legend()

    # Save the figure to a BytesIO buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return fig, buf

def plot_interactive(processed_df):
    """
    Create an interactive Plotly plot of Ion Yield vs. Wavenumber.

    Args:
        processed_df (pd.DataFrame): Processed DataFrame.

    Returns:
        tuple: (Plotly Figure, str) containing the interactive figure and its HTML representation.
    """
    fig = px.line(
        processed_df,
        x="wavenumber",
        y="ion_yield",
        title="Ion Yield vs. Wavenumber",
        markers=True,
        labels={"ion_yield": "Ion Yield", "wavenumber": "Wavenumber (cm⁻¹)"}
    )
    html_str = fig.to_html()
    return fig, html_str

# ------------------------------------------------------------------
# Main Application UI
# ------------------------------------------------------------------

st.title("Ion Yield Normalization (Full Depletion Data)")

# --- Group File Uploads using an Expander ---
with st.expander("File Upload", expanded=True):
    st.header("1. Upload CSV Files")
    st.write(
        "Upload the required CSV files:\n"
        "- **Full Depletion Data CSV**: Must include 'wavenumber', 'sum_withoutIR', 'sum_withIR', "
        "and optionally 'depletion', '-ln(depletion)'.\n"
        "- **FELIX Power Scan CSV**: Must include 'undulator wavelength (µm)' and 'mean power (mJ)'."
    )
    ion_file = st.file_uploader(
        "Upload the 'Full Depletion Data' CSV",
        type=["csv"],
        help="CSV file with columns: wavenumber, sum_withoutIR, sum_withIR, and optionally depletion and -ln(depletion)."
    )
    power_file = st.file_uploader(
        "Upload the 'FELIX Power Scan' CSV",
        type=["csv"],
        help="CSV file with columns: undulator wavelength (µm) and mean power (mJ)."
    )
    if ion_file is not None:
        st.subheader("Specify the Unit for 'wavenumber'")
        unit = st.radio(
            "Is the wavenumber in cm⁻¹ or already in µm?",
            options=["cm⁻¹", "µm"],
            index=0,
            help="Select 'cm⁻¹' if your data is provided in wavenumber (cm⁻¹). "
                    "Select 'µm' if your data is already converted to wavelength (µm)."
        )
    else:
        unit = "cm⁻¹"

# --- Process Data if Both Files are Uploaded ---
if ion_file is not None and power_file is not None:
    # Read files using caching.
    ion_df = read_csv_cached(ion_file, encoding="latin-1")
    power_df = read_csv_cached(power_file, sep=";", encoding="latin-1")

    # Stop if there was an error reading files.
    if ion_df is None or power_df is None:
        st.stop()

    # Validate ion data.
    if not validate_ion_data(ion_df):
        st.stop()

    # Validate and process power scan data.
    power_df = validate_and_process_power_data(power_df)
    if power_df is None:
        st.stop()

    # --- Preview Raw Data in an Expander ---
    with st.expander("Preview Raw Data", expanded=False):
        st.subheader("Raw Full Depletion Data")
        st.dataframe(ion_df.head())
        st.subheader("Raw FELIX Power Scan Data")
        st.dataframe(power_df.head())

    # --- Process Data: Unit Conversion and Interpolation ---
    processed_ion_df = process_data(ion_df, power_df, unit)
    processed_ion_df = calculate_ion_yield(processed_ion_df)

    # Assemble a final processed DataFrame with wavenumber in cm⁻¹.
    processed_df = pd.DataFrame()
    if unit == "cm⁻¹":
        processed_df["wavenumber"] = ion_df["wavenumber"]
    else:
        processed_df["wavenumber"] = 1e4 / processed_ion_df["wavelength"]
    processed_df["sum_withoutIR"] = processed_ion_df["sum_withoutIR"]
    processed_df["sum_withIR"] = processed_ion_df["sum_withIR"]
    processed_df["mean_power"] = processed_ion_df["mean_power"]
    processed_df["ion_yield"] = processed_ion_df["ion_yield"]

    # --- Display Processed Data with Download Option ---
    with st.expander("Processed Data Preview", expanded=True):
        st.subheader("Processed Data")
        st.dataframe(processed_df.head())
        st.download_button(
            label="Download Processed Data as CSV",
            data=processed_df.to_csv(index=False),
            file_name="processed_data.csv",
            mime="text/csv"
        )

    # --- Plotting Section Using Tabs ---
    tab1, tab2 = st.tabs(["Static Plot", "Interactive Plot"])

    with tab1:
        st.subheader("Static Plot: Ion Yield vs. Wavenumber")
        fig_static, buf_static = plot_static(processed_df)
        st.pyplot(fig_static)
        st.download_button(
            label="Download Static Plot as PNG",
            data=buf_static,
            file_name="static_plot.png",
            mime="image/png"
        )

    with tab2:
        st.subheader("Interactive Plot: Ion Yield vs. Wavenumber")
        fig_interactive, html_interactive = plot_interactive(processed_df)
        st.plotly_chart(fig_interactive)
        st.download_button(
            label="Download Interactive Plot as HTML",
            data=html_interactive,
            file_name="interactive_plot.html",
            mime="text/html"
        )
