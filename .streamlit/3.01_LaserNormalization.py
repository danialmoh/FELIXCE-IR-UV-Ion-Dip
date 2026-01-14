# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# from io import BytesIO

# # ------------------------------------------------------------------
# # Caching for File Reading
# # ------------------------------------------------------------------
# @st.cache_data
# def read_csv_cached(file, sep=",", encoding="latin-1"):
#     """
#     Read a CSV file with caching.

#     Args:
#         file (UploadedFile): File uploaded via Streamlit.
#         sep (str): CSV separator.
#         encoding (str): File encoding.

#     Returns:
#         pd.DataFrame or None: DataFrame if successful; otherwise, None.
#     """
#     try:
#         df = pd.read_csv(file, sep=sep, encoding=encoding)
#         return df
#     except Exception as e:
#         st.error(f"Error reading CSV file: {e}")
#         return None

# # ------------------------------------------------------------------
# # File Upload & Validation Functions
# # ------------------------------------------------------------------
# def validate_ion_data(df):
#     """
#     Validate that the ion data file contains the required columns.

#     Args:
#         df (pd.DataFrame): Ion data DataFrame.

#     Returns:
#         bool: True if valid, False otherwise.
#     """
#     required_columns = ["wavenumber", "sum_withoutIR", "sum_withIR"]
#     missing = [col for col in required_columns if col not in df.columns]
#     if missing:
#         st.error(f"The ion data file is missing required columns: {', '.join(missing)}. "
#                  "Ensure your data has 'wavenumber', 'sum_withoutIR', and 'sum_withIR'.")
#         return False
#     return True

# def validate_and_process_power_data(df):
#     """
#     Validate and process the FELIX Power Scan data.
#     Renames required columns for consistency and sorts by 'undulator_wavelength'.

#     Expected original columns:
#         - "undulator wavelength (µm)"
#         - "mean power (mJ)"

#     Args:
#         df (pd.DataFrame): Power scan DataFrame.

#     Returns:
#         pd.DataFrame or None: Processed DataFrame if successful; otherwise, None.
#     """
#     rename_cols = {
#         "undulator wavelength (µm)": "undulator_wavelength",
#         "mean power (mJ)": "mean_power"
#     }
#     df.rename(columns=rename_cols, inplace=True)
#     if "undulator_wavelength" not in df.columns or "mean_power" not in df.columns:
#         st.error("FELIX Power Scan CSV must contain 'undulator wavelength (µm)' and 'mean power (mJ)' columns.")
#         return None
#     df.sort_values("undulator_wavelength", inplace=True)
#     return df

# # ------------------------------------------------------------------
# # Data Processing & Unit Conversion
# # ------------------------------------------------------------------
# @st.cache_data
# def process_data(ion_df, power_df, unit):
#     """
#     Process the ion and power scan data:
#       - Converts ion data from wavenumber to wavelength if needed.
#       - Interpolates the mean power from the power scan onto the ion data.

#     Args:
#         ion_df (pd.DataFrame): Ion data.
#         power_df (pd.DataFrame): Processed power scan data.
#         unit (str): "cm⁻¹" if ion data is in wavenumber, "µm" if already in wavelength.

#     Returns:
#         pd.DataFrame: Ion data with an interpolated 'mean_power' column.
#     """
#     ion_df = ion_df.copy()
#     power_df = power_df.copy()

#     # Convert ion data: if unit is cm⁻¹, convert to wavelength in µm.
#     if unit == "cm⁻¹":
#         ion_df["wavelength"] = 1e4 / ion_df["wavenumber"]
#     else:
#         # If already in µm, rename column for clarity.
#         ion_df.rename(columns={"wavenumber": "wavelength"}, inplace=True)

#     # Interpolate mean power based on wavelength.
#     ion_df["mean_power"] = np.interp(
#         ion_df["wavelength"],
#         power_df["undulator_wavelength"],
#         power_df["mean_power"]
#     )

#     # Warn if some wavelengths fall outside the power scan range.
#     if (ion_df["wavelength"].min() < power_df["undulator_wavelength"].min() or
#         ion_df["wavelength"].max() > power_df["undulator_wavelength"].max()):
#         st.warning("Some wavelengths are outside the FELIX Power Scan range. Interpolation may be inaccurate.")

#     return ion_df

# # ------------------------------------------------------------------
# # Calculation of Normalized Ion Yield
# # ------------------------------------------------------------------
# def calculate_ion_yield(ion_df):
#     """
#     Calculate the normalized ion yield using the formula:
#       ion_yield = -ln(sum_withIR / sum_withoutIR) / mean_power
#     Rows with non-positive values for sum_withIR, sum_withoutIR, or mean_power are set to NaN.

#     Args:
#         ion_df (pd.DataFrame): Ion data with required columns.

#     Returns:
#         pd.DataFrame: Ion data with a new 'ion_yield' column.
#     """
#     valid_mask = (
#         (ion_df["sum_withIR"] > 0) &
#         (ion_df["sum_withoutIR"] > 0) &
#         (ion_df["mean_power"] > 0)
#     )
#     if not valid_mask.all():
#         st.warning("Some rows have non-positive values in sum_withIR, sum_withoutIR, or mean_power. Those rows will be set to NaN.")
#     ion_df["ion_yield"] = np.where(
#         valid_mask,
#         -np.log(ion_df["sum_withIR"] / ion_df["sum_withoutIR"]) / ion_df["mean_power"],
#         np.nan
#     )
#     return ion_df

# # ------------------------------------------------------------------
# # Plotting Functions
# # ------------------------------------------------------------------
# def plot_static(processed_df):
#     """
#     Create a static matplotlib plot of Ion Yield vs. Wavenumber.

#     Args:
#         processed_df (pd.DataFrame): Processed DataFrame with 'wavenumber' and 'ion_yield'.

#     Returns:
#         tuple: (matplotlib.figure.Figure, BytesIO) containing the figure and PNG data.
#     """
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.plot(processed_df["wavenumber"], processed_df["ion_yield"],
#             marker="o", linestyle="-", label="Ion Yield")
#     ax.set_xlabel("Wavenumber (cm⁻¹)")
#     ax.set_ylabel("Ion Yield")
#     ax.set_title("Ion Yield vs. Wavenumber")
#     ax.grid(True)
#     ax.legend()

#     # Save the figure to a BytesIO buffer.
#     buf = BytesIO()
#     fig.savefig(buf, format="png", bbox_inches="tight")
#     buf.seek(0)
#     return fig, buf

# def plot_interactive(processed_df):
#     """
#     Create an interactive Plotly plot of Ion Yield vs. Wavenumber.

#     Args:
#         processed_df (pd.DataFrame): Processed DataFrame.

#     Returns:
#         tuple: (Plotly Figure, str) containing the interactive figure and its HTML representation.
#     """
#     fig = px.line(
#         processed_df,
#         x="wavenumber",
#         y="ion_yield",
#         title="Ion Yield vs. Wavenumber",
#         markers=True,
#         labels={"ion_yield": "Ion Yield", "wavenumber": "Wavenumber (cm⁻¹)"}
#     )
#     html_str = fig.to_html()
#     return fig, html_str

# # ------------------------------------------------------------------
# # Main Application UI
# # ------------------------------------------------------------------

# st.title("3.01 Ion Yield Normalization")

# # Check if data from Section 3.0 is available in session state
# if 'fullrange_depletion_data' in st.session_state and not st.session_state.fullrange_depletion_data.empty:
#     # Use the depletion data from Section 3.0
#     st.success("Successfully loaded depletion data from Section 3.0")
#     # Convert to DataFrame if it's not already (some implementations might store it differently)
#     if isinstance(st.session_state.fullrange_depletion_data, pd.DataFrame):
#         ion_df = st.session_state.fullrange_depletion_data.copy()
#     else:
#         # Try to convert from array or list format if necessary
#         try:
#             ion_df = pd.DataFrame(st.session_state.fullrange_depletion_data)
#             # Check if we need to assign column names
#             if ion_df.shape[1] >= 5 and not all(col in ion_df.columns for col in ["wavenumber", "sum_withoutIR", "sum_withIR"]):
#                 ion_df.columns = ["wavenumber", "sum_withoutIR", "sum_withIR", "depletion", "-ln(depletion)"]
#         except Exception as e:
#             st.error(f"Error converting depletion data: {e}")
#             st.stop()
    
#     # Validate that the ion data has the required columns
#     if not validate_ion_data(ion_df):
#         st.error("The depletion data from Section 3.0 does not have the required columns.")
#         st.stop()
    
#     # Display a preview of the loaded data
#     with st.expander("Preview Depletion Data from Section 3.0", expanded=False):
#         st.dataframe(ion_df.head())
# else:
#     st.warning("No depletion data found from Section 3.0. Please complete Section 3.0 first.")
    
#     # Fallback option to upload data directly
#     with st.expander("Alternatively, Upload Depletion Data Manually", expanded=True):
#         ion_file = st.file_uploader(
#             "Upload the 'Full Depletion Data' CSV",
#             type=["csv"],
#             help="CSV file with columns: wavenumber, sum_withoutIR, sum_withIR, and optionally depletion and -ln(depletion)."
#         )
        
#         if ion_file is not None:
#             ion_df = read_csv_cached(ion_file, encoding="latin-1")
#             if ion_df is None or not validate_ion_data(ion_df):
#                 st.stop()
#         else:
#             st.info("Please either complete Section 3.0 first or upload depletion data manually.")
#             st.stop()

# # Now ask for the FELIX Power Scan data
# with st.expander("FELIX Power Scan Upload", expanded=True):
#     st.subheader("Upload FELIX Power Scan Data")
#     power_file = st.file_uploader(
#         "Upload the 'FELIX Power Scan' CSV",
#         type=["csv"],
#         help="CSV file with columns: undulator wavelength (µm) and mean power (mJ)."
#     )

# # Specify the unit for wavenumber
# st.subheader("Specify the Unit for 'wavenumber'")
# unit = st.radio(
#     "Is the wavenumber in cm⁻¹ or already in µm?",
#     options=["cm⁻¹", "µm"],
#     index=0,
#     help="Select 'cm⁻¹' if your data is provided in wavenumber (cm⁻¹). "
#          "Select 'µm' if your data is already converted to wavelength (µm)."
# )

# # --- Process Data if Power Scan is Uploaded ---
# if power_file is not None:
#     # Read power scan file
#     power_df = read_csv_cached(power_file, sep=";", encoding="latin-1")
#     if power_df is None:
#         st.stop()

#     # Validate and process power scan data
#     power_df = validate_and_process_power_data(power_df)
#     if power_df is None:
#         st.stop()

#     # Preview power scan data
#     with st.expander("Preview FELIX Power Scan Data", expanded=False):
#         st.dataframe(power_df.head())

#     # --- Process Data: Unit Conversion and Interpolation ---
#     processed_ion_df = process_data(ion_df, power_df, unit)
#     processed_ion_df = calculate_ion_yield(processed_ion_df)

#     # Assemble a final processed DataFrame with wavenumber in cm⁻¹
#     processed_df = pd.DataFrame()
#     if unit == "cm⁻¹":
#         processed_df["wavenumber"] = ion_df["wavenumber"]
#     else:
#         processed_df["wavenumber"] = 1e4 / processed_ion_df["wavelength"]
#     processed_df["sum_withoutIR"] = processed_ion_df["sum_withoutIR"]
#     processed_df["sum_withIR"] = processed_ion_df["sum_withIR"]
#     processed_df["mean_power"] = processed_ion_df["mean_power"]
#     processed_df["ion_yield"] = processed_ion_df["ion_yield"]

#     # Keep the original depletion columns if they exist
#     if "depletion" in ion_df.columns:
#         processed_df["depletion"] = ion_df["depletion"]
#     if "-ln(depletion)" in ion_df.columns:
#         processed_df["-ln(depletion)"] = ion_df["-ln(depletion)"]

#     # Store the processed data in session state for use in Section 3.1
#     st.session_state.normalized_data = processed_df

#     # --- Display Processed Data with Download Option ---
#     with st.expander("Processed Data Preview", expanded=True):
#         st.subheader("Processed Data")
#         st.dataframe(processed_df.head())
#         st.download_button(
#             label="Download Processed Data as CSV",
#             data=processed_df.to_csv(index=False),
#             file_name="normalized_data.csv",
#             mime="text/csv"
#         )

#     # --- Plotting Section Using Tabs ---
#     tab1, tab2 = st.tabs(["Static Plot", "Interactive Plot"])

#     with tab1:
#         st.subheader("Static Plot: Ion Yield vs. Wavenumber")
#         fig_static, buf_static = plot_static(processed_df)
#         st.pyplot(fig_static)
#         st.download_button(
#             label="Download Static Plot as PNG",
#             data=buf_static,
#             file_name="static_plot.png",
#             mime="image/png"
#         )

#     with tab2:
#         st.subheader("Interactive Plot: Ion Yield vs. Wavenumber")
#         fig_interactive, html_interactive = plot_interactive(processed_df)
#         st.plotly_chart(fig_interactive)
#         st.download_button(
#             label="Download Interactive Plot as HTML",
#             data=html_interactive,
#             file_name="interactive_plot.html",
#             mime="text/html"
#         )
# else:
#     st.info("Please upload a FELIX Power Scan CSV to continue with normalization.") 
# CODE2
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit
from scipy.stats import chi2
from io import BytesIO
import plotly.express as px

# App title
st.title("FELIX Power Calibration & Ion Yield Normalization")

# ------------------------------------------------------------------
# Utility: Cached CSV Reader
# ------------------------------------------------------------------
@st.cache_data
def read_csv_cached(file, sep=",", encoding="latin-1"):
    """
    Read a CSV file with caching; override sep as needed for semicolons.
    """
    try:
        return pd.read_csv(file, sep=sep, encoding=encoding)
    except Exception as e:
        name = getattr(file, 'name', str(file))
        st.error(f"Error reading CSV '{name}': {e}")
        return None

# ------------------------------------------------------------------
# Section 1: Calibration Curve Fit
# ------------------------------------------------------------------
with st.expander("1. Calibration Curve Fit", expanded=True):
    calib_files = st.file_uploader(
        label="Upload FELIX Power Scan CSVs for Calibration (semicolon-delimited)",
        type=["csv"], accept_multiple_files=True,
        help=("Each CSV must use ';' as delimiter and include columns:"
              " 'undulator wavelength (µm)', 'mean power (mJ)';"
              " additional columns 'standard deviation' and"
              " 'spectrum analyzer mean wavelength (µm)' are allowed and ignored in the fit.")
    )
    max_order = st.number_input(
        "Max Polynomial Order to Test", min_value=1, max_value=10, value=5, step=1
    )

    if calib_files:
        # Load and concatenate all uploaded CSVs
        all_data = []
        for f in calib_files:
            df = read_csv_cached(f, sep=";")
            if df is None:
                st.stop()

            required = ['undulator wavelength (µm)', 'mean power (mJ)']
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"File '{f.name}' missing columns: {', '.join(missing)}")
                st.stop()

            df = df.rename(
                columns={
                    'undulator wavelength (µm)': 'raw_reading',
                    'mean power (mJ)': 'reference_power'
                }
            )
            df = df[['raw_reading', 'reference_power']].copy()
            df['raw_reading'] = pd.to_numeric(df['raw_reading'], errors='coerce')
            df['reference_power'] = pd.to_numeric(df['reference_power'], errors='coerce')
            if df[['raw_reading','reference_power']].isnull().any().any():
                st.error(f"Non-numeric values in '{f.name}' after conversion.")
                st.stop()

            all_data.append(df)

        calib_df = pd.concat(all_data, ignore_index=True)
        x = calib_df['raw_reading'].values
        y = calib_df['reference_power'].values

        # Polynomial model definition
        def poly_model(x, *coeffs):
            order = len(coeffs) - 1
            return sum(coeffs[i] * x**(order - i) for i in range(len(coeffs)))

        # Automatic degree selection based on p-value of chi2 goodness-of-fit
        best_degree = 1
        best_pval = -np.inf
        for deg in range(1, max_order + 1):
            p0 = np.zeros(deg + 1)
            try:
                popt_temp, _ = curve_fit(poly_model, x, y, p0=p0)
            except Exception:
                continue
            residuals = y - poly_model(x, *popt_temp)
            chi2_stat = np.sum(residuals**2 / poly_model(x, *popt_temp))
            dof = len(x) - (deg + 1)
            p_val = 1 - chi2.cdf(chi2_stat, df=dof)
            if p_val > best_pval:
                best_pval = p_val
                best_degree = deg

        # Fit with the automatically selected best degree
        initial_guess = np.zeros(best_degree + 1)
        popt, pcov = curve_fit(poly_model, x, y, p0=initial_guess)
        st.session_state['calib_coeffs'] = popt
        st.success(f"Selected polynomial order {best_degree} with p-value={best_pval:.4f}.")
        # Print the coefficients
        st.write("Fitted coefficients (highest degree first):", popt)
        # Generate fit curve
        xx = np.linspace(x.min(), x.max(), 300)
        yy = poly_model(xx, *popt)

        # Build equation text for annotation
        terms = [f"{coef:.3e} x^{i}" for i, coef in enumerate(popt[::-1])]
        equation = " + ".join(terms).replace("x^0", "")
        # Display the equation
        st.write(f"Fitted polynomial equation: y = {equation}")
        # Plotting
        fig, ax = plt.subplots()
        ax.scatter(x, y, s=10, label='Data')
        ax.plot(xx, yy, color='red',
                label=f'Order {best_degree}, p-value={best_pval:.4f}')
        ax.set_xlabel('Undulator Wavelength (µm)')
        ax.set_ylabel('Mean Power (mJ)')
        ax.set_title(calib_files[0].name.replace('.csv', ''))
        ax.text(0.05, 0.95, f"y = {equation}", transform=ax.transAxes,
                va='top', fontsize=9)
        ax.grid(True)

        # Customize ticks
        ax.set_xticks(np.linspace(x.min(), x.max(), 6))
        ax.set_yticks(np.linspace(y.min(), y.max(), 6))
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
# ------------------------------------------------------------------
# Section 2: Upload & Filter Power Scan CSVs
# ------------------------------------------------------------------
with st.expander("2. Upload & Filter Power Scan CSVs", expanded=True):
    scan_files = st.file_uploader(
        label="Upload FELIX Power Scan CSVs (semicolon-delimited)",
        type=["csv"], accept_multiple_files=True,
        help=("CSV must use ';' and include 'undulator wavelength (µm)' &"
              " 'mean power (mJ)'; extra columns are ignored.")
    )
    if scan_files:
        min_wl = st.number_input("Min Wavelength (µm)", value=0.5, format="%.3f")
        max_wl = st.number_input("Max Wavelength (µm)", value=2.0, format="%.3f")
        scans = []
        for f in scan_files:
            df = read_csv_cached(f, sep=";")
            if df is None or not {'undulator wavelength (µm)', 'mean power (mJ)'}.issubset(df.columns):
                st.warning(f"Skipping '{f.name}': missing required FELIX columns.")
                continue
            df = df.rename(
                columns={
                    'undulator wavelength (µm)': 'wavelength',
                    'mean power (mJ)': 'signal'
                }
            )
            df['wavelength'] = pd.to_numeric(df['wavelength'], errors='coerce')
            df['signal'] = pd.to_numeric(df['signal'], errors='coerce')
            if df[['wavelength','signal']].isnull().any().any():
                st.warning(f"Skipping '{f.name}': non-numeric data.")
                continue
            mean_wl = df['wavelength'].mean()
            if not (min_wl <= mean_wl <= max_wl):
                continue
            df['wavenumber'] = 1e4 / df['wavelength']
            scans.append(df[['wavelength','wavenumber','signal']])
        if not scans:
            st.warning("No scans matched the specified range.")
        else:
            fig2, ax2 = plt.subplots()
            for df in scans:
                ax2.plot(df['wavenumber'], df['signal'], label=f"{df['wavelength'].mean():.3f} µm")
            ax2.set_xlabel('Wavenumber (cm⁻¹)')
            ax2.set_ylabel('Signal (a.u.)')
            ax2.legend()
            st.pyplot(fig2)
            st.session_state['scan_data'] = pd.concat(scans, ignore_index=True)

# ------------------------------------------------------------------
# # Section 3: Ion Yield Normalization
# # ------------------------------------------------------------------
# with st.expander("3. Ion Yield Normalization", expanded=True):
#     ion_df = None
#     if 'fullrange_depletion_data' in st.session_state:
#         ion_df = pd.DataFrame(st.session_state['fullrange_depletion_data'])
#     else:
#         ion_file = st.file_uploader(
#             "Upload Depletion CSV (comma-delimited)", type=["csv"]
#         )
#         if ion_file:
#             ion_df = read_csv_cached(ion_file, sep=",")
#     if ion_df is None or not {'wavenumber','sum_withoutIR','sum_withIR'}.issubset(ion_df.columns):
#         st.error("Missing depletion columns: wavenumber, sum_withoutIR, sum_withIR")
#         st.stop()

#     unit = st.selectbox("Input unit for wavenumber column", options=["cm⁻¹","µm"], index=0)
#     if unit == 'cm⁻¹':
#         ion_df['wavelength'] = 1e4 / ion_df['wavenumber']
#     else:
#         ion_df = ion_df.rename(columns={'wavenumber':'wavelength'})

#     if 'scan_data' in st.session_state:
#         scan_df = st.session_state['scan_data']
#         ion_df['mean_power'] = np.interp(
#             ion_df['wavelength'], scan_df['wavelength'], scan_df['signal']
#         )
#     elif 'calib_coeffs' in st.session_state:
#         coeffs = st.session_state['calib_coeffs']
#         ion_df['mean_power'] = np.polyval(coeffs, ion_df['wavelength'])
#     else:
#         st.error("No power data: run calibration or upload scans first.")
#         st.stop()

#     mask = (
#         (ion_df['sum_withIR']>0)&(ion_df['sum_withoutIR']>0)&(ion_df['mean_power']>0)
#     )
#     ion_df['ion_yield'] = np.where(
#         mask,
#         -np.log(ion_df['sum_withIR']/ion_df['sum_withoutIR']) / ion_df['mean_power'],
#         np.nan
#     )

#     st.dataframe(ion_df.head())
#     st.download_button("Download Normalized Data", ion_df.to_csv(index=False), "normalized_data.csv", "text/csv")

#     fig3, ax3 = plt.subplots()
#     ax3.plot(ion_df['wavenumber'], ion_df['ion_yield'], marker='o', linestyle='-')
#     ax3.set_xlabel('Wavenumber (cm⁻¹)')
#     ax3.set_ylabel('Ion Yield')
#     st.pyplot(fig3)

#     fig_int = px.line(ion_df, x='wavenumber', y='ion_yield', markers=True)
#     st.plotly_chart(fig_int)
