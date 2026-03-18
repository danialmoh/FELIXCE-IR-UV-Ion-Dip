import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

st.title("Calibrated Scans & IR Yield Aggregation")

# ------------------------------------------------------------------
# Upload scan CSVs
# ------------------------------------------------------------------
scan_files = st.file_uploader(
    label="Upload FELIX Scan CSVs for Calibration & Yield",
    type=["csv"], accept_multiple_files=True,
    help=("Each CSV must include columns: wavenumber, sum_withoutIR, "
          "sum_withIR, depletion, -ln(depletion).")
)

if scan_files:
    st.write("### Uploaded Files and Keys")
    file_keys = {}
    for idx, f in enumerate(scan_files):
        name = Path(f.name).stem
        file_keys[idx] = name
        st.write(f"**Key {idx}:** {name}")

    st.write("Use these keys when defining your calibration mapping below.")

    # ------------------------------------------------------------------
    # User inputs: mapping of file keys to calibration coeffs
    # ------------------------------------------------------------------
    st.write("### Define calibration coefficients for each file key")
    coeffs_input = st.text_area(
        "Enter a Python dict mapping file keys to coefficients:",
        value="""{
 0: {
     'coeffs_power': [1.13e-12, -1.28e-8, 5.75e-5, -1.27e-1, 1.37e+2, -5.87e+4],
     'coeffs_wavenumber': [0.9588, 20.3247]
 }
}""",
        height=200
    )

    # ------------------------------------------------------------------
    # Normalize button triggers processing
    # ------------------------------------------------------------------
    if st.button("Normalize"):
        try:
            mapping = eval(coeffs_input)
        except Exception as e:
            st.error(f"Failed to parse mapping: {e}")
            st.stop()

        all_processed = []
        for key, f in enumerate(scan_files):
            # Read CSV
            df = pd.read_csv(f, sep=',', encoding='ISO-8859-1')
            # Ensure required columns exist
            required_cols = ['wavenumber', 'sum_withoutIR', 'sum_withIR', 'depletion', '-ln(depletion)']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.warning(f"Skipping '{f.name}': missing columns {missing}")
                continue

            # Extract by column name
            raw       = df['wavenumber'].astype(float)
            sig_no    = df['sum_withoutIR'].astype(float)
            sig_ir    = df['sum_withIR'].astype(float)
            depletion = df['depletion'].astype(float)
            ln_dep    = df['-ln(depletion)'].astype(float)

            # Get coefficients for this file
            coeffs = mapping.get(key)
            if coeffs is None:
                st.warning(f"No coefficients defined for key {key}, skipping '{f.name}'.")
                continue
            cp = coeffs['coeffs_power']
            cw = coeffs['coeffs_wavenumber']

            # Apply calibrations
            wavenumber_cal = np.polyval(cw, raw)
            yield_ir        = ln_dep / np.polyval(cp, raw)

            # Build per-file table
            processed = pd.DataFrame({
                'filename': Path(f.name).stem,
                'wavenumber': raw,
                'wavenumber_calibrated': wavenumber_cal,
                'sum_withoutIR': sig_no,
                'sum_withIR': sig_ir,
                'depletion': depletion,
                '-ln(depletion)': ln_dep,
                'yield_IR': yield_ir
            })
            # Save individual calibrated CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"Calibrated_{Path(f.name).stem}_{timestamp}.csv"
            processed.to_csv(out_name, index=False)
            all_processed.append(processed)

        if not all_processed:
            st.warning("No files processed.")
        else:
            # Concatenate all processed data
            combined = pd.concat(all_processed, ignore_index=True)

            # Average duplicates at same calibrated wavenumber
            avg = (
                combined
                .groupby('wavenumber_calibrated', as_index=False)
                .agg({
                    'sum_withoutIR':'mean',
                    'sum_withIR':'mean',
                    'depletion':'mean',
                    '-ln(depletion)':'mean',
                    'yield_IR':'mean'
                })
            )

            st.write("### Combined and Averaged Data")
            st.dataframe(avg)

            # Plot all individual yields
            fig, ax = plt.subplots()
            for name, group in combined.groupby('filename'):
                ax.plot(group['wavenumber_calibrated'], group['yield_IR'], label=name)
            ax.set_xlabel('Calibrated Wavenumber (cm⁻¹)')
            ax.set_ylabel('Yield IR')
            ax.legend()
            st.pyplot(fig)

            # Plot averaged yield
            fig2, ax2 = plt.subplots()
            ax2.plot(avg['wavenumber_calibrated'], avg['yield_IR'], marker='o')
            ax2.set_xlabel('Calibrated Wavenumber (cm⁻¹)')
            ax2.set_ylabel('Average Yield IR')
            st.pyplot(fig2)

else:
    st.info("Upload one or more scan CSVs to begin.")
