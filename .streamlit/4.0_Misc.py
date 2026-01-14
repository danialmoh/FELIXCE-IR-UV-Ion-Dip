import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D  # For matplotlib 3D plots
from matplotlib import cm

# --- USER INPUT FOR RANGES ---
import streamlit as st

# --- USER INPUT FOR RANGES ---
col1, col2 = st.columns(2)
with col1:
    # Read inputs into local variables using unique keys
    min_wavenumber_input = st.number_input(
        "Enter Minimum Wavenumber (cm⁻¹)",
        value=st.session_state.get("min_wavenumber", 0.0),
        step=0.1,
        key="min_wavenumber_input"
    )
    max_wavenumber_input = st.number_input(
        "Enter Maximum Wavenumber (cm⁻¹)",
        min_value=min_wavenumber_input,  # use the local variable here
        value=st.session_state.get("max_wavenumber", 2000.0),
        step=0.1,
        key="max_wavenumber_input"
    )
    # Update session state with the new values
    st.session_state["min_wavenumber"] = min_wavenumber_input
    st.session_state["max_wavenumber"] = max_wavenumber_input

with col2:
    # Do the same for m/z if needed (optional: if you want to keep any dependencies, use local vars)
    min_m_z_input = st.number_input(
        "Enter Minimum m/z",
        value=st.session_state.get("min_m_z", 0.0),
        step=1.0,
        key="min_m_z_input"
    )
    max_m_z_input = st.number_input(
        "Enter Maximum m/z",
        min_value=min_m_z_input,  # if you want to enforce min ≤ max; otherwise remove this
        value=st.session_state.get("max_m_z", 2500.0),
        step=1.0,
        key="max_m_z_input"
    )
    st.session_state["min_m_z"] = min_m_z_input
    st.session_state["max_m_z"] = max_m_z_input

# --- PROCESS DATA FIRST ---
if st.button("✨ Process Data for Plots"):
    # Retrieve necessary data from session_state
    x_mass = st.session_state.get("x_mass")
    compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data")
    unique_wavenumbers = st.session_state.get("unique_wavenumbers")
    
    if x_mass is not None and compilation_baseline_corrected_data is not None and unique_wavenumbers is not None:
        # Get user-defined ranges
        min_wavenumber = st.session_state["min_wavenumber"]
        max_wavenumber = st.session_state["max_wavenumber"]
        min_m_z = st.session_state["min_m_z"]
        max_m_z = st.session_state["max_m_z"]

        # Combine data for the selected ranges
        combined_data = pd.DataFrame()
        for wavenumber in unique_wavenumbers:
            if min_wavenumber <= float(wavenumber) <= max_wavenumber:
                data_for_wavenumber = compilation_baseline_corrected_data[wavenumber]
                signal_without_ir = data_for_wavenumber.iloc[:, st.session_state["plot_columnIndex_withoutIR"]]
                signal_with_ir = data_for_wavenumber.iloc[:, st.session_state["plot_columnIndex_withIR"]]

                # Avoid division by zero by adding a small constant (if needed)
                ratio = signal_with_ir / (signal_without_ir)
                temp_data = pd.DataFrame({
                    "m/z": x_mass,
                    "Intensity (No IR)": signal_without_ir,
                    "Intensity (With IR)": signal_with_ir,
                    "-ln(Depletion)": -np.log(ratio),
                    "Wavenumber": [wavenumber] * len(x_mass)
                })

                temp_data = temp_data[(temp_data["m/z"] >= min_m_z) & (temp_data["m/z"] <= max_m_z)]
                combined_data = pd.concat([combined_data, temp_data], ignore_index=True)

        if not combined_data.empty:
            # Pivot data for 2D heatmaps
            pivot_ln_depletion = combined_data.pivot(index="Wavenumber", columns="m/z", values="-ln(Depletion)")
            pivot_signal_with_ir = combined_data.pivot(index="Wavenumber", columns="m/z", values="Intensity (With IR)")
            pivot_signal_without_ir = combined_data.pivot(index="Wavenumber", columns="m/z", values="Intensity (No IR)")

            # Save to session state for later access
            st.session_state["combined_data"] = combined_data
            st.session_state["pivot_ln_depletion"] = pivot_ln_depletion
            st.session_state["pivot_signal_with_ir"] = pivot_signal_with_ir
            st.session_state["pivot_signal_without_ir"] = pivot_signal_without_ir

            st.success("Data processed successfully! You can now generate the plots.")
        else:
            st.warning("No data available in the selected ranges.")
    else:
        st.error("Required data is missing. Ensure preprocessing steps are completed.")

# --- SMOOTHING LEVEL SELECTION (always visible) ---
sigma_options = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
selected_sigma = st.selectbox(
    "Smoothing Level (σ)",
    sigma_options,
    index=sigma_options.index(2.0) if 2.0 in sigma_options else 0
)

# --- GENERATE 3 HEATMAPS ---
if st.button("🔳 Generate 3D Heatmaps"):
    if ("pivot_ln_depletion" in st.session_state and 
        "pivot_signal_with_ir" in st.session_state and 
        "pivot_signal_without_ir" in st.session_state):

        pivot_ln_depletion = st.session_state["pivot_ln_depletion"]
        pivot_signal_with_ir = st.session_state["pivot_signal_with_ir"]
        pivot_signal_without_ir = st.session_state["pivot_signal_without_ir"]

        # Threshold for intensity values: values above THRESHOLD will be highlighted in red.
        THRESHOLD = 0.025

        # Process Intensity (With IR)
        signal_with_ir = pivot_signal_with_ir.copy()
        signal_with_ir_clipped = signal_with_ir.clip(upper=THRESHOLD)
        signal_with_ir_mask = signal_with_ir > THRESHOLD

        # Process Intensity (No IR)
        signal_without_ir = pivot_signal_without_ir.copy()
        signal_without_ir_clipped = signal_without_ir.clip(upper=THRESHOLD)
        signal_without_ir_mask = signal_without_ir > THRESHOLD

        # Helper function: Gaussian smoothing that ignores NaNs
        def nan_gaussian_filter(data, sigma):
            mask = np.isfinite(data).astype(float)
            data_filled = np.where(np.isfinite(data), data, 0)
            data_smoothed = gaussian_filter(data_filled, sigma=sigma)
            mask_smoothed = gaussian_filter(mask, sigma=sigma)
            with np.errstate(divide='ignore', invalid='ignore'):
                data_normalized = data_smoothed / mask_smoothed
            data_normalized[mask_smoothed == 0] = np.nan
            return data_normalized

        # Apply Gaussian smoothing to -ln(Depletion)
        Z_smoothed = nan_gaussian_filter(pivot_ln_depletion.values, sigma=selected_sigma)
        fig_ln_depletion = go.Figure(data=go.Heatmap(
            z=Z_smoothed,
            x=pivot_ln_depletion.columns,
            y=pivot_ln_depletion.index,
            colorscale="Viridis",
            colorbar=dict(title="Smoothed -ln(Depletion)")
        ))
        fig_ln_depletion.update_layout(title="Smoothed -ln(Depletion) Heatmap",xaxis_title="m/z",            # Assign x-axis label
    yaxis_title="Wavenumber (cm⁻¹)" )# Assign y-axis label
        st.plotly_chart(fig_ln_depletion, use_container_width=True)

        # Heatmap for Signal With IR:
        # Values ≤ THRESHOLD are shown with the "Plasma" colorscale.
        # Values > THRESHOLD are overlaid in red.
        fig_signal_with_ir = go.Figure(data=go.Heatmap(
            z=np.where(signal_with_ir_mask, np.nan, signal_with_ir_clipped),
            x=pivot_signal_with_ir.columns,
            y=pivot_signal_with_ir.index,
            colorscale="Plasma",
            colorbar=dict(title=f"Signal With IR (≤ {THRESHOLD})")
        ))
        fig_signal_with_ir.add_trace(go.Heatmap(
            z=np.where(signal_with_ir_mask, 1, np.nan),
            x=pivot_signal_with_ir.columns,
            y=pivot_signal_with_ir.index,
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False
        ))
        fig_signal_with_ir.update_layout(
            title=f"Signal With IR Heatmap (Values > {THRESHOLD} colored RED)"
        )
        st.plotly_chart(fig_signal_with_ir, use_container_width=True)

        # Heatmap for Signal Without IR:
        # Values ≤ THRESHOLD use the "Cividis" colorscale.
        # Values > THRESHOLD are overlaid in red.
        fig_signal_without_ir = go.Figure(data=go.Heatmap(
            z=np.where(signal_without_ir_mask, np.nan, signal_without_ir_clipped),
            x=pivot_signal_without_ir.columns,
            y=pivot_signal_without_ir.index,
            colorscale="Cividis",
            colorbar=dict(title=f"Signal Without IR (≤ {THRESHOLD})")
        ))
        fig_signal_without_ir.add_trace(go.Heatmap(
            z=np.where(signal_without_ir_mask, 1, np.nan),
            x=pivot_signal_without_ir.columns,
            y=pivot_signal_without_ir.index,
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False
        ))
        fig_signal_without_ir.update_layout(
            title=f"Signal Without IR Heatmap (Values > {THRESHOLD} colored RED)"
        )
        st.plotly_chart(fig_signal_without_ir, use_container_width=True)

    else:
        st.error("Please process the data first.")

# --- GENERATE 4D (3D Surface) PLOT USING MATPLOTLIB ---
if st.button("📊 Generate 4D Surface Plot"):
    if "combined_data" in st.session_state:
        combined_data = st.session_state["combined_data"]

        if not combined_data.empty:
            # Extract required data
            X = combined_data["m/z"].values
            Y = combined_data["Wavenumber"].values
            Z = combined_data["-ln(Depletion)"].values
            C = combined_data["Intensity (With IR)"].values  # For color mapping

            # Normalize color values for gradient scaling
            if C.max() != C.min():
                C_normalized = (C - C.min()) / (C.max() - C.min())
            else:
                C_normalized = np.zeros_like(C)

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel("Mass-to-Charge Ratio (m/z)")
            ax.set_ylabel("Wavenumber (cm⁻¹)")
            ax.set_zlabel("-ln(Depletion)")
            ax.set_title("3D Surface Plot: m/z vs Wavenumber vs -ln(Depletion)")

            # Create a surface using plot_trisurf
            surf = ax.plot_trisurf(X, Y, Z, cmap=cm.viridis, linewidth=0.2, antialiased=True)
            cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
            cbar.set_label("Normalized Intensity (With IR)")

            st.pyplot(fig)
        else:
            st.warning("No data available in the selected ranges.")
    else:
        st.error("Please process the data first.")

# --- GENERATE INTERACTIVE 3D SURFACE PLOT USING PLOTLY ---
if st.button("🛠 Generate Interactive 3D Surface Plot"):
    if "pivot_ln_depletion" in st.session_state and "pivot_signal_with_ir" in st.session_state:
        pivot_table_z = st.session_state["pivot_ln_depletion"]
        pivot_table_color = st.session_state["pivot_signal_with_ir"]

        X, Y = np.meshgrid(pivot_table_z.columns, pivot_table_z.index)
        Z = pivot_table_z.values
        C = pivot_table_color.values  # Color values from "Intensity (With IR)"

        # Apply logarithmic scaling to colors (adding a small constant to avoid log(0))
        C_log = np.log(C + 1e-6)

        fig = go.Figure(data=[go.Surface(
            z=Z,
            x=X,
            y=Y,
            surfacecolor=C_log,
            colorscale="Viridis",
            colorbar=dict(title="Log(Intensity With IR)")
        )])
        fig.update_layout(
            title="Interactive 3D Surface Plot",
            scene=dict(
                xaxis=dict(title="Mass-to-Charge Ratio (m/z)"),
                yaxis=dict(title="Wavenumber (cm⁻¹)"),
                zaxis=dict(title="-ln(Depletion)")
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Please process the data first.")
