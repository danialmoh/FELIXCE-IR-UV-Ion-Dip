import streamlit as st
import numpy as np
import configparser
import os
import plotly.graph_objects as go

# Import variables from defaults.ini
def load_defaults():
    """Load default values from defaults.ini file"""
    config = configparser.ConfigParser()
    defaults_file = r'./.streamlit/defaults.ini'  # or provide full path
    defaults= {}
    if os.path.exists(defaults_file):
        try:
            config.read(defaults_file)
            # Update defaults with values from file
            defaults['element1'] = config.get('Complex Parameters', 'element1')
            defaults['element2'] = config.get('Complex Parameters', 'element2')
            defaults['element3'] = config.get('Complex Parameters', 'element3')
            defaults['mass_element1'] = config.getfloat('Complex Parameters', 'mass_element1')
            defaults['mass_element2'] = config.getfloat('Complex Parameters', 'mass_element2')
            defaults['mass_element3'] = config.getfloat('Complex Parameters', 'mass_element3')
            defaults['charge_state'] = config.get('Complex Parameters', 'charge_state')
            defaults['t_off'] = config.getfloat('Experiment Parameters', 't_off')
            defaults['alpha'] = config.getfloat('Experiment Parameters', 'alpha')
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults
defaults = load_defaults()


col1,col2,col3 = st.columns([0.5,1,1]) # col2 is just for spacing

with col1:
    # Species parameters #"What are the elements involved in your experiment?"
    st.markdown("### Species")
    st.session_state["element1"] = st.text_input("Element 1", value = st.session_state.get("element1", defaults.get("element1", None)))
    st.session_state["element2"] = st.text_input("Element 2", value = st.session_state.get("element2", defaults.get("element2", None)))
    st.session_state["element3"] = st.text_input("Element 3", value = st.session_state.get("element3", defaults.get("element3", None)))
    
with col2:#These inputs are converted to float (for numerical calculations) and stored in session_state.
    st.markdown("### Parameters")
    st.session_state["mass_element1"] = float(st.text_input("Mass of element1 in amu", value = st.session_state.get("mass_element1", defaults.get("mass_element1", None))))
    st.session_state["mass_element2"] = float(st.text_input("Mass of element2 in amu", value = st.session_state.get("mass_element2", defaults.get("mass_element2", None))))
    st.session_state["mass_element3"] = float(st.text_input("Mass of element3 in amu", value = st.session_state.get("mass_element3", defaults.get("mass_element3", None))))
    st.session_state["charge_state"] = st.text_input("Charge state", st.session_state.get("charge_state", defaults.get("charge_state", None)))

with col3:
    # Calibration parameters
    st.markdown("### Calibration parameters")
    st.session_state["t_off"] = st.number_input("t_off", value = st.session_state.get("t_off", defaults.get("t_off", None)), key="t_off_input")
    st.session_state["alpha"] = float(st.text_input("alpha", value = st.session_state.get("alpha", defaults.get("alpha", None)), key="alpha_input"))
    st.session_state["dataset_length"] = st.number_input("length of dataset in the time axis", value = st.session_state.get("dataset_length", defaults.get("dataset_length", None)))

st.markdown("---")

if st.button("✍️ Register inputs"):

    # Initialize variables
    mass_element = st.session_state.get("mass_element", None)
    t_off = st.session_state.get("t_off", None)
    alpha = st.session_state.get("alpha", None)
    dataset_length = st.session_state.get("dataset_length", None)                             

    # Generate an x-axis
    x_counts=np.linspace(1,dataset_length,dataset_length)
    # Calibrate spectra
    x_mass = alpha*(x_counts - t_off)**2
    # x_mass_perAtom = alpha*(x_counts - t_off)**2 / mass_element
    x_mass_perAtom = 0

    # Save the variables into memory
    st.session_state["x_mass"] = x_mass
    st.session_state["x_mass_perAtom"] = x_mass_perAtom
    st.success("Inputs registered! 😊")

# --- Mass Spectrum Preview (shown after registration) ---
if st.session_state.get("x_mass", None) is not None:
    st.markdown("---")
    st.markdown("### 📊 Mass Spectrum Preview")
    
    # Check if we have compiled data to plot (use compiled_data from import, or baseline corrected if available)
    compiled_data = st.session_state.get("compiled_data", None)
    compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
    
    # Prefer baseline corrected data if available, otherwise use raw compiled data
    data_to_plot = compilation_baseline_corrected_data if compilation_baseline_corrected_data is not None else compiled_data
    
    t_off = st.session_state.get("t_off", defaults.get("t_off", 58))
    alpha = st.session_state.get("alpha", defaults.get("alpha", 7.6987e-7))
    x_mass = st.session_state.get("x_mass")
    
    if data_to_plot is not None:
        # Sum signals across all wavenumbers to create total mass spectrum
        try:
            # Get first wavenumber to determine structure
            first_wn = list(data_to_plot.keys())[0]
            # Try to get signal data - structure differs between compiled_data and baseline_corrected_data
            first_df = data_to_plot[first_wn]
            
            # For compiled_data: sum all signal columns (excluding mass column)
            # For baseline_corrected_data: use -2 column (withoutIR)
            if 'compilation_baseline_corrected_data' in st.session_state and compilation_baseline_corrected_data is not None:
                summed_signal = first_df.iloc[:, -2].values  # withoutIR signal from baseline corrected
                for wn in list(data_to_plot.keys())[1:]:
                    summed_signal += data_to_plot[wn].iloc[:, -2].values
            else:
                # For compiled_data: sum all signal columns
                summed_signal = first_df.iloc[:, 1:].sum(axis=1).values  # Sum all signal columns
                for wn in list(data_to_plot.keys())[1:]:
                    summed_signal += data_to_plot[wn].iloc[:, 1:].sum(axis=1).values
            
            # Generate channel/bin axis
            dataset_length = st.session_state.get("dataset_length", len(summed_signal))
            x_channels = np.linspace(1, dataset_length, int(dataset_length))
            
            # Create two plots: Mass axis and Channel axis
            plot_tab1, plot_tab2 = st.tabs(["📊 Mass Spectrum (calibrated)", "🔢 Signal vs Channel Number (for calibration)"])
            
            with plot_tab1:
                fig_preview = go.Figure()
                fig_preview.add_trace(go.Scatter(
                    x=x_mass,
                    y=summed_signal,
                    mode='lines',
                    name='Summed Signal',
                    line=dict(color='blue')
                ))
                fig_preview.update_layout(
                    xaxis_title="Mass (amu)",
                    yaxis_title="Intensity",
                    title="Total Mass Spectrum (all wavenumbers summed)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_preview, use_container_width=True)
            
            with plot_tab2:
                st.caption("ℹ️ Hover over peaks to read their channel numbers. Enter them below — vertical lines will mark your selections on the plot.")
                
                # Calibration inputs ABOVE the plot so they update the plot markers
                col_cal1, col_cal2 = st.columns(2)
                with col_cal1:
                    st.markdown("**Peak 1 (lighter mass)**")
                    t1 = st.number_input("Channel number t₁", value=st.session_state.get("cal_t1", 0.0), key="cal_t1", help="Hover on plot to read channel number")
                    m1 = st.number_input("Known mass m₁ (amu)", value=st.session_state.get("cal_m1", 0.0), key="cal_m1")
                with col_cal2:
                    st.markdown("**Peak 2 (heavier mass)**")
                    t2 = st.number_input("Channel number t₂", value=st.session_state.get("cal_t2", 0.0), key="cal_t2", help="Hover on plot to read channel number")
                    m2 = st.number_input("Known mass m₂ (amu)", value=st.session_state.get("cal_m2", 0.0), key="cal_m2")
                
                # Build channel plot with vertical lines for selected peaks
                fig_channels = go.Figure()
                fig_channels.add_trace(go.Scatter(
                    x=x_channels,
                    y=summed_signal,
                    mode='lines',
                    name='Summed Signal',
                    line=dict(color='darkgreen')
                ))
                
                # Add vertical lines and markers for selected peak positions
                if t1 > 0:
                    idx1 = int(min(max(t1 - 1, 0), len(summed_signal) - 1))
                    fig_channels.add_vline(x=t1, line_dash="dash", line_color="red", annotation_text=f"t₁={t1:.0f}")
                    fig_channels.add_trace(go.Scatter(
                        x=[t1], y=[summed_signal[idx1]],
                        mode='markers', marker=dict(color='red', size=12, symbol='x'),
                        name=f'Peak 1 (ch {t1:.0f})', showlegend=True
                    ))
                if t2 > 0:
                    idx2 = int(min(max(t2 - 1, 0), len(summed_signal) - 1))
                    fig_channels.add_vline(x=t2, line_dash="dash", line_color="blue", annotation_text=f"t₂={t2:.0f}")
                    fig_channels.add_trace(go.Scatter(
                        x=[t2], y=[summed_signal[idx2]],
                        mode='markers', marker=dict(color='blue', size=12, symbol='x'),
                        name=f'Peak 2 (ch {t2:.0f})', showlegend=True
                    ))
                
                fig_channels.update_layout(
                    xaxis_title="Channel / Bin Number (counts)",
                    yaxis_title="Intensity",
                    title="Signal vs Channel Number — hover to read peak positions",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig_channels, use_container_width=True)
            
            # Show calibration info
            st.info(f"Current calibration: α = {alpha:.4e}, t_off = {t_off:.2f}")
            
            # --- Calculate Calibration Button ---
            st.markdown("---")
            if st.button("🔬 Calculate Calibration", use_container_width=True):
                try:
                    # Validate inputs
                    if t1 == t2:
                        st.error("❌ Error: t₁ and t₂ must be different!")
                    elif m1 <= 0 or m2 <= 0:
                        st.error("❌ Error: Masses must be positive!")
                    elif m1 == m2:
                        st.warning("⚠️ Warning: Using identical masses may not give meaningful calibration")
                    else:
                        # Calculate t_off using derived formula
                        sqrt_m1 = np.sqrt(m1)
                        sqrt_m2 = np.sqrt(m2)
                        
                        t_off_calc = (sqrt_m2 * t1 - sqrt_m1 * t2) / (sqrt_m2 - sqrt_m1)
                        
                        # Calculate alpha
                        alpha_calc = m1 / (t1 - t_off_calc)**2
                        
                        # Update session state
                        st.session_state["t_off"] = float(t_off_calc)
                        st.session_state["alpha"] = float(alpha_calc)
                        
                        # Display results
                        st.success("✅ Calibration calculated successfully!")
                        st.markdown(f"""
                        **Calculated Parameters:**
                        - **α** = `{alpha_calc:.6e}`
                        - **t_off** = `{t_off_calc:.4f}` counts
                        
                        Scroll up and click **"Register inputs"** to apply the new calibration.
                        """)
                        
                        # Verification
                        m1_check = alpha_calc * (t1 - t_off_calc)**2
                        m2_check = alpha_calc * (t2 - t_off_calc)**2
                        
                        st.markdown(f"""
                        **Verification:**
                        - Peak 1: Input mass = {m1:.4f} amu, Calculated = {m1_check:.4f} amu (error: {abs(m1-m1_check):.4e} amu)
                        - Peak 2: Input mass = {m2:.4f} amu, Calculated = {m2_check:.4f} amu (error: {abs(m2-m2_check):.4e} amu)
                        """)
                        
                except Exception as e:
                    st.error(f"❌ Error calculating calibration: {e}")
            
        except Exception as e:
            st.warning(f"Could not generate mass spectrum preview: {e}")
    else:
        st.info("ℹ️ Mass spectrum preview requires imported data from page 1.0")
else:
    st.info("👆 Click 'Register inputs' above to generate the mass spectrum preview and enable two-point calibration.")
