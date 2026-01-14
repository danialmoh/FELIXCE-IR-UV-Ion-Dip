# # import streamlit as st
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import os
# # import scipy.ndimage as ndimage  # For Gaussian smoothing
# # import numpy as np
# # from scipy.signal import find_peaks, savgol_filter
# # from scipy.optimize import curve_fit
# # import plotly.express as px
# # import plotly.graph_objects as go
# # import io

# # # ----- Wes Anderson Color Palette -----
# # from palettable.wesanderson import Moonrise5_6, Royal1_4
# # color_palette = Moonrise5_6.hex_colors + Royal1_4.hex_colors

# # def get_color(idx):
# #     return color_palette[idx % len(color_palette)]

# # # ----- Smoothing Function -----
# # def smooth_data(series, window_size=3, method="Moving Average"):
# #     if method == "Moving Average":
# #         return series.rolling(window=window_size, center=True).mean()
# #     elif method == "Exponential Moving Average":
# #         return series.ewm(span=window_size, adjust=False).mean()
# #     elif method == "Gaussian":
# #         smoothed_array = ndimage.gaussian_filter1d(series.values, sigma=window_size)
# #         return pd.Series(smoothed_array, index=series.index)
# #     elif method == "Savitzky–Golay":
# #         # Ensure the window size is odd for Savitzky–Golay filter
# #         if window_size % 2 == 0:
# #             window_size += 1
# #         # Choose a polynomial order; must be less than window_size
# #         polyorder = 2 if window_size > 2 else 1
# #         smoothed_array = savgol_filter(series.values, window_length=window_size, polyorder=polyorder)
# #         return pd.Series(smoothed_array, index=series.index)
# #     else:
# #         return series

# # # ----- Define Gaussian Function for Fitting -----
# # def gaussian(x, A, mu, sigma):
# #     return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# # # ----- App Title and Description -----
# # st.title("Depletion Data Analysis App")
# # st.markdown("""
# # This app processes IR-UV ion dip spectroscopy data by smoothing, plotting, estimating noise, detecting peaks, and fitting Gaussian profiles.
# # """)

# # # ----- Sidebar: Global Configuration -----
# # st.sidebar.header("Configuration")
# # uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

# # smoothing_method = st.sidebar.selectbox("Smoothing Method", 
# #                                         options=["Moving Average", "Exponential Moving Average", "Savitzky–Golay", "Gaussian", "None"])
# # apply_smoothing = smoothing_method != "None"
# # if apply_smoothing:
# #     window_size = st.sidebar.slider("Smoothing Window Size", min_value=1, max_value=21, value=3, step=1)
# # else:
# #     window_size = 1

# # # Option to include original (noisy) data in plots
# # show_original = st.sidebar.checkbox("Include Original Data in Plot", value=True)

# # peak_data_choice = st.sidebar.selectbox("Data for Peak Analysis", options=["Smoothed", "Original"])
# # peak_input = st.sidebar.text_input("Enter Peak Wavenumbers (comma-separated)", value="")
# # peak_labels = st.sidebar.text_input("Enter Corresponding Peak Labels (comma-separated)", value="")

# # output_directory = st.sidebar.text_input("Output Directory (default: current folder)", value=os.getcwd())

# # # Button to process files
# # process_button = st.sidebar.button("Process Files")

# # # ----- Process Files and Store in Session State -----
# # if process_button:
# #     if not uploaded_files:
# #         st.sidebar.warning("Please upload at least one file.")
# #     else:
# #         combined_plot_data = []  # List to hold processed file data
# #         for uploaded_file in uploaded_files:
# #             file_name = uploaded_file.name
# #             try:
# #                 data = pd.read_csv(uploaded_file)
# #             except Exception as e:
# #                 st.error(f"Error reading {file_name}: {e}")
# #                 continue

# #             # Check for required columns
# #             required_columns = ['wavenumber', 'depletion', '-ln(depletion)']
# #             if not all(col in data.columns for col in required_columns):
# #                 st.error(f"File {file_name} does not contain the required columns: {required_columns}")
# #                 continue

# #             # Extract columns
# #             wavenumber = data['wavenumber']
# #             depletion = data['depletion']
# #             ln_depletion = data['-ln(depletion)']

# #             # Apply smoothing if requested
# #             if apply_smoothing:
# #                 smoothed_depletion = smooth_data(depletion, window_size, smoothing_method)
# #                 smoothed_ln_depletion = smooth_data(ln_depletion, window_size, smoothing_method)
# #             else:
# #                 smoothed_depletion = depletion
# #                 smoothed_ln_depletion = ln_depletion

# #             combined_plot_data.append((file_name, wavenumber, depletion, smoothed_depletion, ln_depletion, smoothed_ln_depletion))

# #             # Save individual plot automatically if output directory is provided
# #             fig_temp, axs_temp = plt.subplots(2, 1, figsize=(12, 8))
# #             if show_original:
# #                 axs_temp[0].plot(wavenumber, depletion, linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
# #             axs_temp[0].plot(wavenumber, smoothed_depletion, linewidth=2, label="Smoothed", color=get_color(1))
# #             axs_temp[0].set_xlabel("Wavenumber (cm⁻¹)")
# #             axs_temp[0].set_ylabel("Depletion")
# #             axs_temp[0].set_title(f"{file_name}: Depletion vs Wavenumber")
# #             axs_temp[0].legend()
# #             axs_temp[0].grid(True)
            
# #             if show_original:
# #                 axs_temp[1].plot(wavenumber, ln_depletion, linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
# #             axs_temp[1].plot(wavenumber, smoothed_ln_depletion, linewidth=2, label="Smoothed", color=get_color(1))
# #             axs_temp[1].set_xlabel("Wavenumber (cm⁻¹)")
# #             axs_temp[1].set_ylabel("-ln(Depletion)")
# #             axs_temp[1].set_title(f"{file_name}: -ln(Depletion) vs Wavenumber")
# #             axs_temp[1].legend()
# #             axs_temp[1].grid(True)
# #             plt.tight_layout()
# #             if output_directory:
# #                 if not os.path.exists(output_directory):
# #                     os.makedirs(output_directory)
# #                 base_name = os.path.splitext(file_name)[0]
# #                 method_str = "NoSmoothing" if not apply_smoothing else smoothing_method.replace(" ", "")
# #                 output_file = os.path.join(output_directory, f"{base_name}_{method_str}_ws{window_size}_plot.png")
# #                 fig_temp.savefig(output_file, dpi=300)
# #             plt.close(fig_temp)

# #         st.session_state.combined_plot_data = combined_plot_data
# #         st.success("Files processed successfully. Use the tabs below for further analysis.")

# # # ----- Only show analysis tabs if files have been processed -----
# # if "combined_plot_data" in st.session_state and st.session_state.combined_plot_data:
# #     # Create five tabs: Individual Plots, Combined Plot, Noise Region, Peak Analysis, Smoothing Comparison
# #     tabs = st.tabs(["Individual Plots", "Combined Plot", "Noise Region", "Peak Analysis", "Smoothing Comparison"])

# #     # ----- Tab 1: Individual Plots -----
# #     with tabs[0]:
# #         st.header("Individual File Plots")
# #         for (file_name, wavenumber, depletion, smoothed_depletion, ln_depletion, smoothed_ln_depletion) in st.session_state.combined_plot_data:
# #             fig, axs = plt.subplots(2, 1, figsize=(12, 8))
# #             if show_original:
# #                 axs[0].plot(wavenumber, depletion, linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
# #             axs[0].plot(wavenumber, smoothed_depletion, linewidth=2, label="Smoothed", color=get_color(1))
# #             axs[0].set_xlabel("Wavenumber (cm⁻¹)")
# #             axs[0].set_ylabel("Depletion")
# #             axs[0].set_title(f"{file_name}: Depletion vs Wavenumber")
# #             axs[0].legend()
# #             axs[0].grid(True)
            
# #             if show_original:
# #                 axs[1].plot(wavenumber, ln_depletion, linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
# #             axs[1].plot(wavenumber, smoothed_ln_depletion, linewidth=2, label="Smoothed", color=get_color(1))
# #             axs[1].set_xlabel("Wavenumber (cm⁻¹)")
# #             axs[1].set_ylabel("-ln(Depletion)")
# #             axs[1].set_title(f"{file_name}: -ln(Depletion) vs Wavenumber")
# #             axs[1].legend()
# #             axs[1].grid(True)
# #             plt.tight_layout()
# #             st.pyplot(fig)

# #     # ----- Tab 2: Combined Plot with Peak Annotations -----
# #     with tabs[1]:
# #         st.header("Combined -ln(Depletion) Plot with Peak Annotations")
# #         combined_width = st.sidebar.number_input("Combined Plot Width", value=28, min_value=1)
# #         combined_height = st.sidebar.number_input("Combined Plot Height", value=8, min_value=1)
# #         ranges = [(400, 1200), (1200, 2000), (2700, 3200)]
# #         fig, axs = plt.subplots(3, 1, figsize=(combined_width, combined_height * 3))
# #         for ax, (low, high) in zip(axs, ranges):
# #             for idx, (file_name, wavenumber, depletion, smoothed_depletion, ln_depletion, smoothed_ln_depletion) in enumerate(st.session_state.combined_plot_data):
# #                 mask = (wavenumber >= low) & (wavenumber <= high)
# #                 if mask.sum() > 0:
# #                     if show_original:
# #                         ax.plot(wavenumber[mask], ln_depletion[mask], linestyle='--', alpha=0.5,
# #                                 label=f"{file_name} Noisy", color=get_color(2*idx))
# #                     ax.plot(wavenumber[mask], smoothed_ln_depletion[mask], linewidth=2,
# #                             label=f"{file_name} Smoothed", color=get_color(2*idx+1))
# #             if peak_input and peak_labels:
# #                 try:
# #                     peak_values = [float(val.strip()) for val in peak_input.split(',') if val.strip()]
# #                     peak_names = [name.strip() for name in peak_labels.split(',') if name.strip()]
# #                     if len(peak_values) != len(peak_names):
# #                         st.error("The number of peak values and labels must be equal!")
# #                     else:
# #                         y_max = ax.get_ylim()[1]
# #                         for p, name in zip(peak_values, peak_names):
# #                             if low <= p <= high:
# #                                 ax.axvline(x=p, color=get_color(0), linestyle='--', linewidth=1)
# #                                 ax.text(p, y_max, name, rotation=70,
# #                                         verticalalignment='bottom', color=get_color(0), fontsize=14)
# #                 except Exception as e:
# #                     st.error(f"Error processing peak annotations: {e}")
# #             ax.set_xlabel("Wavenumber (cm⁻¹)")
# #             ax.set_ylabel("-ln(Depletion)")
# #             ax.set_xlim(low, high)
# #             ax.grid(True)
# #             ax.legend()
# #         plt.tight_layout()
# #         st.pyplot(fig)
# #         if output_directory:
# #             file_names = [data[0] for data in st.session_state.combined_plot_data]
# #             common_prefix = os.path.commonprefix(file_names).strip("_-")
# #             if not common_prefix:
# #                 common_prefix = "combined_plot"
# #             method_str = "NoSmoothing" if not apply_smoothing else smoothing_method.replace(" ", "")
# #             output_file_combined = os.path.join(
# #                 output_directory,
# #                 f"{common_prefix}_{method_str}_ws{window_size}_combined_ln_depletion_plot.png"
# #             )
# #             fig.savefig(output_file_combined, dpi=300)
# #             st.write(f"Combined plot saved to: {output_file_combined}")

# #     # ----- Tab 3: Manual Noise Region Selection -----
# #     with tabs[2]:
# #         st.header("Manual Noise Region Selection")
# #         file_name, wavenumber, depletion, smoothed_depletion, ln_depletion, smoothed_ln_depletion = st.session_state.combined_plot_data[0]
# #         noise_data = smoothed_ln_depletion if peak_data_choice == "Smoothed" else ln_depletion
# #         df_plot = pd.DataFrame({
# #             "wavenumber": wavenumber,
# #             "-ln(depletion)": noise_data
# #         })
# #         fig_noise = px.line(df_plot, x="wavenumber", y="-ln(depletion)",
# #                             title=f"Zoom and Select Noise Region ({file_name})")
# #         st.plotly_chart(fig_noise, use_container_width=True)

# #         noise_start = st.number_input("Noise Region Start (cm⁻¹)", value=float(wavenumber.iloc[0]))
# #         noise_end = st.number_input("Noise Region End (cm⁻¹)", value=float(wavenumber.iloc[-1]))
# #         if st.button("Set Noise Region"):
# #             wn = wavenumber.values if isinstance(wavenumber, pd.Series) else np.array(wavenumber)
# #             y_signal = noise_data.values if isinstance(noise_data, pd.Series) else np.array(noise_data)
# #             noise_indices = np.where((wn >= noise_start) & (wn <= noise_end))[0]
# #             if len(noise_indices) == 0:
# #                 st.error("No data points found in the specified noise region. Please adjust your values.")
# #             else:
# #                 noise_region_manual = y_signal[noise_indices]
# #                 manual_noise_level = np.std(noise_region_manual)
# #                 st.success(f"Manual Noise Level Set: {manual_noise_level:.4f}")
# #                 st.write("Noise region spans from", noise_start, "to", noise_end)
# #                 st.session_state.manual_noise = {"start": noise_start, "end": noise_end, "noise_level": manual_noise_level}

# #     # ----- Tab 4: Peak Analysis and Gaussian Fitting -----
# #     with tabs[3]:
# #         st.header("Peak Analysis and Gaussian Fitting")
# #         peak_results = []  # To store fitting results for all files
# #         for (file_name, wavenumber, depletion, smoothed_depletion, ln_depletion, smoothed_ln_depletion) in st.session_state.combined_plot_data:
# #             y_signal = smoothed_ln_depletion.values if peak_data_choice == "Smoothed" else ln_depletion.values
# #             wn = wavenumber.values if isinstance(wavenumber, pd.Series) else np.array(wavenumber)
# #             n_points = len(wn)
            
# #             # Default noise estimation from first and last 10%
# #             noise_region = np.concatenate([y_signal[:max(1, n_points//10)], y_signal[-max(1, n_points//10):]])
# #             noise_level = np.std(noise_region)
# #             if "manual_noise" in st.session_state:
# #                 noise_start = st.session_state.manual_noise["start"]
# #                 noise_end = st.session_state.manual_noise["end"]
# #                 noise_idx = np.where((wn >= noise_start) & (wn <= noise_end))[0]
# #                 if len(noise_idx) > 0:
# #                     noise_region = y_signal[noise_idx]
# #                     noise_level = np.std(noise_region)
            
# #             peaks, properties = find_peaks(y_signal, height=2 * noise_level)
# #             fitted_peaks = []
# #             window_half_width = max(5, n_points // 20)
# #             for peak in peaks:
# #                 start_idx = max(0, peak - window_half_width)
# #                 end_idx = min(n_points, peak + window_half_width)
# #                 x_window = wn[start_idx:end_idx]
# #                 y_window = y_signal[start_idx:end_idx]
# #                 try:
# #                     p0 = [y_signal[peak], wn[peak], (x_window[-1] - x_window[0]) / 6]
# #                     popt, _ = curve_fit(gaussian, x_window, y_window, p0=p0)
# #                     fitted_peaks.append({"file": file_name,
# #                                          "peak_index": peak,
# #                                          "A": popt[0],
# #                                          "mu": popt[1],
# #                                          "sigma": popt[2],
# #                                          "noise_level": noise_level})
# #                 except Exception as e:
# #                     st.error(f"Gaussian fit failed for file {file_name} at peak index {peak}: {e}")

# #             fig_peak, ax_peak = plt.subplots(figsize=(12, 6))
# #             ax_peak.plot(wn, y_signal, label="Data", color=get_color(0))
# #             ax_peak.plot(wn[peaks], y_signal[peaks], "x", color=get_color(1), label="Detected Peaks")
# #             x_dense = np.linspace(wn[0], wn[-1], 1000)
# #             for i, fit in enumerate(fitted_peaks):
# #                 ax_peak.plot(x_dense, gaussian(x_dense, fit["A"], fit["mu"], fit["sigma"]), 
# #                              label=f"Fit: μ={fit['mu']:.2f}", color=get_color(i+2))
# #             ax_peak.set_xlabel("Wavenumber (cm⁻¹)")
# #             ax_peak.set_ylabel("-ln(Depletion)")
# #             ax_peak.set_title(f"Peak Analysis for {file_name}\nEstimated Noise Level = {noise_level:.3f}")
# #             ax_peak.legend()
# #             ax_peak.grid(True)
# #             st.pyplot(fig_peak)
            
# #             for fit in fitted_peaks:
# #                 peak_results.append({
# #                     "File": file_name,
# #                     "Peak Index": fit["peak_index"],
# #                     "Amplitude": fit["A"],
# #                     "Center": fit["mu"],
# #                     "Sigma": fit["sigma"],
# #                     "Noise Level": fit["noise_level"]
# #                 })
        
# #         if peak_results:
# #             st.header("Gaussian Fitting Results")
# #             df_results = pd.DataFrame(peak_results)
# #             st.dataframe(df_results)
# #         else:
# #             st.info("No peaks detected with the current settings.")

# #     # ----- Tab 5: Smoothing Comparison -----
# #     with tabs[4]:
# #         st.header("Smoothing Comparison")
# #         # Initialize comparison pairs if not already done
# #         if "comparison_pairs" not in st.session_state:
# #             st.session_state.comparison_pairs = []

# #         st.subheader("Add Smoothing Method & Window Size Pair")
# #         with st.form("comparison_pair_form", clear_on_submit=True):
# #             comp_method = st.selectbox("Select Smoothing Method", 
# #                                        options=["Moving Average", "Exponential Moving Average", "Savitzky–Golay", "Gaussian"])
# #             comp_window_size = st.number_input("Window Size", min_value=1, max_value=21, value=3, step=1)
# #             submitted = st.form_submit_button("Add Pair")
# #             if submitted:
# #                 st.session_state.comparison_pairs.append({"method": comp_method, "window_size": comp_window_size})
# #                 st.success(f"Added pair: {comp_method} with window size {comp_window_size}")

# #         st.subheader("Current Comparison Pairs")
# #         if st.session_state.comparison_pairs:
# #             for i, pair in enumerate(st.session_state.comparison_pairs):
# #                 st.write(f"Pair {i+1}: {pair['method']} with window size {pair['window_size']}")
# #         else:
# #             st.info("No comparison pairs added yet.")

# #         include_original = st.checkbox("Include Original Data", value=True)
        
# #         if st.button("compareeee"):
# #             if not st.session_state.comparison_pairs and not include_original:
# #                 st.error("Please add at least one comparison pair or select to include the original data.")
# #             else:
# #                 file_name, wavenumber, depletion, _, _, _ = st.session_state.combined_plot_data[0]
# #                 # Create static matplotlib figure
# #                 fig_comp, ax_comp = plt.subplots(figsize=(16, 6))
# #                 color_idx = 0
# #                 if include_original:
# #                     ax_comp.plot(wavenumber, depletion, linewidth=2, label="Original", color=get_color(color_idx))
# #                     color_idx += 1
# #                 for pair in st.session_state.comparison_pairs:
# #                     data_to_plot = smooth_data(depletion, pair["window_size"], pair["method"])
# #                     label = f"{pair['method']} (ws={pair['window_size']})"
# #                     ax_comp.plot(wavenumber, data_to_plot, linewidth=2, label=label, color=get_color(color_idx))
# #                     color_idx += 1
# #                 ax_comp.set_xlabel("Wavenumber (cm⁻¹)")
# #                 ax_comp.set_ylabel("Depletion")
# #                 ax_comp.set_title(f"Smoothing Comparison on {file_name}")
# #                 ax_comp.legend()
# #                 ax_comp.grid(True)
# #                 st.pyplot(fig_comp)
                
# #                 # Prepare static plot download as PNG
# #                 buf = io.BytesIO()
# #                 fig_comp.savefig(buf, format="png")
# #                 buf.seek(0)
# #                 st.download_button(label="Download Static Plot (PNG)",
# #                                    data=buf,
# #                                    file_name="smoothing_comparison.png",
# #                                    mime="image/png")
                
# #                 # Create interactive Plotly figure using the same color assignments
# #                 fig_plotly = go.Figure()
# #                 color_idx = 0
# #                 if include_original:
# #                     fig_plotly.add_trace(go.Scatter(x=wavenumber, y=depletion,
# #                                                     mode="lines", name="Original",
# #                                                     line=dict(color=get_color(color_idx))))
# #                     color_idx += 1
# #                 for pair in st.session_state.comparison_pairs:
# #                     data_to_plot = smooth_data(depletion, pair["window_size"], pair["method"])
# #                     fig_plotly.add_trace(go.Scatter(x=wavenumber, y=data_to_plot,
# #                                                     mode="lines",
# #                                                     name=f"{pair['method']} (ws={pair['window_size']})",
# #                                                     line=dict(color=get_color(color_idx))))
# #                     color_idx += 1
# #                 fig_plotly.update_layout(title=f"Smoothing Comparison on {file_name}",
# #                                          xaxis_title="Wavenumber (cm⁻¹)",
# #                                          yaxis_title="Depletion")
# #                 st.plotly_chart(fig_plotly, use_container_width=True)
                
# #                 # Prepare interactive plot download as HTML
# #                 html_bytes = fig_plotly.to_html(full_html=True).encode("utf-8")
# #                 st.download_button(label="Download Interactive Plot (HTML)",
# #                                    data=html_bytes,
# #                                    file_name="smoothing_comparison.html",
# #                                    mime="text/html")
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import scipy.ndimage as ndimage  # For Gaussian smoothing
# import numpy as np
# from scipy.signal import find_peaks, savgol_filter
# from scipy.optimize import curve_fit
# import plotly.express as px
# import plotly.graph_objects as go
# import io

# # ----- Wes Anderson Color Palette -----
# from palettable.wesanderson import Moonrise5_6, Royal1_4
# color_palette = Moonrise5_6.hex_colors + Royal1_4.hex_colors

# def get_color(idx):
#     return color_palette[idx % len(color_palette)]

# # ----- Smoothing Function -----
# def smooth_data(series, window_size=3, method="Moving Average"):
#     if method == "Moving Average":
#         return series.rolling(window=window_size, center=True).mean()
#     elif method == "Exponential Moving Average":
#         return series.ewm(span=window_size, adjust=False).mean()
#     elif method == "Gaussian":
#         smoothed_array = ndimage.gaussian_filter1d(series.values, sigma=window_size)
#         return pd.Series(smoothed_array, index=series.index)
#     elif method == "Savitzky–Golay":
#         if window_size % 2 == 0:
#             window_size += 1
#         polyorder = 2 if window_size > 2 else 1
#         smoothed_array = savgol_filter(series.values, window_length=window_size, polyorder=polyorder)
#         return pd.Series(smoothed_array, index=series.index)
#     else:
#         return series

# # ----- Define Gaussian Function for Fitting -----
# def gaussian(x, A, mu, sigma):
#     return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# # ----- App Title and Description -----
# st.title("Depletion Data Analysis App")
# st.markdown("""
# This app processes IR-UV ion dip spectroscopy data by smoothing, plotting, estimating noise, detecting peaks, and fitting Gaussian profiles.
# """)

# # ----- Sidebar: Global Configuration -----
# st.sidebar.header("Configuration")
# # Add a radio button to select which data type to plot
# data_type = st.sidebar.radio("Select Data to Plot", options=["Depletion", "Ion Yield"])

# uploaded_files = st.sidebar.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

# smoothing_method = st.sidebar.selectbox("Smoothing Method", 
#                                         options=["Moving Average", "Exponential Moving Average", "Savitzky–Golay", "Gaussian", "None"])
# apply_smoothing = smoothing_method != "None"
# if apply_smoothing:
#     window_size = st.sidebar.slider("Smoothing Window Size", min_value=1, max_value=21, value=3, step=1)
# else:
#     window_size = 1

# show_original = st.sidebar.checkbox("Include Original Data in Plot", value=True)
# peak_data_choice = st.sidebar.selectbox("Data for Peak Analysis", options=["Smoothed", "Original"])
# peak_input = st.sidebar.text_input("Enter Peak Wavenumbers (comma-separated)", value="")
# peak_labels = st.sidebar.text_input("Enter Corresponding Peak Labels (comma-separated)", value="")

# output_directory = st.sidebar.text_input("Output Directory (default: current folder)", value=os.getcwd())

# process_button = st.sidebar.button("Process Files")

# # ----- Process Files and Store in Session State -----
# if process_button:
#     if not uploaded_files:
#         st.sidebar.warning("Please upload at least one file.")
#     else:
#         combined_plot_data = []  # List to hold processed file data
#         for uploaded_file in uploaded_files:
#             file_name = uploaded_file.name
#             try:
#                 data = pd.read_csv(uploaded_file)
#             except Exception as e:
#                 st.error(f"Error reading {file_name}: {e}")
#                 continue

#             if data_type == "Depletion":
#                 # Check for required depletion columns
#                 required_columns = ['wavenumber', 'depletion', '-ln(depletion)']
#                 if not all(col in data.columns for col in required_columns):
#                     st.error(f"File {file_name} does not contain the required columns: {required_columns}")
#                     continue
#                 wavenumber = data['wavenumber']
#                 depletion = data['depletion']
#                 ln_depletion = data['-ln(depletion)']
#                 if apply_smoothing:
#                     smoothed_depletion = smooth_data(depletion, window_size, smoothing_method)
#                     smoothed_ln_depletion = smooth_data(ln_depletion, window_size, smoothing_method)
#                 else:
#                     smoothed_depletion = depletion
#                     smoothed_ln_depletion = ln_depletion
#                 combined_plot_data.append((file_name, wavenumber, depletion, smoothed_depletion, ln_depletion, smoothed_ln_depletion))
#             elif data_type == "Ion Yield":
#                 # Check for ion_yield column
#                 if 'ion_yield' not in data.columns:
#                     st.error(f"File {file_name} does not contain the required column: 'ion_yield'")
#                     continue
#                 wavenumber = data['wavenumber']
#                 ion_yield = data['ion_yield']
#                 if apply_smoothing:
#                     smoothed_ion_yield = smooth_data(ion_yield, window_size, smoothing_method)
#                 else:
#                     smoothed_ion_yield = ion_yield
#                 # For uniformity, we store (file_name, wavenumber, original_data, smoothed_data, None, None)
#                 combined_plot_data.append((file_name, wavenumber, ion_yield, smoothed_ion_yield, None, None))
                
#                 # Optionally, you can compute additional transforms (like -ln) if needed for ion yield.
                
#             # (Optional) Save individual plot automatically if output directory is provided
#             # [Your saving code here...]
            
#         st.session_state.combined_plot_data = combined_plot_data
#         st.success("Files processed successfully. Use the tabs below for further analysis.")

# # ----- Show Analysis Tabs Only if Files Have Been Processed -----
# if "combined_plot_data" in st.session_state and st.session_state.combined_plot_data:
#     # Create tabs – note that some tabs may be relevant only for depletion data.
#     tabs = st.tabs(["Individual Plots", "Combined Plot", "Noise Region", "Peak Analysis", "Smoothing Comparison"])

#     # ----- Tab 1: Individual Plots -----
#     with tabs[0]:
#         st.header("Individual File Plots")
#         for (file_name, wavenumber, original, smoothed, ln_data, smoothed_ln) in st.session_state.combined_plot_data:
#             if data_type == "Depletion":
#                 fig, axs = plt.subplots(2, 1, figsize=(12, 8))
#                 if show_original:
#                     axs[0].plot(wavenumber, original, linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
#                 axs[0].plot(wavenumber, smoothed, linewidth=2, label="Smoothed", color=get_color(1))
#                 axs[0].set_xlabel("Wavenumber (cm⁻¹)")
#                 axs[0].set_ylabel("Depletion")
#                 axs[0].set_title(f"{file_name}: Depletion vs Wavenumber")
#                 axs[0].legend()
#                 axs[0].grid(True)
                
#                 if show_original:
#                     axs[1].plot(wavenumber, ln_data, linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
#                 axs[1].plot(wavenumber, smoothed_ln, linewidth=2, label="Smoothed", color=get_color(1))
#                 axs[1].set_xlabel("Wavenumber (cm⁻¹)")
#                 axs[1].set_ylabel("-ln(Depletion)")
#                 axs[1].set_title(f"{file_name}: -ln(Depletion) vs Wavenumber")
#                 axs[1].legend()
#                 axs[1].grid(True)
#                 plt.tight_layout()
#                 st.pyplot(fig)
#             elif data_type == "Ion Yield":
#                 fig, ax = plt.subplots(figsize=(12, 6))
#                 if show_original:
#                     ax.plot(wavenumber, original, linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
#                 ax.plot(wavenumber, smoothed, linewidth=2, label="Smoothed", color=get_color(1))
#                 ax.set_xlabel("Wavenumber (cm⁻¹)")
#                 ax.set_ylabel("Ion Yield")
#                 ax.set_title(f"{file_name}: Ion Yield vs Wavenumber")
#                 ax.legend()
#                 ax.grid(True)
#                 plt.tight_layout()
#                 st.pyplot(fig)

#     # ----- Tab 2: Combined Plot -----
#     with tabs[1]:
#         st.header("Combined Plot")
#         combined_width = st.sidebar.number_input("Combined Plot Width", value=28, min_value=1)
#         combined_height = st.sidebar.number_input("Combined Plot Height", value=8, min_value=1)
#         ranges = [(400, 1200), (1200, 2000), (2700, 3200)]
#         fig, axs = plt.subplots(3, 1, figsize=(combined_width, combined_height * 3))
#         for ax, (low, high) in zip(axs, ranges):
#             for idx, (file_name, wavenumber, original, smoothed, ln_data, smoothed_ln) in enumerate(st.session_state.combined_plot_data):
#                 mask = (wavenumber >= low) & (wavenumber <= high)
#                 if mask.sum() > 0:
#                     if data_type == "Depletion":
#                         if show_original:
#                             ax.plot(wavenumber[mask], ln_data[mask], linestyle='--', alpha=0.5,
#                                     label=f"{file_name} Noisy", color=get_color(2*idx))
#                         ax.plot(wavenumber[mask], smoothed_ln[mask], linewidth=2,
#                                 label=f"{file_name} Smoothed", color=get_color(2*idx+1))
#                         ax.set_ylabel("-ln(Depletion)")
#                     elif data_type == "Ion Yield":
#                         if show_original:
#                             ax.plot(wavenumber[mask], original[mask], linestyle='--', alpha=0.5,
#                                     label=f"{file_name} Noisy", color=get_color(2*idx))
#                         ax.plot(wavenumber[mask], smoothed[mask], linewidth=2,
#                                 label=f"{file_name} Smoothed", color=get_color(2*idx+1))
#                         ax.set_ylabel("Ion Yield")
#             if peak_input and peak_labels and data_type == "Depletion":
#                 try:
#                     peak_values = [float(val.strip()) for val in peak_input.split(',') if val.strip()]
#                     peak_names = [name.strip() for name in peak_labels.split(',') if name.strip()]
#                     if len(peak_values) != len(peak_names):
#                         st.error("The number of peak values and labels must be equal!")
#                     else:
#                         y_max = ax.get_ylim()[1]
#                         for p, name in zip(peak_values, peak_names):
#                             if low <= p <= high:
#                                 ax.axvline(x=p, color=get_color(0), linestyle='--', linewidth=1)
#                                 ax.text(p, y_max, name, rotation=70,
#                                         verticalalignment='bottom', color=get_color(0), fontsize=14)
#                 except Exception as e:
#                     st.error(f"Error processing peak annotations: {e}")
#             ax.set_xlabel("Wavenumber (cm⁻¹)")
#             ax.set_xlim(low, high)
#             ax.grid(True)
#             ax.legend()
#         plt.tight_layout()
#         st.pyplot(fig)
#         # (Optional) Save combined plot to output directory as done in your original code

#     # ----- Additional Tabs for Depletion Analysis -----
#     if data_type == "Depletion":
#         # Tab 3: Noise Region Selection
#         with tabs[2]:
#             st.header("Manual Noise Region Selection")
#             file_name, wavenumber, depletion, smoothed_depletion, ln_depletion, smoothed_ln_depletion = st.session_state.combined_plot_data[0]
#             noise_data = smoothed_ln_depletion if peak_data_choice == "Smoothed" else ln_depletion
#             df_plot = pd.DataFrame({
#                 "wavenumber": wavenumber,
#                 "-ln(depletion)": noise_data
#             })
#             fig_noise = px.line(df_plot, x="wavenumber", y="-ln(depletion)",
#                                 title=f"Zoom and Select Noise Region ({file_name})")
#             st.plotly_chart(fig_noise, use_container_width=True)
#             noise_start = st.number_input("Noise Region Start (cm⁻¹)", value=float(wavenumber.iloc[0]))
#             noise_end = st.number_input("Noise Region End (cm⁻¹)", value=float(wavenumber.iloc[-1]))
#             if st.button("Set Noise Region"):
#                 wn = wavenumber.values if isinstance(wavenumber, pd.Series) else np.array(wavenumber)
#                 y_signal = noise_data.values if isinstance(noise_data, pd.Series) else np.array(noise_data)
#                 noise_indices = np.where((wn >= noise_start) & (wn <= noise_end))[0]
#                 if len(noise_indices) == 0:
#                     st.error("No data points found in the specified noise region. Please adjust your values.")
#                 else:
#                     noise_region_manual = y_signal[noise_indices]
#                     manual_noise_level = np.std(noise_region_manual)
#                     st.success(f"Manual Noise Level Set: {manual_noise_level:.4f}")
#                     st.write("Noise region spans from", noise_start, "to", noise_end)
#                     st.session_state.manual_noise = {"start": noise_start, "end": noise_end, "noise_level": manual_noise_level}

#         # Tab 4: Peak Analysis and Gaussian Fitting
#         with tabs[3]:
#             st.header("Peak Analysis and Gaussian Fitting")
#             # [Your depletion-specific peak analysis code here...]
#             st.info("Peak analysis code runs only for depletion data.")

#         # Tab 5: Smoothing Comparison
#         with tabs[4]:
#             st.header("Smoothing Comparison")
#         # Initialize comparison pairs if not already done
#         if "comparison_pairs" not in st.session_state:
#             st.session_state.comparison_pairs = []

#         st.subheader("Add Smoothing Method & Window Size Pair")
#         with st.form("comparison_pair_form", clear_on_submit=True):
#             comp_method = st.selectbox("Select Smoothing Method", 
#                                        options=["Moving Average", "Exponential Moving Average", "Savitzky–Golay", "Gaussian"])
#             comp_window_size = st.number_input("Window Size", min_value=1, max_value=21, value=3, step=1)
#             submitted = st.form_submit_button("Add Pair")
#             if submitted:
#                 st.session_state.comparison_pairs.append({"method": comp_method, "window_size": comp_window_size})
#                 st.success(f"Added pair: {comp_method} with window size {comp_window_size}")

#         st.subheader("Current Comparison Pairs")
#         if st.session_state.comparison_pairs:
#             for i, pair in enumerate(st.session_state.comparison_pairs):
#                 st.write(f"Pair {i+1}: {pair['method']} with window size {pair['window_size']}")
#         else:
#             st.info("No comparison pairs added yet.")

#         include_original = st.checkbox("Include Original Data", value=True)
        
#         if st.button("compareeee"):
#             if not st.session_state.comparison_pairs and not include_original:
#                 st.error("Please add at least one comparison pair or select to include the original data.")
#             else:
#                 file_name, wavenumber, depletion, _, _, _ = st.session_state.combined_plot_data[0]
#                 # Create static matplotlib figure
#                 fig_comp, ax_comp = plt.subplots(figsize=(16, 6))
#                 color_idx = 0
#                 if include_original:
#                     ax_comp.plot(wavenumber, depletion, linewidth=2, label="Original", color=get_color(color_idx))
#                     color_idx += 1
#                 for pair in st.session_state.comparison_pairs:
#                     data_to_plot = smooth_data(depletion, pair["window_size"], pair["method"])
#                     label = f"{pair['method']} (ws={pair['window_size']})"
#                     ax_comp.plot(wavenumber, data_to_plot, linewidth=2, label=label, color=get_color(color_idx))
#                     color_idx += 1
#                 ax_comp.set_xlabel("Wavenumber (cm⁻¹)")
#                 ax_comp.set_ylabel("Depletion")
#                 ax_comp.set_title(f"Smoothing Comparison on {file_name}")
#                 ax_comp.legend()
#                 ax_comp.grid(True)
#                 st.pyplot(fig_comp)
                
#                 # Prepare static plot download as PNG
#                 buf = io.BytesIO()
#                 fig_comp.savefig(buf, format="png")
#                 buf.seek(0)
#                 st.download_button(label="Download Static Plot (PNG)",
#                                    data=buf,
#                                    file_name="smoothing_comparison.png",
#                                    mime="image/png")
                
#                 # Create interactive Plotly figure using the same color assignments
#                 fig_plotly = go.Figure()
#                 color_idx = 0
#                 if include_original:
#                     fig_plotly.add_trace(go.Scatter(x=wavenumber, y=depletion,
#                                                     mode="lines", name="Original",
#                                                     line=dict(color=get_color(color_idx))))
#                     color_idx += 1
#                 for pair in st.session_state.comparison_pairs:
#                     data_to_plot = smooth_data(depletion, pair["window_size"], pair["method"])
#                     fig_plotly.add_trace(go.Scatter(x=wavenumber, y=data_to_plot,
#                                                     mode="lines",
#                                                     name=f"{pair['method']} (ws={pair['window_size']})",
#                                                     line=dict(color=get_color(color_idx))))
#                     color_idx += 1
#                 fig_plotly.update_layout(title=f"Smoothing Comparison on {file_name}",
#                                          xaxis_title="Wavenumber (cm⁻¹)",
#                                          yaxis_title="Depletion")
#                 st.plotly_chart(fig_plotly, use_container_width=True)
                
#                 # Prepare interactive plot download as HTML
#                 html_bytes = fig_plotly.to_html(full_html=True).encode("utf-8")
#                 st.download_button(label="Download Interactive Plot (HTML)",
#                                    data=html_bytes,
#                                    file_name="smoothing_comparison.html",
#                                    mime="text/html")
#     else:
#         # For Ion Yield, you may choose to hide or simplify additional analyses
#         st.info("Additional analysis tabs (Noise Region, Peak Analysis, Smoothing Comparison) are available only for Depletion data.")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndimage  # For Gaussian smoothing
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go
import io

# ----- Import Color Palette from Wes Anderson ----- #
from palettable.wesanderson import Moonrise5_6, Royal1_4
color_palette = Moonrise5_6.hex_colors + Royal1_4.hex_colors

def get_color(idx):
    """
    Return a color from the palette based on index.

    Args:
        idx (int): Index to select the color.

    Returns:
        str: Hex code for the color.
    """
    return color_palette[idx % len(color_palette)]

@st.cache_data
def read_csv_file(uploaded_file):
    """
    Read CSV file using pandas and return a DataFrame.

    Args:
        uploaded_file (UploadedFile): The file uploaded via Streamlit.

    Returns:
        pd.DataFrame: Data read from the CSV file.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return None
    return df

@st.cache_data
def smooth_series(series, window_size=3, method="Moving Average"):
    """
    Smooth the data series using the specified method.

    Args:
        series (pd.Series): The data series to smooth.
        window_size (int): Smoothing window size.
        method (str): Smoothing method (Moving Average, Exponential Moving Average, Gaussian, Savitzky–Golay, or None).

    Returns:
        pd.Series: The smoothed series.
    """
    if method == "Moving Average":
        return series.rolling(window=window_size, center=True).mean()
    elif method == "Exponential Moving Average":
        return series.ewm(span=window_size, adjust=False).mean()
    elif method == "Gaussian":
        smoothed_array = ndimage.gaussian_filter1d(series.values, sigma=window_size)
        return pd.Series(smoothed_array, index=series.index)
    elif method == "Savitzky–Golay":
        if window_size % 2 == 0:
            window_size += 1
        polyorder = 2 if window_size > 2 else 1
        smoothed_array = savgol_filter(series.values, window_length=window_size, polyorder=polyorder)
        return pd.Series(smoothed_array, index=series.index)
    else:
        return series

def gaussian(x, A, mu, sigma):
    """
    Gaussian function for curve fitting.

    Args:
        x (array-like): The independent variable.
        A (float): Amplitude.
        mu (float): Mean value.
        sigma (float): Standard deviation.

    Returns:
        array-like: Gaussian function values.
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def process_file(uploaded_file, data_type, smoothing_method, window_size, apply_smoothing):
    """
    Process an uploaded file and extract required data.

    Args:
        uploaded_file: The uploaded CSV file.
        data_type (str): "Depletion" or "Ion Yield".
        smoothing_method (str): Selected smoothing method.
        window_size (int): Window size for smoothing.
        apply_smoothing (bool): Whether to apply smoothing.

    Returns:
        dict or None: Dictionary containing processed data, or None if there is an error.
    """
    df = read_csv_file(uploaded_file)
    if df is None:
        return None

    file_name = uploaded_file.name
    result = {"file_name": file_name}

    if data_type == "Depletion":
        required_columns = ['wavenumber', 'depletion', '-ln(depletion)']
        if not all(col in df.columns for col in required_columns):
            st.error(f"File {file_name} is missing required columns: {required_columns}. Please check the file format.")
            return None
        result["wavenumber"] = df['wavenumber']
        result["original"] = df['depletion']
        result["ln_data"] = df['-ln(depletion)']
        if apply_smoothing:
            result["smoothed"] = smooth_series(df['depletion'], window_size, smoothing_method)
            result["smoothed_ln"] = smooth_series(df['-ln(depletion)'], window_size, smoothing_method)
        else:
            result["smoothed"] = df['depletion']
            result["smoothed_ln"] = df['-ln(depletion)']
    elif data_type == "Ion Yield":
        if 'ion_yield' not in df.columns:
            st.error(f"File {file_name} does not contain the required 'ion_yield' column. Please check the file.")
            return None
        result["wavenumber"] = df['wavenumber']
        result["original"] = df['ion_yield']
        if apply_smoothing:
            result["smoothed"] = smooth_series(df['ion_yield'], window_size, smoothing_method)
        else:
            result["smoothed"] = df['ion_yield']
    else:
        st.error("Invalid data type selected.")
        return None
    return result

def plot_individual(file_data, data_type, show_original):
    """
    Generate individual plots for a file.

    Args:
        file_data (dict): Dictionary with processed file data.
        data_type (str): "Depletion" or "Ion Yield".
        show_original (bool): Whether to overlay original data.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    file_name = file_data["file_name"]
    wavenumber = file_data["wavenumber"]
    fig = None

    if data_type == "Depletion":
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        if show_original:
            axs[0].plot(wavenumber, file_data["original"], linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
        axs[0].plot(wavenumber, file_data["smoothed"], linewidth=2, label="Smoothed", color=get_color(1))
        axs[0].set_xlabel("Wavenumber (cm⁻¹)")
        axs[0].set_ylabel("Depletion")
        axs[0].set_title(f"{file_name}: Depletion vs Wavenumber")
        axs[0].legend()
        axs[0].grid(True)
        
        if show_original:
            axs[1].plot(wavenumber, file_data["ln_data"], linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
        axs[1].plot(wavenumber, file_data["smoothed_ln"], linewidth=2, label="Smoothed", color=get_color(1))
        axs[1].set_xlabel("Wavenumber (cm⁻¹)")
        axs[1].set_ylabel("-ln(Depletion)")
        axs[1].set_title(f"{file_name}: -ln(Depletion) vs Wavenumber")
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()
    elif data_type == "Ion Yield":
        fig, ax = plt.subplots(figsize=(12, 6))
        if show_original:
            ax.plot(wavenumber, file_data["original"], linestyle='--', alpha=0.7, label="Noisy", color=get_color(0))
        ax.plot(wavenumber, file_data["smoothed"], linewidth=2, label="Smoothed", color=get_color(1))
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Ion Yield")
        ax.set_title(f"{file_name}: Ion Yield vs Wavenumber")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
    return fig

def plot_combined(data_list, data_type, show_original, peak_input, peak_labels, combined_width, combined_height):
    """
    Generate a combined plot for multiple files over predefined wavenumber ranges.

    Args:
        data_list (list): List of processed file data dictionaries.
        data_type (str): "Depletion" or "Ion Yield".
        show_original (bool): Whether to show original data.
        peak_input (str): Comma-separated peak wavenumbers (for depletion data).
        peak_labels (str): Comma-separated labels for peaks.
        combined_width (int): Plot width.
        combined_height (int): Plot height (per segment).

    Returns:
        matplotlib.figure.Figure: The combined plot figure.
    """
    ranges = [(400, 1200), (1200, 2000), (2700, 3200)]
    fig, axs = plt.subplots(3, 1, figsize=(combined_width, combined_height * 3))
    for ax, (low, high) in zip(axs, ranges):
        for idx, file_data in enumerate(data_list):
            wavenumber = file_data["wavenumber"]
            mask = (wavenumber >= low) & (wavenumber <= high)
            if mask.sum() > 0:
                if data_type == "Depletion":
                    if show_original:
                        ax.plot(wavenumber[mask], file_data["ln_data"][mask], linestyle='--', alpha=0.5,
                                label=f"{file_data['file_name']} Noisy", color=get_color(2*idx))
                    ax.plot(wavenumber[mask], file_data["smoothed_ln"][mask], linewidth=2,
                            label=f"{file_data['file_name']} Smoothed", color=get_color(2*idx+1))
                    ax.set_ylabel("-ln(Depletion)")
                elif data_type == "Ion Yield":
                    if show_original:
                        ax.plot(wavenumber[mask], file_data["original"][mask], linestyle='--', alpha=0.5,
                                label=f"{file_data['file_name']} Noisy", color=get_color(2*idx))
                    ax.plot(wavenumber[mask], file_data["smoothed"][mask], linewidth=2,
                            label=f"{file_data['file_name']} Smoothed", color=get_color(2*idx+1))
                    ax.set_ylabel("Ion Yield")
        if peak_input and peak_labels and data_type == "Depletion":
            try:
                peak_values = [float(val.strip()) for val in peak_input.split(',') if val.strip()]
                peak_names = [name.strip() for name in peak_labels.split(',') if name.strip()]
                if len(peak_values) != len(peak_names):
                    st.error("The number of peak values and labels must be equal! Please ensure they match.")
                else:
                    y_max = ax.get_ylim()[1]
                    for p, name in zip(peak_values, peak_names):
                        if low <= p <= high:
                            ax.axvline(x=p, color=get_color(0), linestyle='--', linewidth=1)
                            ax.text(p, y_max, name, rotation=70, verticalalignment='bottom',
                                    color=get_color(0), fontsize=14)
            except Exception as e:
                st.error(f"Error processing peak annotations: {e}")
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_xlim(low, high)
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    return fig


"""
Main function to run the Depletion Data Analysis App.
"""
st.title("Depletion Data Analysis App")
st.markdown("""
This app processes IR-UV ion dip spectroscopy data by smoothing, plotting, estimating noise, detecting peaks, and fitting Gaussian profiles.
""")

# ----- Sidebar Configuration with Expanders and Tooltips ----- #
with st.sidebar:
    with st.expander("File Upload", expanded=True):
        st.write("Upload one or more CSV files containing the required data columns.")
        data_type = st.radio("Select Data Type", options=["Depletion", "Ion Yield"],
                            help="Choose 'Depletion' for depletion data or 'Ion Yield' for ion yield data.")
        uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True,
                                        help="Select CSV files to analyze.")

    with st.expander("Smoothing Options", expanded=True):
        smoothing_method = st.selectbox("Smoothing Method", 
                                        options=["Moving Average", "Exponential Moving Average", "Savitzky–Golay", "Gaussian", "None"],
                                        help="Select the method to smooth the data. Choose 'None' to disable smoothing.")
        apply_smoothing = smoothing_method != "None"
        if apply_smoothing:
            window_size = st.slider("Smoothing Window Size", min_value=1, max_value=21, value=3, step=1,
                                    help="Adjust the window size for smoothing the data.")
        else:
            window_size = 1

    with st.expander("Output Options", expanded=True):
        show_original = st.checkbox("Include Original Data in Plot", value=True,
                                    help="Check to overlay the original (noisy) data on the plots.")
        peak_input = st.text_input("Enter Peak Wavenumbers (comma-separated)", value="",
                                help="For Depletion data, enter the wavenumber values for peak annotations.")
        peak_labels = st.text_input("Enter Corresponding Peak Labels (comma-separated)", value="",
                                    help="Enter labels for the peaks in the same order as the wavenumber values.")
        output_directory = st.text_input("Output Directory (default: current folder)", value=os.getcwd(),
                                        help="Specify a directory to save generated plots (optional).")
    process_button = st.button("Process Files", help="Click to process the uploaded files and generate plots.")

# ----- File Processing ----- #
processed_data = []
if process_button:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one file.")
    else:
        for file in uploaded_files:
            file_data = process_file(file, data_type, smoothing_method, window_size, apply_smoothing)
            if file_data is not None:
                processed_data.append(file_data)
        if processed_data:
            st.session_state.processed_data = processed_data
            st.success("Files processed successfully. Use the tabs below for further analysis.")

# ----- Tabs for Analysis ----- #
if "processed_data" in st.session_state and st.session_state.processed_data:
    tabs = st.tabs(["Individual Plots", "Combined Plot", "Noise Region", "Peak Analysis", "Smoothing Comparison"])
    
    # Tab 1: Individual Plots
    with tabs[0]:
        st.header("Individual File Plots")
        st.info("This tab shows individual plots for each file. You can view both the original and smoothed data.")
        for file_data in st.session_state.processed_data:
            fig = plot_individual(file_data, data_type, show_original)
            st.pyplot(fig)
    
    # Tab 2: Combined Plot
    with tabs[1]:
        st.header("Combined Plot")
        st.info("This tab combines data from multiple files over defined wavenumber ranges. Hover over the plot for more details.")
        combined_width = st.number_input("Combined Plot Width", value=28, min_value=1,
                                        help="Set the width of the combined plot.")
        combined_height = st.number_input("Combined Plot Height", value=8, min_value=1,
                                        help="Set the height of each segment in the combined plot.")
        fig_combined = plot_combined(st.session_state.processed_data, data_type, show_original, peak_input, peak_labels, combined_width, combined_height)
        st.pyplot(fig_combined)
    
    # Tab 3: Noise Region Selection (only for Depletion data)
    if data_type == "Depletion":
        with tabs[2]:
            st.header("Manual Noise Region Selection")
            st.info("Select a noise region from the data to estimate the noise level. Use the interactive plot below to zoom and select.")
            file_data = st.session_state.processed_data[0]
            noise_data_choice = st.selectbox("Data for Noise Analysis", options=["Smoothed", "Original"],
                                            help="Choose the data type for noise analysis.")
            noise_data = file_data["smoothed_ln"] if noise_data_choice == "Smoothed" else file_data["ln_data"]
            df_plot = pd.DataFrame({"wavenumber": file_data["wavenumber"], "-ln(depletion)": noise_data})
            fig_noise = px.line(df_plot, x="wavenumber", y="-ln(depletion)",
                                title=f"Zoom and Select Noise Region ({file_data['file_name']})")
            st.plotly_chart(fig_noise, use_container_width=True)
            noise_start = st.number_input("Noise Region Start (cm⁻¹)", value=float(file_data["wavenumber"].iloc[0]),
                                        help="Enter the start wavenumber for the noise region.")
            noise_end = st.number_input("Noise Region End (cm⁻¹)", value=float(file_data["wavenumber"].iloc[-1]),
                                        help="Enter the end wavenumber for the noise region.")
            if st.button("Set Noise Region"):
                wn = file_data["wavenumber"].values if isinstance(file_data["wavenumber"], pd.Series) else np.array(file_data["wavenumber"])
                y_signal = noise_data.values if isinstance(noise_data, pd.Series) else np.array(noise_data)
                noise_indices = np.where((wn >= noise_start) & (wn <= noise_end))[0]
                if len(noise_indices) == 0:
                    st.error("No data points found in the specified noise region. Please adjust your values.")
                else:
                    noise_region = y_signal[noise_indices]
                    noise_level = np.std(noise_region)
                    st.success(f"Manual Noise Level Set: {noise_level:.4f}")
                    st.write("Noise region spans from", noise_start, "to", noise_end)
                    st.session_state.manual_noise = {"start": noise_start, "end": noise_end, "noise_level": noise_level}
    
    # Tab 4: Peak Analysis and Gaussian Fitting (only for Depletion data)
    if data_type == "Depletion":
        with tabs[3]:
            st.header("Peak Analysis and Gaussian Fitting")
            st.info("This tab performs peak detection and fits Gaussian curves to the detected peaks. Detailed results will be shown below.")
            # Peak analysis code can be implemented here.
            st.info("Peak analysis functionality is under development.")
    
    # Tab 5: Smoothing Comparison (only for Depletion data)
    if data_type == "Depletion":
        with tabs[4]:
            st.header("Smoothing Comparison")
            st.info("Compare different smoothing methods and window sizes. Add pairs below and view the comparison plot.")
            if "comparison_pairs" not in st.session_state:
                st.session_state.comparison_pairs = []
            st.subheader("Add Smoothing Method & Window Size Pair")
            with st.form("comparison_pair_form", clear_on_submit=True):
                comp_method = st.selectbox("Select Smoothing Method", 
                                        options=["Moving Average", "Exponential Moving Average", "Savitzky–Golay", "Gaussian"],
                                        help="Choose a smoothing method for comparison.")
                comp_window_size = st.number_input("Window Size", min_value=1, max_value=21, value=3, step=1,
                                                help="Set the window size for the smoothing method.")
                submitted = st.form_submit_button("Add Pair")
                if submitted:
                    st.session_state.comparison_pairs.append({"method": comp_method, "window_size": comp_window_size})
                    st.success(f"Added pair: {comp_method} with window size {comp_window_size}")
            st.subheader("Current Comparison Pairs")
            if st.session_state.comparison_pairs:
                for i, pair in enumerate(st.session_state.comparison_pairs):
                    st.write(f"Pair {i+1}: {pair['method']} with window size {pair['window_size']}")
            else:
                st.info("No comparison pairs added yet.")
            include_original = st.checkbox("Include Original Data", value=True,
                                        help="Check to include the original data in the comparison plot.")
            if st.button("Compare"):
                if not st.session_state.comparison_pairs and not include_original:
                    st.error("Please add at least one comparison pair or select to include the original data.")
                else:
                    file_data = st.session_state.processed_data[0]
                    wavenumber = file_data["wavenumber"]
                    depletion = file_data["original"]
                    fig_comp, ax_comp = plt.subplots(figsize=(16, 6))
                    color_idx = 0
                    if include_original:
                        ax_comp.plot(wavenumber, depletion, linewidth=2, label="Original", color=get_color(color_idx))
                        color_idx += 1
                    for pair in st.session_state.comparison_pairs:
                        data_to_plot = smooth_series(file_data["original"], pair["window_size"], pair["method"])
                        label = f"{pair['method']} (ws={pair['window_size']})"
                        ax_comp.plot(wavenumber, data_to_plot, linewidth=2, label=label, color=get_color(color_idx))
                        color_idx += 1
                    ax_comp.set_xlabel("Wavenumber (cm⁻¹)")
                    ax_comp.set_ylabel("Depletion")
                    ax_comp.set_title(f"Smoothing Comparison on {file_data['file_name']}")
                    ax_comp.legend()
                    ax_comp.grid(True)
                    st.pyplot(fig_comp)
                    buf = io.BytesIO()
                    fig_comp.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button(label="Download Static Plot (PNG)",
                                    data=buf,
                                    file_name="smoothing_comparison.png",
                                    mime="image/png")
                    fig_plotly = go.Figure()
                    color_idx = 0
                    if include_original:
                        fig_plotly.add_trace(go.Scatter(x=wavenumber, y=depletion,
                                                        mode="lines", name="Original",
                                                        line=dict(color=get_color(color_idx))))
                        color_idx += 1
                    for pair in st.session_state.comparison_pairs:
                        data_to_plot = smooth_series(file_data["original"], pair["window_size"], pair["method"])
                        fig_plotly.add_trace(go.Scatter(x=wavenumber, y=data_to_plot,
                                                        mode="lines",
                                                        name=f"{pair['method']} (ws={pair['window_size']})",
                                                        line=dict(color=get_color(color_idx))))
                        color_idx += 1
                    fig_plotly.update_layout(title=f"Smoothing Comparison on {file_data['file_name']}",
                                            xaxis_title="Wavenumber (cm⁻¹)",
                                            yaxis_title="Depletion")
                    st.plotly_chart(fig_plotly, use_container_width=True)
                    html_bytes = fig_plotly.to_html(full_html=True).encode("utf-8")
                    st.download_button(label="Download Interactive Plot (HTML)",
                                    data=html_bytes,
                                    file_name="smoothing_comparison.html",
                                    mime="text/html")
    else:
        st.info("Additional analysis tabs (Noise Region, Peak Analysis, Smoothing Comparison) are available only for Depletion data.")


