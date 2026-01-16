# # # import streamlit as st
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # from packages.BaselineCorrection import mass_range, baseline

# # # # Import variables from session_state
# # # file_directory = st.session_state.get("file_directory", None)
# # # compiled_data = st.session_state.get("compiled_data", None)
# # # unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
# # # element1 = st.session_state.get("element1", None)
# # # element2 = st.session_state.get("element2", None)
# # # element3 = st.session_state.get("element3", None)
# # # mass_element1 = st.session_state.get("mass_element1", None)
# # # mass_element2 = st.session_state.get("mass_element2", None)
# # # mass_element3 = st.session_state.get("mass_element3", None)
# # # charge_state = st.session_state.get("charge_state", None)
# # # x_mass = st.session_state.get("x_mass", None)
# # # x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)

# # # col1, col2, col3 = st.columns([1, 0.1, 3])  # col2 is just spacing

# # # with col1:
# # #     st.markdown("#### Complex parameters")
# # #     st.session_state["n"] = st.number_input(f"Size of {element1}", value=st.session_state.get("n", 14))
# # #     st.session_state["m"] = st.number_input(f"Size of {element2}", value=st.session_state.get("m", 10))
# # #     st.session_state["o"] = st.number_input(f"Size of {element3}", value=st.session_state.get("o", 0))
    
# # #     st.markdown("#### Baseline parameters")
# # #     st.session_state["baseline_reference"] = float(
# # #         st.text_input("Start of baseline in amu", value=st.session_state.get("baseline_reference", 172.0))
# # #     )
# # #     st.session_state["baseline_width"] = float(
# # #         st.text_input("Width of baseline in amu", value=st.session_state.get("baseline_width", 4.0))
# # #     )
# # #     # Choose between "Mean Subtraction" and "arPLS"
# # #     st.session_state["baseline_method"] = st.selectbox(
# # #         "Select baseline correction method",
# # #         options=["Mean Subtraction", "arPLS"],
# # #         index=0
# # #     )

# # # with col3:
# # #     st.markdown("#### Plot parameters")
# # #     st.session_state["check_wavenumber"] = float(
# # #         st.text_input("Wavenumber to check plots", value=st.session_state.get("check_wavenumber", 1320))
# # #     )
# # #     st.session_state["y_max0"] = float(
# # #         st.text_input("Maximum y-value for top plot", value=st.session_state.get("y_max0", 0.05))
# # #     )
# # #     st.session_state["y_max1"] = float(
# # #         st.text_input("Maximum y-value for bottom plot", value=st.session_state.get("y_max1", 0.02))
# # #     )
# # #     st.session_state["plot_columnIndex_withoutIR"] = int(
# # #         st.number_input("Column index for signal without IR", value=st.session_state.get("plot_columnIndex_withoutIR", -2))
# # #     )
# # #     st.session_state["plot_columnIndex_withIR"] = int(
# # #         st.number_input("Column index for signal with IR", value=st.session_state.get("plot_columnIndex_withIR", -1))
# # #     )

# # # if st.button("✨ Register parameters and make plot!"):
# # #     # Retrieve parameters
# # #     element1 = st.session_state.get("element1", None)
# # #     element2 = st.session_state.get("element2", None)
# # #     element3 = st.session_state.get("element3", None)
# # #     mass_element1 = st.session_state.get("mass_element1", None)
# # #     mass_element2 = st.session_state.get("mass_element2", None)
# # #     mass_element3 = st.session_state.get("mass_element3", None)
# # #     n = st.session_state.get("n", None)
# # #     m = st.session_state.get("m", None)
# # #     o = st.session_state.get("o", None)
# # #     charge_state = st.session_state.get("charge_state", None)
# # #     baseline_reference = st.session_state.get("baseline_reference", None)
# # #     baseline_width = st.session_state.get("baseline_width", None)
# # #     check_wavenumber = st.session_state.get("check_wavenumber", None)
# # #     y_max0 = st.session_state.get("y_max0", None)
# # #     y_max1 = st.session_state.get("y_max1", None)
# # #     plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
# # #     plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
# # #     baseline_method = st.session_state.get("baseline_method", "Mean Subtraction")
    
# # #     # Get complex properties (for plotting purposes)
# # #     complex_name, mass_complex, mass_range_indices = mass_range(
# # #         n, m, o, element1, element2, element3,
# # #         mass_element1, mass_element2, mass_element3,
# # #         charge_state, x_mass
# # #     )
    
# # #     # Instantiate the baseline correction object
# # #     baseline_correction = baseline(
# # #         baseline_reference=baseline_reference,
# # #         interval=baseline_width,
# # #         wavenumber=check_wavenumber,
# # #         column_withoutIR=compiled_data[check_wavenumber].columns[plot_columnIndex_withoutIR],
# # #         column_withIR=compiled_data[check_wavenumber].columns[plot_columnIndex_withIR],
# # #         data_withoutIR=compiled_data[check_wavenumber].iloc[:, plot_columnIndex_withoutIR],
# # #         data_withIR=compiled_data[check_wavenumber].iloc[:, plot_columnIndex_withIR],
# # #         target_mass=x_mass,
# # #         method=baseline_method
# # #     )
    
# # #     # If using Mean Subtraction, compute baseline range & mean; for arPLS, skip these.
# # #     if baseline_method == "Mean Subtraction":
# # #         baseline_range_indices = baseline_correction.baseline_range()
# # #         baseline_correction.baseline_mean()
# # #     else:
# # #         # For arPLS, we can define the plotting range as the entire x_mass
# # #         baseline_range_indices = np.arange(len(x_mass))
    
# # #     # Perform baseline correction
# # #     baseline_corrected_data = baseline_correction.baseline_correction()
# # import streamlit as st
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from packages.BaselineCorrection import mass_range
# # from packages.BaselineCorrection import baseline_new  # Use Danial's new class

# # # ... (rest of your imports and session_state variables)

# # with col1:
# #     st.markdown("#### Complex parameters")
# #     st.session_state["n"] = st.number_input(f"Size of {element1}", value=st.session_state.get("n", 14))
# #     st.session_state["m"] = st.number_input(f"Size of {element2}", value=st.session_state.get("m", 10))
# #     st.session_state["o"] = st.number_input(f"Size of {element3}", value=st.session_state.get("o", 0))
    
# #     st.markdown("#### Baseline parameters")
# #     st.session_state["baseline_reference"] = float(
# #         st.text_input("Start of baseline in amu", value=st.session_state.get("baseline_reference", 172.0))
# #     )
# #     st.session_state["baseline_width"] = float(
# #         st.text_input("Width of baseline in amu", value=st.session_state.get("baseline_width", 4.0))
# #     )
# #     # Updated options: added "airPLS", "ASLS", "Rubberband"
# #     st.session_state["baseline_method"] = st.selectbox(
# #         "Select baseline correction method",
# #         options=["Mean Subtraction", "airPLS", "ASLS", "Rubberband"],
# #         index=0
# #     )

# # # ... (rest of your Streamlit code remains the same)

# # if st.button("✨ Register parameters and make plot!"):
# #     # Retrieve parameters (including baseline_method)
# #     baseline_method = st.session_state.get("baseline_method", "Mean Subtraction")
# #     # ... (other parameter retrieval)
    
# #     # Get complex properties (for plotting purposes)
# #     complex_name, mass_complex, mass_range_indices = mass_range(
# #         n, m, o, element1, element2, element3,
# #         mass_element1, mass_element2, mass_element3,
# #         charge_state, x_mass
# #     )
    
# #     # Instantiate the baseline correction object using the new class (Danial's code)
# #     baseline_correction = baseline_new(
# #         baseline_reference=baseline_reference,
# #         interval=baseline_width,
# #         wavenumber=check_wavenumber,
# #         column_withoutIR=compiled_data[check_wavenumber].columns[plot_columnIndex_withoutIR],
# #         column_withIR=compiled_data[check_wavenumber].columns[plot_columnIndex_withIR],
# #         data_withoutIR=compiled_data[check_wavenumber].iloc[:, plot_columnIndex_withoutIR],
# #         data_withIR=compiled_data[check_wavenumber].iloc[:, plot_columnIndex_withIR],
# #         target_mass=x_mass,
# #         method=baseline_method
# #     )
    
# #     # For Mean Subtraction, compute baseline range & mean; for pybaselines methods, use full range
# #     if baseline_method == "Mean Subtraction":
# #         baseline_range_indices = baseline_correction.baseline_range()
# #         baseline_correction.baseline_mean()
# #     else:
# #         baseline_range_indices = np.arange(len(x_mass))
    
# #     # Perform baseline correction
# #     baseline_corrected_data = baseline_correction.baseline_correction()
    

# #     # Save variables into session_state for later use
# #     st.session_state["complex"] = complex_name
# #     st.session_state["mass_complex"] = mass_complex
# #     st.session_state["mass_range_indices"] = mass_range_indices
# #     st.session_state["baseline_range_indices"] = baseline_range_indices

# #     # Plot raw data and baseline-corrected data
# #     fig, ax = plt.subplots(2, 1)
# #     ax[0].axvline(mass_complex, alpha=0.75, linestyle="solid", linewidth=1, color="green", label=complex_name)
# #     ax[0].plot(x_mass[mass_range_indices],
# #                compiled_data[check_wavenumber].iloc[mass_range_indices, plot_columnIndex_withoutIR],
# #                label=compiled_data[check_wavenumber].columns[plot_columnIndex_withoutIR])
# #     ax[0].plot(x_mass[mass_range_indices],
# #                compiled_data[check_wavenumber].iloc[mass_range_indices, plot_columnIndex_withIR],
# #                label=compiled_data[check_wavenumber].columns[plot_columnIndex_withIR])
# #     ax[0].fill_between(x_mass[baseline_range_indices], 0.1, color="lightgray", label="baseline range")
# #     ax[0].hlines(0, xmin=x_mass[mass_range_indices][0], xmax=x_mass[mass_range_indices][-1], color="lime")
# #     ax[0].legend(fontsize=5)
    
# #     ax[1].axvline(mass_complex, alpha=0.75, linestyle="solid", linewidth=1, color="green")
# #     ax[1].plot(x_mass[mass_range_indices],
# #                baseline_corrected_data.iloc[mass_range_indices, 0],
# #                label="baseline corrected signal without IR")
# #     ax[1].plot(x_mass[mass_range_indices],
# #                baseline_corrected_data.iloc[mass_range_indices, 1],
# #                label="baseline corrected signal with IR")
# #     ax[1].fill_between(x_mass[baseline_range_indices], 0.1, color="lightgray", label="baseline range")
# #     ax[1].hlines(0, xmin=x_mass[mass_range_indices][0], xmax=x_mass[mass_range_indices][-1], color="lime")
# #     ax[1].legend(fontsize=5)
    
# #     ax[0].set_ylim(-0.001, y_max0)
# #     ax[1].set_xlim(mass_complex-5, mass_complex+5)
# #     ax[1].set_ylim(-0.001, y_max1)
    
# #     st.pyplot(fig)
# #     plt.close(fig)
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from packages.BaselineCorrection import mass_range, baseline_new
from packages.utils import require_state

if not require_state(
    [
        "compiled_data",
        "unique_wavenumbers",
        "element1",
        "element2",
        "element3",
        "mass_element1",
        "mass_element2",
        "mass_element3",
        "charge_state",
        "x_mass",
    ],
    section="2.0 Baseline correction",
    hint="Import data (1.0) and register experiment parameters (1.3) before running this step.",
):
    st.stop()

# Import variables from session_state
file_directory = st.session_state.get("file_directory", None)
compiled_data = st.session_state.get("compiled_data", None)
unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
element1 = st.session_state.get("element1", None)
element2 = st.session_state.get("element2", None)
element3 = st.session_state.get("element3", None)
mass_element1 = st.session_state.get("mass_element1", None)
mass_element2 = st.session_state.get("mass_element2", None)
mass_element3 = st.session_state.get("mass_element3", None)
charge_state = st.session_state.get("charge_state", None)
x_mass = st.session_state.get("x_mass", None)
x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)

col1, col2, col3 = st.columns([1, 0.1, 3])  # col2 is spacing

with col1:
    st.markdown("#### Complex parameters")
    st.session_state["n"] = st.number_input(f"Size of {element1}", value=st.session_state.get("n", 14))
    st.session_state["m"] = st.number_input(f"Size of {element2}", value=st.session_state.get("m", 10))
    st.session_state["o"] = st.number_input(f"Size of {element3}", value=st.session_state.get("o", 0))
    
    st.markdown("#### Baseline parameters")
    st.session_state["baseline_reference"] = float(
        st.text_input("Start of baseline in amu", value=st.session_state.get("baseline_reference", 172.0))
    )
    st.session_state["baseline_width"] = float(
        st.text_input("Width of baseline in amu", value=st.session_state.get("baseline_width", 4.0))
    )
    st.session_state["baseline_method"] = st.selectbox(
        "Select baseline correction method",
        options=["Mean Subtraction","airpls", "airPLS", "ASLS", "Rubberband"],
        index=0
    )
    # Show additional parameter boxes based on the selected method
    if st.session_state["baseline_method"].lower() in ["airpls"]:
         with st.expander("airPLS parameters", expanded=True):
              st.session_state["airpls_lam"] = st.number_input("Lambda (lam) for airPLS", value=1e6)
    elif st.session_state["baseline_method"].lower() in ["arpls"]:
         with st.expander("airPLS parameters", expanded=True):
              st.session_state["arpls_lam"] = st.number_input("Lambda (lam) for arPLS", value=1e6)
    elif st.session_state["baseline_method"].upper() == "ASLS":
         with st.expander("ASLS parameters", expanded=True):
              st.session_state["asls_lam"] = st.number_input("Lambda for ASLS", value=1e7)
              st.session_state["asls_p"] = st.number_input("p for ASLS", value=0.02)

with col3:
    st.markdown("#### Plot parameters")
    available_wavenumbers = sorted(compiled_data.keys())
    default_idx = 0
    if "check_wavenumber" in st.session_state and st.session_state["check_wavenumber"] in available_wavenumbers:
        default_idx = available_wavenumbers.index(st.session_state["check_wavenumber"])
    st.session_state["check_wavenumber"] = st.selectbox(
        "Wavenumber to check plots",
        options=available_wavenumbers,
        index=default_idx,
        format_func=lambda x: f"{x:.2f}",
    )
    
    # Display count for selected wavenumber
    unique_wavenumbers_df = st.session_state.get("unique_wavenumbers_df")
    if unique_wavenumbers_df is not None:
        selected_wn = st.session_state["check_wavenumber"]
        matching_rows = unique_wavenumbers_df[unique_wavenumbers_df["Unique Wavenumbers"] == selected_wn]
        if not matching_rows.empty:
            count = matching_rows.iloc[0]["Counts"]
            st.info(f"📊 Count for wavenumber {selected_wn:.2f}: **{int(count)}**")
    st.session_state["y_max0"] = float(
        st.text_input("Maximum y-value for top plot", value=st.session_state.get("y_max0", 0.05))
    )
    st.session_state["y_max1"] = float(
        st.text_input("Maximum y-value for bottom plot", value=st.session_state.get("y_max1", 0.02))
    )
    st.session_state["plot_columnIndex_withoutIR"] = int(
        st.number_input("Column index for signal without IR", value=st.session_state.get("plot_columnIndex_withoutIR", -2))
    )
    st.session_state["plot_columnIndex_withIR"] = int(
        st.number_input("Column index for signal with IR", value=st.session_state.get("plot_columnIndex_withIR", -1))
    )

if st.button("✨ Register parameters and make plot!"):
    # Retrieve parameters
    element1 = st.session_state.get("element1", None)
    element2 = st.session_state.get("element2", None)
    element3 = st.session_state.get("element3", None)
    mass_element1 = st.session_state.get("mass_element1", None)
    mass_element2 = st.session_state.get("mass_element2", None)
    mass_element3 = st.session_state.get("mass_element3", None)
    n = st.session_state.get("n", None)
    m = st.session_state.get("m", None)
    o = st.session_state.get("o", None)
    charge_state = st.session_state.get("charge_state", None)
    baseline_reference = st.session_state.get("baseline_reference", None)
    baseline_width = st.session_state.get("baseline_width", None)
    check_wavenumber = st.session_state.get("check_wavenumber", None)
    y_max0 = st.session_state.get("y_max0", None)
    y_max1 = st.session_state.get("y_max1", None)
    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    baseline_method = st.session_state.get("baseline_method", "Mean Subtraction")
    key = round(check_wavenumber, 2)#this is for when no rounding is happening, remove when rounding

    
    # Get complex properties (for plotting purposes)
    complex_name, mass_complex, mass_range_indices = mass_range(
        n, m, o, element1, element2, element3,
        mass_element1, mass_element2, mass_element3,
        charge_state, x_mass
    )
    
    # Instantiate the baseline correction object using the new class, passing extra parameters
    baseline_correction = baseline_new(
        baseline_reference=baseline_reference,
        interval=baseline_width,
        # wavenumber=check_wavenumber,##this is for when you rounded the wavumnumbers
        wavenumber=key,#this is for when no rounding is happening
        column_withoutIR=compiled_data[check_wavenumber].columns[plot_columnIndex_withoutIR],
        column_withIR=compiled_data[check_wavenumber].columns[plot_columnIndex_withIR],
        data_withoutIR=compiled_data[check_wavenumber].iloc[:, plot_columnIndex_withoutIR],
        data_withIR=compiled_data[check_wavenumber].iloc[:, plot_columnIndex_withIR],
        target_mass=x_mass,
        method=baseline_method,
        airpls_lam=st.session_state.get("airpls_lam", 1e6),
        arpls_lam=st.session_state.get("arpls_lam", 1e6),
        asls_lam=st.session_state.get("asls_lam", 1e7),
        asls_p=st.session_state.get("asls_p", 0.02)
    )
    
    # For Mean Subtraction, compute baseline range & mean; for other methods, use full range
    if baseline_method == "Mean Subtraction":
        baseline_range_indices = baseline_correction.baseline_range()
        baseline_correction.baseline_mean()
    else:
        baseline_range_indices = np.arange(len(x_mass))
    
    # Perform baseline correction
    baseline_corrected_data = baseline_correction.baseline_correction()

    # Save variables into session_state for later use
    st.session_state["complex"] = complex_name
    st.session_state["mass_complex"] = mass_complex
    st.session_state["mass_range_indices"] = mass_range_indices
    st.session_state["baseline_range_indices"] = baseline_range_indices

    # Plot both raw and corrected spectra stacked vertically
    fig, (ax_raw, ax_corr) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Raw mass spectra
    ax_raw.axvline(
        mass_complex,
        alpha=0.75,
        linestyle="solid",
        linewidth=1,
        color="green",
        label=complex_name,
    )
    ax_raw.plot(
        x_mass[mass_range_indices],
        compiled_data[check_wavenumber].iloc[
            mass_range_indices, plot_columnIndex_withoutIR
        ],
        label=compiled_data[check_wavenumber].columns[
            plot_columnIndex_withoutIR
        ],
    )
    ax_raw.plot(
        x_mass[mass_range_indices],
        compiled_data[check_wavenumber].iloc[
            mass_range_indices, plot_columnIndex_withIR
        ],
        label=compiled_data[check_wavenumber].columns[
            plot_columnIndex_withIR
        ],
    )
    ax_raw.fill_between(
        x_mass[baseline_range_indices], 0.1, color="lightgray", label="baseline range"
    )
    ax_raw.hlines(
        0,
        xmin=x_mass[mass_range_indices][0],
        xmax=x_mass[mass_range_indices][-1],
        color="lime",
    )
    ax_raw.set_ylim(-0.001, y_max0)
    ax_raw.legend(fontsize=6)
    ax_raw.set_xlabel("Mass (amu)")
    ax_raw.set_ylabel("Intensity")
    ax_raw.set_title("📈 Raw mass spectra")
    
    # Baseline-corrected spectra
    ax_corr.axvline(
        mass_complex,
        alpha=0.75,
        linestyle="solid",
        linewidth=1,
        color="green",
        label=complex_name,
    )
    ax_corr.plot(
        x_mass[mass_range_indices],
        baseline_corrected_data.iloc[mass_range_indices, 0],
        label="baseline corrected signal without IR",
    )
    ax_corr.plot(
        x_mass[mass_range_indices],
        baseline_corrected_data.iloc[mass_range_indices, 1],
        label="baseline corrected signal with IR",
    )
    ax_corr.fill_between(
        x_mass[baseline_range_indices], 0.1, color="lightgray", label="baseline range"
    )
    ax_corr.hlines(
        0,
        xmin=x_mass[mass_range_indices][0],
        xmax=x_mass[mass_range_indices][-1],
        color="lime",
    )
    ax_corr.set_xlim(mass_complex - 5, mass_complex + 5)
    ax_corr.set_ylim(-0.001, y_max1)
    ax_corr.legend(fontsize=6)
    ax_corr.set_xlabel("Mass (amu)")
    ax_corr.set_ylabel("Intensity")
    ax_corr.set_title("✨ Baseline-corrected spectra")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# from packages.BaselineCorrection import mass_range, baseline_new

# # Import variables from session_state
# file_directory = st.session_state.get("file_directory", None)
# compiled_data = st.session_state.get("compiled_data", None)
# unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
# element1 = st.session_state.get("element1", None)
# element2 = st.session_state.get("element2", None)
# element3 = st.session_state.get("element3", None)
# mass_element1 = st.session_state.get("mass_element1", None)
# mass_element2 = st.session_state.get("mass_element2", None)
# mass_element3 = st.session_state.get("mass_element3", None)
# charge_state = st.session_state.get("charge_state", None)
# x_mass = st.session_state.get("x_mass", None)
# x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)

# col1, col2, col3 = st.columns([1, 0.1, 3])  # col2 is spacing

# with col1:
#     st.markdown("#### Complex parameters")
#     st.session_state["n"] = st.number_input(f"Size of {element1}", value=st.session_state.get("n", 14))
#     st.session_state["m"] = st.number_input(f"Size of {element2}", value=st.session_state.get("m", 10))
#     st.session_state["o"] = st.number_input(f"Size of {element3}", value=st.session_state.get("o", 0))
    
#     st.markdown("#### Baseline parameters")
#     st.session_state["baseline_reference"] = float(
#         st.text_input("Start of baseline in amu", value=st.session_state.get("baseline_reference", 172.0))
#     )
#     st.session_state["baseline_width"] = float(
#         st.text_input("Width of baseline in amu", value=st.session_state.get("baseline_width", 4.0))
#     )
#     st.session_state["baseline_method"] = st.selectbox(
#         "Select baseline correction method",
#         options=["Mean Subtraction", "airPLS","arPLS", "ASLS", "Rubberband"],
#         index=0
#     )
    
# # Only show plot parameter inputs when using Mean Subtraction
# if st.session_state["baseline_method"] == "Mean Subtraction":
#     with col3:
#         st.markdown("#### Plot parameters")
#         st.session_state["check_wavenumber"] = float(
#             st.text_input("Wavenumber to check plots", value=st.session_state.get("check_wavenumber", 1320))
#         )
#         st.session_state["y_max0"] = float(
#             st.text_input("Maximum y-value for top plot", value=st.session_state.get("y_max0", 0.05))
#         )
#         st.session_state["y_max1"] = float(
#             st.text_input("Maximum y-value for bottom plot", value=st.session_state.get("y_max1", 0.02))
#         )
#         st.session_state["plot_columnIndex_withoutIR"] = int(
#             st.number_input("Column index for signal without IR", value=st.session_state.get("plot_columnIndex_withoutIR", -2))
#         )
#         st.session_state["plot_columnIndex_withIR"] = int(
#             st.number_input("Column index for signal with IR", value=st.session_state.get("plot_columnIndex_withIR", -1))
#         )
# else:
#     # For other methods, you can either set default values or skip these inputs.
#     st.session_state["check_wavenumber"] = 0  # or some sensible default
#     st.session_state["y_max0"] = 0.05
#     st.session_state["y_max1"] = 0.02
#     st.session_state["plot_columnIndex_withoutIR"] = -2
#     st.session_state["plot_columnIndex_withIR"] = -1

# if st.button("✨ Register parameters and make plot!"):
#     # Retrieve parameters
#     element1 = st.session_state.get("element1", None)
#     element2 = st.session_state.get("element2", None)
#     element3 = st.session_state.get("element3", None)
#     mass_element1 = st.session_state.get("mass_element1", None)
#     mass_element2 = st.session_state.get("mass_element2", None)
#     mass_element3 = st.session_state.get("mass_element3", None)
#     n = st.session_state.get("n", None)
#     m = st.session_state.get("m", None)
#     o = st.session_state.get("o", None)
#     charge_state = st.session_state.get("charge_state", None)
#     baseline_reference = st.session_state.get("baseline_reference", None)
#     baseline_width = st.session_state.get("baseline_width", None)
#     baseline_method = st.session_state.get("baseline_method", "Mean Subtraction")
    
#     # For plotting with Mean Subtraction, use check plot parameters; for others, use default key
#     if baseline_method == "Mean Subtraction":
#         check_wavenumber = st.session_state.get("check_wavenumber", None)
#     else:
#         # If not Mean Subtraction, choose a default or derive key as needed.
#         check_wavenumber = list(compiled_data.keys())[0]
    
#     # Round the check key to ensure consistency
#     key = round(check_wavenumber, 2)
    
#     # Get complex properties (for plotting purposes)
#     complex_name, mass_complex, mass_range_indices = mass_range(
#         n, m, o, element1, element2, element3,
#         mass_element1, mass_element2, mass_element3,
#         charge_state, x_mass
#     )
    
#     # Instantiate the baseline correction object using the new class
#     baseline_correction = baseline_new(
#         baseline_reference=baseline_reference,
#         interval=baseline_width,
#         wavenumber=key,
#         column_withoutIR=compiled_data[check_wavenumber].columns[st.session_state["plot_columnIndex_withoutIR"]],
#         column_withIR=compiled_data[check_wavenumber].columns[st.session_state["plot_columnIndex_withIR"]],
#         data_withoutIR=compiled_data[check_wavenumber].iloc[:, st.session_state["plot_columnIndex_withoutIR"]],
#         data_withIR=compiled_data[check_wavenumber].iloc[:, st.session_state["plot_columnIndex_withIR"]],
#         target_mass=x_mass,
#         method=baseline_method,
#         airpls_lam=st.session_state.get("airpls_lam", 100),
#         arpls_lam=st.session_state.get("arpls_lam", 1e6),
#         asls_lam=st.session_state.get("asls_lam", 1e7),
#         asls_p=st.session_state.get("asls_p", 0.02)
#     )
    
#     if baseline_method == "Mean Subtraction":
#         baseline_range_indices = baseline_correction.baseline_range()
#         baseline_correction.baseline_mean()
#     else:
#         baseline_range_indices = np.arange(len(x_mass))
    
#     baseline_corrected_data = baseline_correction.baseline_correction()
    
#     st.session_state["complex"] = complex_name
#     st.session_state["mass_complex"] = mass_complex
#     st.session_state["mass_range_indices"] = mass_range_indices
#     st.session_state["baseline_range_indices"] = baseline_range_indices

#     # Plotting code (only using the plot parameters if applicable)
#     fig, ax = plt.subplots(2, 1)
#     ax[0].axvline(mass_complex, alpha=0.75, linestyle="solid", linewidth=1,
#                   color="green", label=complex_name)
#     ax[0].plot(x_mass[mass_range_indices],
#                compiled_data[check_wavenumber].iloc[mass_range_indices, st.session_state["plot_columnIndex_withoutIR"]],
#                label=compiled_data[check_wavenumber].columns[st.session_state["plot_columnIndex_withoutIR"]])
#     ax[0].plot(x_mass[mass_range_indices],
#                compiled_data[check_wavenumber].iloc[mass_range_indices, st.session_state["plot_columnIndex_withIR"]],
#                label=compiled_data[check_wavenumber].columns[st.session_state["plot_columnIndex_withIR"]])
#     ax[0].fill_between(x_mass[baseline_range_indices], 0.1, color="lightgray", label="baseline range")
#     ax[0].hlines(0, xmin=x_mass[mass_range_indices][0], xmax=x_mass[mass_range_indices][-1], color="lime")
#     ax[0].legend(fontsize=5)
    
#     ax[1].axvline(mass_complex, alpha=0.75, linestyle="solid", linewidth=1, color="green")
#     ax[1].plot(x_mass[mass_range_indices],
#                baseline_corrected_data.iloc[mass_range_indices, 0],
#                label="baseline corrected signal without IR")
#     ax[1].plot(x_mass[mass_range_indices],
#                baseline_corrected_data.iloc[mass_range_indices, 1],
#                label="baseline corrected signal with IR")
#     ax[1].fill_between(x_mass[baseline_range_indices], 0.1, color="lightgray", label="baseline range")
#     ax[1].hlines(0, xmin=x_mass[mass_range_indices][0], xmax=x_mass[mass_range_indices][-1], color="lime")
#     ax[1].legend(fontsize=5)
    
#     st.pyplot(fig)
#     plt.close(fig)
