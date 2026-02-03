# import streamlit as st
# import numpy as np
# import pandas as pd
# from packages.BaselineCorrection import *
# from packages.DepletionCalculator import *
# import matplotlib.pyplot as plt
# import plotly.graph_objs as go
# import os

# # Import variables
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
# baseline_reference = st.session_state.get("baseline_reference", None)
# baseline_width = st.session_state.get("baseline_width", None)
# complex = st.session_state.get("complex", None)
# mass_complex = st.session_state.get("mass_complex", None)
# mass_range_indices = st.session_state.get("mass_range_indices", None)
# baseline_range_indices = st.session_state.get("baseline_range_indices",None)
# plot_wavenumber = st.session_state.get("plot_wavenumber", None)
# plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
# plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
# compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)


# col1,col2,col3,col4,col5,col6 = st.columns([1,1,0.1,1,1,1]) # col 3 for spacing

# with col1:
#     # Integration parameters
#     st.markdown("#### Integration parameters")

#     # st.session_state["mass_peaks"] = st.radio("Enter mass peaks, comma separated", options = ["Average mass","Custom input"], index=0)
#     options = ["Average mass", "Custom input"]
#     selected_option = st.session_state.get("mass_peaks", options[0])  # Default to the first option if not set
#     selected_index = options.index(selected_option)  # Get the index of the selected option

#     st.session_state["mass_peaks"] = st.radio("Enter mass peaks, comma separated", options=options, index=selected_index)
#     mass_peaks = st.session_state.get("mass_peaks", None)

#     if mass_peaks == "Average mass":
        
#         mass_input = st.text_input("Enter mass peaks, comma separated", value = (mass_complex), label_visibility="collapsed")
#         list_mass_isotope = [float(mass_input)]
#         st.session_state["list_mass_isotope"] = list_mass_isotope
#     else:
#         mass_input = st.text_input("Enter mass peaks, comma separated", value = (", ".join(str(x) for x in st.session_state.get("list_mass_isotope", None))), label_visibility="collapsed")
#         list_mass_isotope = [float(x.strip()) for x in mass_input.split(",")]
#         st.session_state["list_mass_isotope"] = list_mass_isotope
  
#     # save input history
#     if 'input_history' not in st.session_state:
#         st.session_state['input_history'] = []  # List to store input history
#     if list_mass_isotope not in st.session_state["input_history"]:
#         st.session_state["input_history"].append(list_mass_isotope)
#     if len(st.session_state['input_history']) > 5:  # Keep last 5 entries
#         st.session_state['input_history'].pop(0)

#     with st.popover("Input history"):
#         input_history = st.session_state.get("input_history", None)
#         for item in input_history:
#             st.code(", ".join(str(x) for x in item))


# with col2:
#     st.markdown("#### ")
#     # st.markdown("#### Active mass peak(s):")
#     st.write("Active mass peak(s):")
#     st.code(", ".join(str(x) for x in list_mass_isotope))
#     st.session_state["isotope_scan_width"] = float(st.text_input("Integration width per peak in amu", value = st.session_state.get("isotope_scan_width", 0.3)))
#     st.session_state["save_output"] = st.toggle("Save output", value= st.session_state.get("save_output", True))
    

# with col4:
#     # Plot parameters
#     st.markdown("#### Plot parameters")
#     st.session_state["plot_columnIndex_withoutIR"] = int(st.number_input("Column index for signal without IR irradiation", value = st.session_state.get("plot_columnIndex_withoutIR", -2)))
#     st.session_state["plot_columnIndex_withIR"] = int(st.number_input("Column index for signal with IR irradiation", value = st.session_state.get("plot_columnIndex_withIR", -1)))
#     st.session_state["plot_wavenumber"] = float(st.text_input("Wavenumber to check plots", value = st.session_state.get("plot_wavenumber",400.0)))
    

# with col5:
#     st.markdown("#### ")
#     st.session_state["mass_xmin"] = float(st.text_input("Mass Spectra: minimum x-value", value = st.session_state.get("mass_xmin", 0.0)))
#     st.session_state["mass_xmax"] = float(st.text_input("Mass Spectra: maximum x-value", value = st.session_state.get("mass_xmax", 1300)))
#     st.session_state["mass_ymax"] = float(st.text_input("Mass Spectra: maximum y-value", value = st.session_state.get("mass_ymax",0.1)))
    

# with col6:
#     st.markdown("#### ")
#     st.session_state["depletion_ymin"] = float(st.text_input("Depletion: minimum y-value", st.session_state.get("depletion_ymin", -0.1)))
#     st.session_state["depletion_ymax"] = float(st.text_input("Depletion: maximum y-value", st.session_state.get("depletion_ymax", 1.5)))
#     st.session_state["ln_depletion_ymin"] = float(st.text_input("-ln(depletion): minimum y-value", st.session_state.get("ln_depletion_ymin", -0.4)))
#     st.session_state["ln_depletion_ymax"] = float(st.text_input("-ln(depletion): maximum y-value", st.session_state.get("ln_depletion_ymax", 0.3)))

# if st.button("✨ Analyze!"):

#     # Initialize variables
#     list_mass_isotope = st.session_state.get("list_mass_isotope")
#     isotope_scan_width = st.session_state.get("isotope_scan_width", None)
#     x_mass = st.session_state.get("x_mass", None)
#     compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
#     plot_wavenumber = st.session_state.get("plot_wavenumber", None)
#     plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
#     plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
#     mass_xmin = st.session_state.get("mass_xmin", None)
#     mass_xmax = st.session_state.get("mass_xmax", None)
#     mass_ymax = st.session_state.get("mass_ymax", None)
#     depletion_ymin = st.session_state.get("depletion_ymin", None)
#     depletion_ymax = st.session_state.get("depletion_ymax", None)
#     ln_depletion_ymin = st.session_state.get("ln_depletion_ymin", None)
#     ln_depletion_ymax = st.session_state.get("ln_depletion_ymax", None)
#     save_output = st.session_state.get("save_output")

    
#     # Calculate depletioin
#     # initialize class
#     fullrange_depletion_spectra_multi_peak = depletion(mass_complex = list_mass_isotope, scan_width = isotope_scan_width, target_mass=x_mass)
    
#     for wavenumber in unique_wavenumbers:
#         # print(wavenumber)
#         fullrange_depletion_spectra_multi_peak.wavenumber = wavenumber
#         fullrange_depletion_spectra_multi_peak.column_withoutIR = compilation_baseline_corrected_data[wavenumber].columns[-2]
#         fullrange_depletion_spectra_multi_peak.column_withIR = compilation_baseline_corrected_data[wavenumber].columns[-1]
#         fullrange_depletion_spectra_multi_peak.data_withoutIR = compilation_baseline_corrected_data[wavenumber].iloc[:,-2]
#         fullrange_depletion_spectra_multi_peak.data_withIR = compilation_baseline_corrected_data[wavenumber].iloc[:,-1]

#         fullrange_depletion_data = fullrange_depletion_spectra_multi_peak.make_depletion_spectra_multi_peak()


#     list_mass_isotope = fullrange_depletion_spectra_multi_peak.list_mass_isotope
#     list_scanwidth_isotope = fullrange_depletion_spectra_multi_peak.list_scanwidth_isotope

#     # convert to numpy array for easy plotting.
#     data = np.array(fullrange_depletion_data)

#     if save_output:
#         fullrange_depletion_data.to_csv(f"{file_directory}/output/fullrange_depletion_data_{complex}_{int(data[-1,0])}-{int(data[0,0])}.csv", index=False)
#         st.write(f"Output saved @ {file_directory}/output/fullrange_depletion_data_{complex}_{int(data[-1,0])}-{int(data[0,0])}.csv")
#     else:
#         st.write("Note: save output is currently off.")



#     tab1, tab2, tab3= st.tabs(["📈 Mass spectra - specified wavenumber", "💥 Depletion - full range", "💥 -ln(depletion) - full range"])

#     with tab1:

#         st.markdown("###### *:green[Interactive plot with plotly]*")
#         # Plot via plotly - interactive
#         fig = go.Figure()

#         # Plot vertical line at mass_complex
#         fig.add_trace(go.Scatter(
#             x=[mass_complex, mass_complex],
#             y=[-0.001, mass_ymax],
#             mode='lines',
#             line=dict(color="green", width=2, dash="solid"),
#             name=str(complex)+" average mass"
#         ))

#  # Add isotope peaks and scanwidths
#         for index, isotope in enumerate(list_mass_isotope):
#             # Add vertical line for each isotope
#             fig.add_trace(go.Scatter(
#                 x=[isotope, isotope],
#                 y=[0, mass_ymax],  # Adjust the y range according to your data
#                 mode='lines',
#                 line=dict(color='black', width=2),
#                 showlegend=False
#             ))
            
#             # Fill the area for scan widths
#             fig.add_shape(
#                 type="rect",
#                 x0 = isotope - isotope_scan_width,
#                 x1 = isotope + isotope_scan_width,
#                 y0 = 0,
#                 y1 = mass_ymax,
#                 fillcolor='lightgray',
#                 line=dict(color='rgba(0,0,0,0)'),
#                 opacity=0.4,
#                 name='Fill',
#                 layer = "below"
#             )

#         fig.add_trace(go.Scatter(
#             x=[0,0],  # Use None for x to avoid displaying a line
#             y=[0,0],  # Use None for y to avoid displaying a line
#             mode='lines',  # Mode can be anything; it won't show
#             name='Isotope peaks',  # This will appear in the legend
#             line=dict(color='black', width=2)  # Invisible line
#         ))

#         fig.add_trace(go.Scatter(
#             x=[0,0],  # Use None for x to avoid displaying a line
#             y=[0,0],  # Use None for y to avoid displaying a line
#             mode='lines',  # Mode can be anything; it won't show
#             name='scan width range',  # This will appear in the legend
#             line=dict(color='lightgray', width=3)  # Invisible line
#         ))



#         # Plot data for "without IR" signal
#         fig.add_trace(go.Scatter(
#             x=x_mass[:],
#             y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withoutIR],
#             mode='lines',
#             name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withoutIR],
#             line=dict(color= "#1f77b4"),
#             opacity=1
#         ))
        
#         # Plot data for "with IR" signal
#         fig.add_trace(go.Scatter(
#             x=x_mass[:],
#             y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withIR],
#             mode='lines',
#             name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withIR],
#             line=dict(color="#ff7f0e"),
#             opacity=1
#         ))

       

#         # Add rectangle region for baseline range
#         fig.add_shape(
#             type="rect",
#             x0=min(x_mass[baseline_range_indices]),  # Starting x coordinate
#             y0=0,                                    # Starting y coordinate
#             x1=max(x_mass[baseline_range_indices]),  # Ending x coordinate
#             y1=mass_ymax,                               # Ending y coordinate
#             fillcolor="lightsteelblue",                   # Fill color
#             line=dict(color='rgba(0,0,0,0)'),            # Border color
#             opacity = 0.4,
#             layer = "below"
#         )
        
#         # Add an invisible scatter trace to represent the rectangle in the legend
#         fig.add_trace(go.Scatter(
#             x=[0,0],  # Use None for x to avoid displaying a line
#             y=[0,0],  # Use None for y to avoid displaying a line
#             mode='lines',  # Mode can be anything; it won't show
#             name='Baseline Range',  # This will appear in the legend
#             line=dict(color='lightsteelblue', width=2)  # Invisible line
#         ))

#         # Plot horizontal line at y=0
#         fig.add_trace(go.Scatter(
#             x=[x_mass[mass_range_indices][0], x_mass[mass_range_indices][-1]],
#             y=[0, 0],
#             mode='lines',
#             line=dict(color="lime", width=1),
#             name='zero Line'
#         ))

#         # Update layout for the plot
#         fig.update_layout(
#             yaxis=dict(range=[-0.001, mass_ymax]),
#             xaxis_title="Mass (amu)",
#             yaxis_title="Intensity",
#             xaxis=dict(range=[mass_xmin, mass_xmax]),
#             title=complex,
#             legend=dict(x=0.8, y=0.9)
#         )

#         # Display plot in Streamlit
#         st.plotly_chart(fig)
#         if save_output:
#             # Save interactive plot as PNG (requires kaleido)
#             fig.write_image(f"{file_directory}/output/mass_spectra_{complex}_{int(data[-1,0])}-{int(data[0,0])}.png")

#         # matplotlib static plot
#         st.markdown("###### *:green[Static plot with matplotlib]*")
#         fig, ax = plt.subplots(figsize=(5, 3))

#         ax.axvline(mass_complex,alpha=0.75,linestyle="solid",linewidth=2, color="green", label= f"{complex} average mass")
        
#         # multi peak version
#         for index, isotope in enumerate(list_mass_isotope):
#             plt.axvline(isotope,alpha=0.75,linestyle="solid",linewidth=1, color="black")
#             plt.fill_between(x_mass[list_scanwidth_isotope[index]],0.5, color = "lightgray")

#         ax.axvline(0,0, color='black', label='Isotope peaks')
#         ax.axvline(0,0, color='lightgray', label='scan width range')
        
#         ax.plot(x_mass[:],compilation_baseline_corrected_data[plot_wavenumber].iloc[:,plot_columnIndex_withoutIR], label=f"{compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withoutIR]}")
#         ax.plot(x_mass[:],compilation_baseline_corrected_data[plot_wavenumber].iloc[:,plot_columnIndex_withIR], label = f"{compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withIR]}")
        
#         ax.fill_between(x_mass[baseline_range_indices],0.2, color = "lightsteelblue", label = "baseline range")
#         ax.hlines(0,xmin = x_mass[mass_range_indices][0], xmax =x_mass[mass_range_indices][-1], color="lime")

#         ax.set_xlim(mass_xmin, mass_xmax)
#         ax.set_ylim(-0.001, mass_ymax)
#         ax.set_xlabel("Mass (amu)")
#         ax.set_ylabel("Intensity")
#         ax.legend(fontsize=5)
#         fig.tight_layout()

#         st.pyplot(fig)
#         if save_output:
#         # Save static plot using matplotlib
#          fig.savefig(f"{file_directory}/output/mass_spectra_{complex}_{int(data[-1,0])}-{int(data[0,0])}.png", dpi=300)
        

    

#     with tab2:
#         # convert to numpy array for easy plotting.
#         data = np.array(fullrange_depletion_data)

#         # Interactive plot - plotly
#         st.markdown("###### *:green[Interactive plot with plotly]*")
        
#         fig = go.Figure()

#         fig.add_trace(go.Scatter(
#             x = data[:,0],
#             y = data[:,3],
#             name = fullrange_depletion_data.columns[3],
#             line=dict(color= "#1f77b4")
#         ))

#         # Plot horizontal line at y=0
#         fig.add_trace(go.Scatter(
#             x=[data[0,0],data[-1,0]],
#             y=[0, 0],
#             mode='lines',
#             line=dict(color="lime", width=1),
#             name = "zero line"
#         ))

#         # Update layout for the plot
#         fig.update_layout(
#             yaxis=dict(range=[depletion_ymin, depletion_ymax]),
#             xaxis_title="wavenumber (cm-1)",
#             yaxis_title="Intensity",
#             title= fullrange_depletion_data.columns[3] + " " + complex,
#             legend=dict(x=0.8, y=0.9)
#         )

#         st.plotly_chart(fig)
#         if save_output:
#             fig.write_image(f"{file_directory}/output/depletion_{complex}_{int(data[-1,0])}-{int(data[0,0])}.png")


#         # Static plot - matplotlib
#         st.markdown("###### *:green[Static plot with matplotlib]*")

#         fig, ax = plt.subplots(figsize=(8, 3))

#         ax.plot(data[:,0],data[:,3])
#         ax.scatter(data[:,0],data[:,3])
#         ax.legend([fullrange_depletion_data.columns[3]], fontsize=5,  loc = "upper right")
#         ax.hlines(0,xmin = data[:,0][0], xmax =data[:,0][-1], color="lime")

#         mass_label = [round(item,2) for item in list_mass_isotope]
#         textstr = f"Complex: {complex} \nMass peaks: {mass_label} amu \nIntegration width = {isotope_scan_width} amu\nBaseline reference: {baseline_reference} amu \nBaseline width = {baseline_width} amu"
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

#         # Place a text box in the plot
#         ax.text(0.6, 0.25, textstr, transform=plt.gca().transAxes, fontsize=5,
#                 verticalalignment='top', bbox=props)

#         ax.set_ylim(depletion_ymin,depletion_ymax)
#         ax.set_xlim(1200,2000)
#         ax.set_xlabel("wavenumber (cm-1)")

#         st.pyplot(fig)
#         if save_output:
#             fig.savefig(f"{file_directory}/output/depletion_{complex}_{int(data[-1,0])}-{int(data[0,0])}.png", dpi=300)

#     with tab3:

#         # Interactive plot - plotly
#         st.markdown("###### *:green[Interactive plot with plotly]*")
        
#         fig = go.Figure()

#         fig.add_trace(go.Scatter(
#             x = data[:,0],
#             y = data[:,4],
#             name = fullrange_depletion_data.columns[4],
#             line=dict(color= "#1f77b4")
#         ))

#         # Plot horizontal line at y=0
#         fig.add_trace(go.Scatter(
#             x=[data[0,0],data[-1,0]],
#             y=[0, 0],
#             mode='lines',
#             line=dict(color="lime", width=1),
#             name = "zero line"
#         ))

#         # Update layout for the plot
#         fig.update_layout(
#             yaxis=dict(range=[ln_depletion_ymin, ln_depletion_ymax]),
#             xaxis_title="wavenumber (cm-1)",
#             yaxis_title="Intensity",
#             title= fullrange_depletion_data.columns[4] + " " + complex,
#             legend=dict(x=0.8, y=0.9)
#         )

#         st.plotly_chart(fig)
#         if save_output:
#             fig.write_image(f"{file_directory}/output/ln_depletion_{complex}_{int(data[-1,0])}-{int(data[0,0])}.png")



#         # Static plot - matplotlib
#         st.markdown("###### *:green[Static plot with matplotlib]*")

#         fig, ax = plt.subplots(figsize=(8, 3))

#         ax.plot(data[:,0],data[:,4])
#         ax.scatter(data[:,0],data[:,4])
#         ax.legend([fullrange_depletion_data.columns[4]], fontsize=5, loc = "lower right")
#         ax.hlines(0,xmin = data[:,0][0], xmax =data[:,0][-1], color="lime")

#         mass_label = [round(item,2) for item in list_mass_isotope]
#         textstr = f"Complex: {complex} \nMass peaks: {mass_label} amu \nIntegration width = {isotope_scan_width} amu\nBaseline reference: {baseline_reference} amu \nBaseline width = {baseline_width} amu"
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

#         # Place a text box in the plot
#         ax.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=5,
#                 verticalalignment='top', bbox=props)

#         ax.set_ylim(ln_depletion_ymin,ln_depletion_ymax)
#         ax.set_xlim(1200,2000)
#         ax.set_xlabel("wavenumber (cm-1)")

#         st.pyplot(fig)
#         if save_output:
#             fig.savefig(f"{file_directory}/output/ln_depletion_{complex}_{int(data[-1,0])}-{int(data[0,0])}.png", dpi=300)

#     st.markdown(f"#### Full range depletion data {complex}:")
#     st.table(fullrange_depletion_data)
# #The following CODE WAS EDITED ON 26 MArch and works properly, however it's commneted out becuase I wanted to test an additionalla tab with the PAH database module
# import streamlit as st
# import numpy as np
# import pandas as pd
# from packages.BaselineCorrection import *
# from packages.DepletionCalculator import *
# import matplotlib.pyplot as plt
# import plotly.graph_objs as go
# import os
# from scipy.signal import savgol_filter  # Reuse your existing smoothing function

# # Import variables from session state
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
# baseline_reference = st.session_state.get("baseline_reference", None)
# baseline_width = st.session_state.get("baseline_width", None)
# complex = st.session_state.get("complex", None)
# mass_complex = st.session_state.get("mass_complex", None)
# mass_range_indices = st.session_state.get("mass_range_indices", None)
# baseline_range_indices = st.session_state.get("baseline_range_indices", None)
# plot_wavenumber = st.session_state.get("plot_wavenumber", None)
# plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
# plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
# compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)


# # Layout for input parameters (6 columns)
# col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 0.1, 1, 1, 1])

# # --- Integration parameters ---
# with col1:
#     st.markdown("#### Integration parameters")
#     options = ["Average mass", "Custom input"]
#     selected_option = st.session_state.get("mass_peaks", options[0])
#     selected_index = options.index(selected_option)
#     st.session_state["mass_peaks"] = st.radio("Enter mass peaks, comma separated", options=options, index=selected_index)
#     mass_peaks = st.session_state.get("mass_peaks", None)
    
#     if mass_peaks == "Average mass":
#         mass_input = st.text_input("Enter mass peaks, comma separated", value=(mass_complex), label_visibility="collapsed")
#         list_mass_isotope = [float(mass_input)]
#         st.session_state["list_mass_isotope"] = list_mass_isotope
#     else:
#         mass_input = st.text_input("Enter mass peaks, comma separated",
#                                    value=(", ".join(str(x) for x in st.session_state.get("list_mass_isotope", None))),
#                                    label_visibility="collapsed")
#         list_mass_isotope = [float(x.strip()) for x in mass_input.split(",")]
#         st.session_state["list_mass_isotope"] = list_mass_isotope

#     # Save input history
#     if 'input_history' not in st.session_state:
#         st.session_state['input_history'] = []
#     if list_mass_isotope not in st.session_state["input_history"]:
#         st.session_state["input_history"].append(list_mass_isotope)
#     if len(st.session_state['input_history']) > 5:
#         st.session_state['input_history'].pop(0)
    
#     with st.popover("Input history"):
#         input_history = st.session_state.get("input_history", None)
#         for item in input_history:
#             st.code(", ".join(str(x) for x in item))

# # --- Other parameters ---
# with col2:
#     st.markdown("#### ")
#     st.write("Active mass peak(s):")
#     st.code(", ".join(str(x) for x in list_mass_isotope))
#     st.session_state["isotope_scan_width"] = float(st.text_input("Integration width per peak in amu",
#                                                                  value=st.session_state.get("isotope_scan_width", 0.3)))
#     st.session_state["save_output"] = st.toggle("Save output", value=st.session_state.get("save_output", True))

# with col4:
#     st.markdown("#### Plot parameters")
#     st.session_state["plot_columnIndex_withoutIR"] = int(st.number_input("Column index for signal without IR irradiation",
#                                                                          value=st.session_state.get("plot_columnIndex_withoutIR", -2)))
#     st.session_state["plot_columnIndex_withIR"] = int(st.number_input("Column index for signal with IR irradiation",
#                                                                       value=st.session_state.get("plot_columnIndex_withIR", -1)))
#     st.session_state["plot_wavenumber"] = float(st.text_input("Wavenumber to check plots",
#                                                                value=st.session_state.get("plot_wavenumber", 400.0)))

# with col5:
#     st.markdown("#### Mass Spectra parameters")
#     st.session_state["mass_xmin"] = float(st.text_input("Mass Spectra: minimum x-value",
#                                                      value=st.session_state.get("mass_xmin", 0.0)))
#     st.session_state["mass_xmax"] = float(st.text_input("Mass Spectra: maximum x-value",
#                                                      value=st.session_state.get("mass_xmax", 1300)))
#     st.session_state["mass_ymax"] = float(st.text_input("Mass Spectra: maximum y-value",
#                                                      value=st.session_state.get("mass_ymax", 0.1)))

# with col6:
#     st.markdown("#### Depletion parameters")
#     # These inputs will later be updated with computed defaults if not manually changed.
#     st.session_state["depletion_xmin"] = float(st.text_input("Depletion: minimum wavenumber",
#                                                              value=st.session_state.get("depletion_xmin", 0.0)))
#     st.session_state["depletion_xmax"] = float(st.text_input("Depletion: maximum wavenumber",
#                                                              value=st.session_state.get("depletion_xmax", 2000)))
#     st.session_state["depletion_ymin"] = float(st.text_input("Depletion: minimum y-value",
#                                                              value=st.session_state.get("depletion_ymin", -0.1)))
#     st.session_state["depletion_ymax"] = float(st.text_input("Depletion: maximum y-value",
#                                                              value=st.session_state.get("depletion_ymax", 1.5)))
#     st.session_state["ln_depletion_ymin"] = float(st.text_input("-ln(depletion): minimum y-value",
#                                                                value=st.session_state.get("ln_depletion_ymin", -0.4)))
#     st.session_state["ln_depletion_ymax"] = float(st.text_input("-ln(depletion): maximum y-value",
#                                                                value=st.session_state.get("ln_depletion_ymax", 0.3)))
#     # --- NEW: Smoothing options (only for the depletion module) ---
#     data_display_option = st.radio("Display Data", options=["Original", "Smoothed"],
#                                    index=0,
#                                    help="Select whether to display the original or Savitzky–Golay smoothed depletion data")
#     st.session_state["data_display_option"] = data_display_option
#     if data_display_option == "Smoothed":
#         # Use a slider with odd numbers only (step=2, default 9)
#         smoothing_window = st.slider("Savitzky–Golay Window Size", min_value=3, max_value=21,
#                                      value=9, step=2,
#                                      help="Window size for smoothing (must be odd)")
#         st.session_state["smoothing_window"] = smoothing_window
# # --- When analysis button is pressed ---
# if st.button("✨ Analyze!"):
#     # Retrieve variables from session state
#     list_mass_isotope = st.session_state.get("list_mass_isotope")
#     isotope_scan_width = st.session_state.get("isotope_scan_width", None)
#     x_mass = st.session_state.get("x_mass", None)
#     compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
#     plot_wavenumber = st.session_state.get("plot_wavenumber", None)
#     plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
#     plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
#     mass_xmin = st.session_state.get("mass_xmin", None)
#     mass_xmax = st.session_state.get("mass_xmax", None)
#     mass_ymax = st.session_state.get("mass_ymax", None)
#     save_output = st.session_state.get("save_output")
    
#     # Retrieve depletion parameters (initially from user inputs)
#     depletion_ymin = st.session_state.get("depletion_ymin", None)
#     depletion_ymax = st.session_state.get("depletion_ymax", None)
#     ln_depletion_ymin = st.session_state.get("ln_depletion_ymin", None)
#     ln_depletion_ymax = st.session_state.get("ln_depletion_ymax", None)
#     depletion_xmin = st.session_state.get("depletion_xmin")
#     depletion_xmax = st.session_state.get("depletion_xmax")
    
#     # Calculate depletion using your class/function
#     fullrange_depletion_spectra_multi_peak = depletion(mass_complex=list_mass_isotope, scan_width=isotope_scan_width, target_mass=x_mass)
    
#     for wavenumber in unique_wavenumbers:
#         fullrange_depletion_spectra_multi_peak.wavenumber = wavenumber
#         fullrange_depletion_spectra_multi_peak.column_withoutIR = compilation_baseline_corrected_data[wavenumber].columns[-2]
#         fullrange_depletion_spectra_multi_peak.column_withIR = compilation_baseline_corrected_data[wavenumber].columns[-1]
#         fullrange_depletion_spectra_multi_peak.data_withoutIR = compilation_baseline_corrected_data[wavenumber].iloc[:, -2]
#         fullrange_depletion_spectra_multi_peak.data_withIR = compilation_baseline_corrected_data[wavenumber].iloc[:, -1]
#         fullrange_depletion_data = fullrange_depletion_spectra_multi_peak.make_depletion_spectra_multi_peak()
    
#     list_mass_isotope = fullrange_depletion_spectra_multi_peak.list_mass_isotope
#     list_scanwidth_isotope = fullrange_depletion_spectra_multi_peak.list_scanwidth_isotope
    
#     # Convert to numpy array for plotting
#     data = np.array(fullrange_depletion_data)
#     # --- Apply smoothing if the user selected "Smoothed" ---
#     if st.session_state.get("data_display_option") == "Smoothed":
#         # Define a helper function to apply Savitzky–Golay filtering
#         def apply_savgol(series, window_size):
#             if window_size % 2 == 0:
#                 window_size += 1  # ensure window is odd
#             polyorder = 2 if window_size > 2 else 1
#             return savgol_filter(series, window_length=window_size, polyorder=polyorder)
#         # Smooth the depletion data (column 3) and ln(depletion) (column 4)
#         data[:, 3] = apply_savgol(data[:, 3], st.session_state.get("smoothing_window"))
#         data[:, 4] = apply_savgol(data[:, 4], st.session_state.get("smoothing_window"))
#     # --- Update depletion default parameters based on computed data ---
#     # For the depletion x-range (wavenumber), use the data's min and max (max + 1)
#     if st.session_state.get("depletion_xmin", 0.0) == +0.1:
#         st.session_state["depletion_xmin"] = float(np.min(data[:, 0]))
#     if st.session_state.get("depletion_xmax", 2000) == 2000:
#         st.session_state["depletion_xmax"] = float(np.max(data[:, 0]) + 1)
#     # For depletion y–range, use the min and (max + 0.1)
#     if st.session_state.get("depletion_ymin", -0.1) == -0.1:
#         st.session_state["depletion_ymin"] = float(np.min(data[:, 3]))
#     if st.session_state.get("depletion_ymax", 1.5) == 1.5:
#         st.session_state["depletion_ymax"] = float(np.max(data[:, 3]) + 0.1)
#     # For ln(depletion) y–range, use the min and (max + 0.1)
#     if st.session_state.get("ln_depletion_ymin", -0.4) == -0.4:
#         st.session_state["ln_depletion_ymin"] = float(np.min(data[:, 4]))
#     if st.session_state.get("ln_depletion_ymax", 0.3) == 0.3:
#         st.session_state["ln_depletion_ymax"] = float(np.max(data[:, 4]) + 0.1)
    
#     # Refresh local variables after potential update
#     depletion_xmin = st.session_state["depletion_xmin"]
#     depletion_xmax = st.session_state["depletion_xmax"]
#     depletion_ymin = st.session_state["depletion_ymin"]
#     depletion_ymax = st.session_state["depletion_ymax"]
#     ln_depletion_ymin = st.session_state["ln_depletion_ymin"]
#     ln_depletion_ymax = st.session_state["ln_depletion_ymax"]
    
#     # Save CSV output if enabled
#     if save_output:
#         csv_filename = f"{file_directory}/output/fullrange_depletion_data_{complex}_{int(data[-1,0])}-{int(data[0,0])}.csv"
#         fullrange_depletion_data.to_csv(csv_filename, index=False)
#         st.write(f"CSV output saved @ {csv_filename}")
#     else:
#         st.write("Note: save output is currently off.")
    
#     # Use the updated depletion x-range for the depletion plots
#     x_range_dep = [depletion_xmin, depletion_xmax]
    
#     tab1, tab2, tab3 = st.tabs(["📈 Mass spectra - specified wavenumber", "💥 Depletion - full range", "💥 -ln(depletion) - full range"])
    
#     # ------------------- Tab 1: Mass Spectra -------------------
#     with tab1:
#         st.markdown("###### *:green[Interactive plot with plotly]*")
#         fig = go.Figure()
#         # Plot average mass vertical line
#         fig.add_trace(go.Scatter(
#             x=[mass_complex, mass_complex],
#             y=[-0.001, mass_ymax],
#             mode='lines',
#             line=dict(color="green", width=2, dash="solid"),
#             name=f"{complex} average mass"
#         ))
#         # Plot isotope peaks and scan width rectangles
#         for index, isotope in enumerate(list_mass_isotope):
#             fig.add_trace(go.Scatter(
#                 x=[isotope, isotope],
#                 y=[0, mass_ymax],
#                 mode='lines',
#                 line=dict(color='black', width=2),
#                 showlegend=False
#             ))
#             fig.add_shape(
#                 type="rect",
#                 x0=isotope - isotope_scan_width,
#                 x1=isotope + isotope_scan_width,
#                 y0=0,
#                 y1=mass_ymax,
#                 fillcolor='lightgray',
#                 line=dict(color='rgba(0,0,0,0)'),
#                 opacity=0.4,
#                 name='Fill',
#                 layer="below"
#             )
#         # Invisible traces for legend
#         fig.add_trace(go.Scatter(
#             x=[0, 0],
#             y=[0, 0],
#             mode='lines',
#             name='Isotope peaks',
#             line=dict(color='black', width=2)
#         ))
#         fig.add_trace(go.Scatter(
#             x=[0, 0],
#             y=[0, 0],
#             mode='lines',
#             name='scan width range',
#             line=dict(color='lightgray', width=3)
#         ))
#         # Plot the signals (without and with IR)
#         fig.add_trace(go.Scatter(
#             x=x_mass[:],
#             y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withoutIR],
#             mode='lines',
#             name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withoutIR],
#             line=dict(color="#1f77b4"),
#             opacity=1
#         ))
#         fig.add_trace(go.Scatter(
#             x=x_mass[:],
#             y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withIR],
#             mode='lines',
#             name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withIR],
#             line=dict(color="#ff7f0e"),
#             opacity=1
#         ))
#         # Baseline range rectangle
#         fig.add_shape(
#             type="rect",
#             x0=min(x_mass[baseline_range_indices]),
#             y0=0,
#             x1=max(x_mass[baseline_range_indices]),
#             y1=mass_ymax,
#             fillcolor="lightsteelblue",
#             line=dict(color='rgba(0,0,0,0)'),
#             opacity=0.4,
#             layer="below"
#         )
#         fig.add_trace(go.Scatter(
#             x=[0, 0],
#             y=[0, 0],
#             mode='lines',
#             name='Baseline Range',
#             line=dict(color='lightsteelblue', width=2)
#         ))
#         # Zero line
#         fig.add_trace(go.Scatter(
#             x=[x_mass[mass_range_indices][0], x_mass[mass_range_indices][-1]],
#             y=[0, 0],
#             mode='lines',
#             line=dict(color="lime", width=1),
#             name='zero Line'
#         ))
#         fig.update_layout(
#             xaxis=dict(range=[mass_xmin, mass_xmax]),
#             yaxis=dict(range=[-0.001, mass_ymax]),
#             xaxis_title="Mass (amu)",
#             yaxis_title="Intensity",
#             title=complex,
#             legend=dict(x=0.8, y=0.9)
#         )
#         st.plotly_chart(fig)
#         if save_output:
#             html_filename = f"{file_directory}/output/mass_spectra_{complex}_{int(data[-1,0])}-{int(data[0,0])}.html"
#             fig.write_html(html_filename, include_plotlyjs='cdn')
#             st.write(f"Interactive Mass Spectra plot saved as HTML @ {html_filename}")
        
#         st.markdown("###### *:green[Static plot with matplotlib]*")
#         fig_static, ax = plt.subplots(figsize=(5, 3))
#         ax.axvline(mass_complex, alpha=0.75, linestyle="solid", linewidth=2, color="green", label=f"{complex} average mass")
#         for index, isotope in enumerate(list_mass_isotope):
#             ax.axvline(isotope, alpha=0.75, linestyle="solid", linewidth=1, color="black")
#             ax.fill_between(x_mass[list_scanwidth_isotope[index]], 0.5, color="lightgray")
#         ax.axvline(0, color='black', label='Isotope peaks')
#         ax.axvline(0, color='lightgray', label='scan width range')
#         ax.plot(x_mass[:], compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withoutIR],
#                 label=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withoutIR])
#         ax.plot(x_mass[:], compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withIR],
#                 label=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withIR])
#         ax.fill_between(x_mass[baseline_range_indices], 0.2, color="lightsteelblue", label="baseline range")
#         ax.hlines(0, xmin=x_mass[mass_range_indices][0], xmax=x_mass[mass_range_indices][-1], color="lime")
#         ax.set_xlim(mass_xmin, mass_xmax)
#         ax.set_ylim(-0.001, mass_ymax)
#         ax.set_xlabel("Mass (amu)")
#         ax.set_ylabel("Intensity")
#         ax.legend(fontsize=5)
#         fig_static.tight_layout()
#         st.pyplot(fig_static)
#         if save_output:
#             fig_static.savefig(f"{file_directory}/output/mass_spectra_{complex}_{int(data[-1,0])}-{int(data[0,0])}.png", dpi=300)
    
#     # ------------------- Tab 2: Depletion Plot -------------------
#     with tab2:
#         st.markdown("###### *:green[Interactive plot with plotly]*")
#         fig_dep = go.Figure()
#         fig_dep.add_trace(go.Scatter(
#             x=data[:, 0],
#             y=data[:, 3],
#             name=fullrange_depletion_data.columns[3],
#             line=dict(color="#1f77b4")
#         ))

#         zero_line_depletion = 1

#         fig_dep.add_trace(go.Scatter(
#             x=[data[0, 0], data[-1, 0]],
#             y=[zero_line_depletion, zero_line_depletion],
#             mode='lines',
#             line=dict(color="lime", width=1),
#             name="Zero Line"
#         ))

#         fig_dep.update_layout(
#             xaxis=dict(range=x_range_dep),
#             yaxis=dict(range=[depletion_ymin, depletion_ymax]),
#             xaxis_title="wavenumber (cm-1)",
#             yaxis_title="Intensity",
#             title=fullrange_depletion_data.columns[3] + " " + complex,
#             legend=dict(x=0.8, y=0.9)
#         )
#         st.plotly_chart(fig_dep)
#         if save_output:
#             html_filename = f"{file_directory}/output/depletion_{complex}_{int(data[-1,0])}-{int(data[0,0])}.html"
#             fig_dep.write_html(html_filename, include_plotlyjs='cdn')
#             st.write(f"Interactive Depletion plot saved as HTML @ {html_filename}")
        
#         st.markdown("###### *:green[Static plot with matplotlib]*")
#         fig_dep_static, ax = plt.subplots(figsize=(21, 6))
#         ax.plot(data[:, 0], data[:, 3])
#         ax.scatter(data[:, 0], data[:, 3])
#         ax.legend([fullrange_depletion_data.columns[3]], fontsize=5, loc="upper right")
#         ax.hlines(zero_line_depletion, xmin=depletion_xmin, xmax=depletion_xmax, color="lime")
#         mass_label = [round(item, 2) for item in list_mass_isotope]
#         textstr = (f"Complex: {complex} \nMass peaks: {mass_label} amu \nIntegration width = {isotope_scan_width} amu\n"
#                    f"Baseline reference: {baseline_reference} amu \nBaseline width = {baseline_width} amu")
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         ax.text(0.6, 0.25, textstr, transform=plt.gca().transAxes, fontsize=5,
#                 verticalalignment='top', bbox=props)
#         ax.set_ylim(depletion_ymin, depletion_ymax)
#         ax.set_xlim(depletion_xmin, depletion_xmax)
#         ax.set_xlabel("wavenumber (cm-1)")
#         st.pyplot(fig_dep_static)
#         if save_output:
#             fig_dep_static.savefig(f"{file_directory}/output/depletion_{complex}_{int(data[-1,0])}-{int(data[0,0])}.png", dpi=300)
    
#     # ------------------- Tab 3: ln(Depletion) Plot -------------------
#     with tab3:
#         st.markdown("###### *:green[Interactive plot with plotly]*")
#         fig_ln = go.Figure()
#         fig_ln.add_trace(go.Scatter(
#             x=data[:, 0],
#             y=data[:, 4],
#             name=fullrange_depletion_data.columns[4],
#             line=dict(color="#1f77b4")
#         ))
#         # zero_line_ln = (ln_depletion_ymin + ln_depletion_ymax) / 2
#         zero_line_ln = 0
#         fig_ln.add_trace(go.Scatter(
#             x=[data[0, 0], data[-1, 0]],
#             y=[zero_line_ln, zero_line_ln],
#             mode='lines',
#             line=dict(color="lime", width=1),
#             name="Zero Line"
#         ))

#         fig_ln.update_layout(
#             xaxis=dict(range=x_range_dep),
#             yaxis=dict(range=[ln_depletion_ymin, ln_depletion_ymax]),
#             xaxis_title="wavenumber (cm-1)",
#             yaxis_title="Intensity",
#             title=fullrange_depletion_data.columns[4] + " " + complex,
#             legend=dict(x=0.8, y=0.9)
#         )
#         st.plotly_chart(fig_ln)
#         if save_output:
#             html_filename = f"{file_directory}/output/ln_depletion_{complex}_{int(data[-1,0])}-{int(data[0,0])}.html"
#             fig_ln.write_html(html_filename, include_plotlyjs='cdn')
#             st.write(f"Interactive -ln(Depletion) plot saved as HTML @ {html_filename}")
        
#         st.markdown("###### *:green[Static plot with matplotlib]*")
#         fig_ln_static, ax = plt.subplots(figsize=(21, 6))
#         ax.plot(data[:, 0], data[:, 4])
#         ax.scatter(data[:, 0], data[:, 4])
#         ax.legend([fullrange_depletion_data.columns[4]], fontsize=5, loc="lower right")
#         ax.hlines(zero_line_ln, xmin=depletion_xmin, xmax=depletion_xmax, color="lime")
#         mass_label = [round(item, 2) for item in list_mass_isotope]
#         textstr = (f"Complex: {complex} \nMass peaks: {mass_label} amu \nIntegration width = {isotope_scan_width} amu\n"
#                    f"Baseline reference: {baseline_reference} amu \nBaseline width = {baseline_width} amu")
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         ax.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=5,
#                 verticalalignment='top', bbox=props)
#         ax.set_ylim(ln_depletion_ymin, ln_depletion_ymax)
#         ax.set_xlim(depletion_xmin, depletion_xmax)
#         ax.set_xlabel("wavenumber (cm-1)")
#         st.pyplot(fig_ln_static)
#         if save_output:
#             fig_ln_static.savefig(f"{file_directory}/output/ln_depletion_{complex}_{int(data[-1,0])}-{int(data[0,0])}.png", dpi=300)
    
#     st.markdown(f"#### Full range depletion data {complex}:")
#     st.table(fullrange_depletion_data)
#2. The following was the trial code for the PAH database implementation
import streamlit as st
import numpy as np
import pandas as pd
# from packages.DepletionCalculator import *
from packages.IR_yield_Calculator_v2 import *
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import configparser
import os
from scipy.signal import savgol_filter  # Reuse your existing smoothing function

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
            defaults["depletion_ymin"] = config.getfloat('Plot Parameters', 'depletion_ymin')
            defaults["depletion_ymax"] = config.getfloat('Plot Parameters', 'depletion_ymax')
            defaults["ln_depletion_ymin"] = config.getfloat('Plot Parameters', 'ln_depletion_ymin')
            defaults["ln_depletion_ymax"] = config.getfloat('Plot Parameters', 'ln_depletion_ymax')
            defaults["scan_width"] = config.getfloat('Integration Parameters', 'scan_width')
            defaults["save_output"] = config.getboolean('Integration Parameters', 'save_output')
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults
defaults = load_defaults()

# Import variables from session state
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
baseline_reference = st.session_state.get("baseline_reference", None)
baseline_width = st.session_state.get("baseline_width", None)
complex = st.session_state.get("complex", None)
mass_complex = st.session_state.get("mass_complex", None)
mass_range_indices = st.session_state.get("mass_range_indices", None)
baseline_range_indices = st.session_state.get("baseline_range_indices", None)
plot_wavenumber = st.session_state.get("plot_wavenumber", None)
plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)

st.title("3.0 · Depletion & Data Analysis")
st.caption("Review baseline-corrected spectra, compute depletion, and inspect full-range outputs.")
st.divider()

# Import variables from session state
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
baseline_reference = st.session_state.get("baseline_reference", None)
baseline_width = st.session_state.get("baseline_width", None)
complex = st.session_state.get("complex", None)
mass_complex = st.session_state.get("mass_complex", None)
mass_range_indices = st.session_state.get("mass_range_indices", None)
baseline_range_indices = st.session_state.get("baseline_range_indices", None)
plot_wavenumber = st.session_state.get("plot_wavenumber", None)
plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)

with st.expander("Step 1 · Integration parameters", expanded=True):
    col_integrate, col_history = st.columns([3, 1])
    with col_integrate:
        options = ["Average mass", "Custom input"]
        selected_option = st.session_state.get("mass_peaks", options[0])
        selected_index = options.index(selected_option)
        st.session_state["mass_peaks"] = st.radio(
            "Mass peak selection method",
            options=options,
            index=selected_index,
            horizontal=True,
            help="Use the average complex mass or provide custom isotope peaks.",
        )
        mass_peaks = st.session_state.get("mass_peaks", None)

        if mass_peaks == "Average mass":
            mass_input = st.text_input(
                "Active mass peak(s)",
                value=mass_complex,
                help="Enter a single mass or comma-separated list.",
            )
            list_mass_isotope = [float(mass_input)]
            st.session_state["list_mass_isotope"] = list_mass_isotope
        else:
            default_list = ", ".join(
                str(x) for x in st.session_state.get("list_mass_isotope", [])
            )
            mass_input = st.text_input(
                "Custom mass peaks (comma-separated)", value=default_list
            )
            list_mass_isotope = [float(x.strip()) for x in mass_input.split(",")]
            st.session_state["list_mass_isotope"] = list_mass_isotope

    with col_history:
        st.write("Recent selections")
        if "input_history" not in st.session_state:
            st.session_state["input_history"] = []
        if list_mass_isotope and list_mass_isotope not in st.session_state["input_history"]:
            st.session_state["input_history"].append(list_mass_isotope)
            if len(st.session_state["input_history"]) > 5:
                st.session_state["input_history"].pop(0)
        for item in st.session_state.get("input_history", []):
            st.code(", ".join(str(x) for x in item))

with st.expander("Step 2 · Mass & plot configuration", expanded=False):
    col_mass, col_plot = st.columns(2)
    with col_mass:
        st.write("Active peaks")
        st.code(", ".join(str(x) for x in list_mass_isotope))
        st.session_state["search_width"] = float(
            st.text_input(
                "Peak-finding width per peak (amu)",
                value=st.session_state.get("search_width", 0.5),
            )
        )
        st.session_state["isotope_scan_width"] = float(
            st.text_input(
                "Integration width per peak (amu)",
                value=st.session_state.get("isotope_scan_width", 0.3),
            )
        )
        st.session_state["save_output"] = st.toggle(
            "Save outputs to output directory",
            value=st.session_state.get("save_output", True),
        )

    with col_plot:
        st.session_state["plot_columnIndex_withoutIR"] = int(
            st.number_input(
                "Column index · without IR signal",
                value=st.session_state.get("plot_columnIndex_withoutIR", None),
            )
        )
        st.session_state["plot_columnIndex_withIR"] = int(
            st.number_input(
                "Column index · with IR signal",
                value=st.session_state.get("plot_columnIndex_withIR", None),
            )
        )
        # available_wavenumbers = sorted(compiled_data.keys()) if compiled_data else []
        # if available_wavenumbers:
        #     default_idx = 0
        #     prev_selection = st.session_state.get("plot_wavenumber")
        #     if prev_selection in available_wavenumbers:
        #         default_idx = available_wavenumbers.index(prev_selection)
        #     st.session_state["plot_wavenumber"] = st.selectbox(
        #         "Wavenumber to inspect",
        #         options=available_wavenumbers,
        #         index=default_idx,
        #         format_func=lambda x: f"{x:.2f}",
        #         help="Select from wavenumbers available in compiled data.",
        #     )
            
        #     # Display count for selected wavenumber
        #     unique_wavenumbers = st.session_state.get("unique_wavenumbers")
        #     if unique_wavenumbers is not None:
        #         selected_wn = st.session_state["plot_wavenumber"]
        #         matching_rows = unique_wavenumbers[unique_wavenumbers["Unique Wavenumbers"] == selected_wn]
        #         if not matching_rows.empty:
        #             count = matching_rows.iloc[0]["Counts"]
        #             st.info(f"📊 Count for wavenumber {selected_wn:.2f}: **{int(count)}**")
        # else:
        #     st.info(
        #         "No compiled wavenumbers available yet. Run earlier steps to populate data.",
        #         icon="ℹ️",
        #     )
        #     st.session_state["plot_wavenumber"] = st.session_state.get("plot_wavenumber", 0.0)

        col_axes = st.columns(3)
        with col_axes[0]:
            st.session_state["mass_xmin"] = float(
                st.text_input(
                    "Mass plot X min",
                    value=st.session_state.get("mass_xmin", None),
                )
            )
        with col_axes[1]:
            st.session_state["mass_xmax"] = float(
                st.text_input(
                    "Mass plot X max",
                    value=st.session_state.get("mass_xmax", None),
                )
            )
        with col_axes[2]:
            st.session_state["mass_ymax"] = float(
                st.text_input(
                    "Mass plot Y max",
                    value=st.session_state.get("mass_ymax", None),
                )
            )

with st.expander("Step 3 · Depletion window & smoothing", expanded=False):
    col_dep, col_ln = st.columns(2)
    with col_dep:
        st.session_state["depletion_xmin"] = float(
            st.text_input(
                "Depletion plot wavenumber min",
                value=st.session_state.get("depletion_xmin", 0.0),
            )
        )
        st.session_state["depletion_xmax"] = float(
            st.text_input(
                "Depletion plot wavenumber max",
                value=st.session_state.get("depletion_xmax", 2000),
            )
        )
        st.session_state["depletion_ymin"] = float(
            st.text_input(
                "Depletion y-axis min",
                value=st.session_state.get("depletion_ymin", defaults.get("depletion_ymin", None)),
            )
        )
        st.session_state["depletion_ymax"] = float(
            st.text_input(
                "Depletion y-axis max",
                value=st.session_state.get("depletion_ymax", defaults.get("depletion_ymax", None)),
            )
        )

    with col_ln:
        st.session_state["ln_depletion_ymin"] = float(
            st.text_input(
                "-ln(Depletion) y-axis min",
                value=st.session_state.get("ln_depletion_ymin", defaults.get("ln_depletion_ymin", None)),
            )
        )
        st.session_state["ln_depletion_ymax"] = float(
            st.text_input(
                "-ln(Depletion) y-axis max",
                value=st.session_state.get("ln_depletion_ymax", defaults.get("ln_depletion_ymax", None)),
            )
        )
        st.session_state["data_display_option"] = st.radio(
            "Display data",
            options=["Original", "Smoothed"],
            index=0,
            help="Apply Savitzky–Golay smoothing to depletion traces.",
        )
        if st.session_state["data_display_option"] == "Smoothed":
            st.session_state["smoothing_window"] = st.slider(
                "Smoothing window size",
                min_value=3,
                max_value=21,
                value=9,
                step=2,
                help="Window size must be odd.",
            )

st.divider()

action_cols = st.columns([1, 3])
with action_cols[0]:
    run_button = st.button("🚀 Analyze depletion", use_container_width=True)
with action_cols[1]:
    st.info(
        "Run analysis after updating inputs. Results include mass spectra, depletion plots, and download links.",
        icon="ℹ️",
    )

results_container = st.container()

# --- When analysis button is pressed ---
if run_button:
    # Retrieve variables from session state
    list_mass_isotope = st.session_state.get("list_mass_isotope")
    isotope_scan_width = st.session_state.get("isotope_scan_width", None)
    x_mass = st.session_state.get("x_mass", None)
    compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
    plot_wavenumber = st.session_state.get("plot_wavenumber", None)
    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    mass_xmin = st.session_state.get("mass_xmin", None)
    mass_xmax = st.session_state.get("mass_xmax", None)
    mass_ymax = st.session_state.get("mass_ymax", None)
    save_output = st.session_state.get("save_output")
    
    depletion_ymin = st.session_state.get("depletion_ymin", None)
    depletion_ymax = st.session_state.get("depletion_ymax", None)
    ln_depletion_ymin = st.session_state.get("ln_depletion_ymin", None)
    ln_depletion_ymax = st.session_state.get("ln_depletion_ymax", None)
    depletion_xmin = st.session_state.get("depletion_xmin")
    depletion_xmax = st.session_state.get("depletion_xmax")
    unique_wavenumbers = st.session_state.get("unique_wavenumbers")
    
    # Calculate depletion using your class/function
    fullrange_depletion_spectra_multi_peak = IR_yield_Calculator(
    mass_peaks=st.session_state["list_mass_isotope"],
    integration_width=st.session_state["isotope_scan_width"],
    search_width=st.session_state["search_width"],
    mass_range = st.session_state["x_mass"],
    )

    
    for wavenumber in unique_wavenumbers:
        fullrange_depletion_spectra_multi_peak.wavenumber = wavenumber
        # fullrange_depletion_spectra_multi_peak.wavenumber_counts= unique_wavenumbers.set_index("Unique Wavenumbers").at[wavenumber, "Counts"]
        fullrange_depletion_spectra_multi_peak.column_withoutIR = compilation_baseline_corrected_data[wavenumber].columns[-2]
        fullrange_depletion_spectra_multi_peak.column_withIR = compilation_baseline_corrected_data[wavenumber].columns[-1]
        fullrange_depletion_spectra_multi_peak.data_withoutIR = compilation_baseline_corrected_data[wavenumber].iloc[:, -2]
        fullrange_depletion_spectra_multi_peak.data_withIR = compilation_baseline_corrected_data[wavenumber].iloc[:, -1]
        fullrange_depletion_data = fullrange_depletion_spectra_multi_peak.make_IR_yield_spectra()
        orig_fullrange_data = fullrange_depletion_data.copy()
    
    list_mass_isotope = fullrange_depletion_spectra_multi_peak.isotope_mass_peaks
    list_scanwidth_isotope = fullrange_depletion_spectra_multi_peak.isotope_scanwidths
    st.session_state.fullrange_depletion_data = fullrange_depletion_data
    st.session_state.analysis_done = True

    # --- Analysis Parameters Report ---
    with st.expander("📊 Analysis Parameters Report", expanded=False):
        col_params, col_peaks = st.columns(2)
        with col_params:
            st.markdown("##### Width Parameters")
            st.markdown(f"""
            | Parameter | Value | Description |
            |-----------|-------|-------------|
            | **Search Width** | ±{st.session_state.get('search_width', 'N/A')} amu | Window for finding peak maxima |
            | **Integration Width** | ±{st.session_state.get('isotope_scan_width', 'N/A')} amu | Window for signal integration |
            """)
        with col_peaks:
            st.markdown("##### Refined Peak Positions")
            peak_data = []
            for i, (mass, scanwidth_idx) in enumerate(zip(list_mass_isotope, list_scanwidth_isotope)):
                mass_range = st.session_state.get("x_mass")
                if mass_range is not None and len(scanwidth_idx) > 0:
                    integration_range = f"{mass_range[scanwidth_idx[0]]:.2f} – {mass_range[scanwidth_idx[-1]]:.2f}"
                else:
                    integration_range = "N/A"
                peak_data.append({
                    "Isotope": i + 1,
                    "Theoretical Mass": f"{st.session_state['list_mass_isotope'][i]:.2f} amu",
                    "Refined Peak": f"{mass:.2f} amu",
                    "Integration Range": integration_range
                })
            st.dataframe(pd.DataFrame(peak_data), hide_index=True)

    # Convert to numpy array for plotting.
    # Only show the analysis tabs if analysis is done.
    if st.session_state.get("analysis_done"):
        # Convert results to a NumPy array.
        data = np.array(st.session_state.fullrange_depletion_data)
        exp_df = pd.DataFrame({
            "wavenumber": data[:, 0],
            "norm_intensity": data[:, 3] / np.max(data[:, 3])
        })
        st.write("Experimental spectrum preview:", exp_df.head())

    
    # --- Apply smoothing if "Smoothed" is selected ---
    if st.session_state.get("data_display_option") == "Smoothed":
        def apply_savgol(series, window_size):
            if window_size % 2 == 0:
                window_size += 1
            polyorder = 2 if window_size > 2 else 1
            return savgol_filter(series, window_length=window_size, polyorder=polyorder)
        data[:, 3] = apply_savgol(data[:, 3], st.session_state.get("smoothing_window"))
        data[:, 4] = apply_savgol(data[:, 4], st.session_state.get("smoothing_window"))
        # Mark that we’ve already smoothed the experimental data
        st.session_state.smoothed_experimental = True
        fullrange_depletion_data = fullrange_depletion_data.copy()
        fullrange_depletion_data.iloc[:, 3] = data[:, 3]
        fullrange_depletion_data.iloc[:, 4] = data[:, 4]
        st.session_state.fullrange_depletion_data = fullrange_depletion_data

    else:
        # if not smoothing, just overwrite with the raw results
        st.session_state.fullrange_depletion_data = fullrange_depletion_data

    st.session_state.analysis_done = True
    orig_fullrange_data.columns = [
        "wavenumber",
        "intensity_withoutIR_orig",
        "intensity_withIR_orig",
        "depletion_orig",
        "ln_depletion_orig",
    ]

    export_df = orig_fullrange_data.copy()
    export_df["depletion_smoothed"]    = fullrange_depletion_data.iloc[:, 3]
    export_df["ln_depletion_smoothed"] = fullrange_depletion_data.iloc[:, 4]
    
    # --- Update depletion plot ranges based on computed data ---
    if st.session_state.get("depletion_xmin", 0.0) == +0.1:
        st.session_state["depletion_xmin"] = float(np.min(data[:, 0]))
    if st.session_state.get("depletion_xmax", 2000) == 2000:
        st.session_state["depletion_xmax"] = float(np.max(data[:, 0]) + 1)
    if st.session_state.get("depletion_ymin", -0.1) == -0.1:
        st.session_state["depletion_ymin"] = float(np.min(data[:, 3]))
    if st.session_state.get("depletion_ymax", 1.5) == 1.5:
        st.session_state["depletion_ymax"] = float(np.max(data[:, 3]) + 0.1)
    if st.session_state.get("ln_depletion_ymin", -0.4) == -0.4:
        st.session_state["ln_depletion_ymin"] = float(np.min(data[:, 4]))
    if st.session_state.get("ln_depletion_ymax", 0.3) == 0.3:
        st.session_state["ln_depletion_ymax"] = float(np.max(data[:, 4]) + 0.1)
    
    depletion_xmin = st.session_state["depletion_xmin"]
    depletion_xmax = st.session_state["depletion_xmax"]
    depletion_ymin = st.session_state["depletion_ymin"]
    depletion_ymax = st.session_state["depletion_ymax"]
    ln_depletion_ymin = st.session_state["ln_depletion_ymin"]
    ln_depletion_ymax = st.session_state["ln_depletion_ymax"]
    st.write(fullrange_depletion_data)

    if save_output:
        filename_fullrange_depletion = f"fullrange_depletion_data_{complex}_{int(data[0,0])}-{int(data[-1,0])}cm⁻¹.csv"
        filename_fullrange_depletion_both = f"fullrange_depletion_both_{complex}_{int(data[0,0])}-{int(data[-1,0])}cm⁻¹.csv"
        export_filename_fullrange_depletion = os.path.join(file_directory, filename_fullrange_depletion)
        export_filename_fullrange_depletion_both = os.path.join(file_directory, filename_fullrange_depletion_both)
        fullrange_depletion_data.to_csv(export_filename_fullrange_depletion, index=False)
        export_df.to_csv(export_filename_fullrange_depletion_both, index=False)
        st.success(f"Original + smoothed data saved @ {export_filename_fullrange_depletion_both}")
    else:
        st.info("Save output is off; combined CSV not written.")
    x_range_dep = [st.session_state["depletion_xmin"], st.session_state["depletion_xmax"]]
    
    # ------------------- Tabs for Output -------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Mass spectra - specified wavenumber",
    "💥 Depletion - full range",
    "💥 -ln(depletion) - full range",
    "🔬 PAH Comparison",
    "📊 Intensity vs Wavenumber"
])
    
    # ---------- Tab 1: Mass Spectra ----------
    with tab1:
        st.markdown("###### *:green[Interactive plot with plotly]*")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[mass_complex, mass_complex],
            y=[-0.001, mass_ymax],
            mode='lines',
            line=dict(color="green", width=2, dash="solid"),
            name=f"{complex} average mass"
        ))
        for index, isotope in enumerate(list_mass_isotope):
            fig.add_trace(go.Scatter(
                x=[isotope, isotope],
                y=[0, mass_ymax],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
            fig.add_shape(
                type="rect",
                x0=isotope - isotope_scan_width,
                x1=isotope + isotope_scan_width,
                y0=0,
                y1=mass_ymax,
                fillcolor='lightgray',
                line=dict(color='rgba(0,0,0,0)'),
                opacity=0.4,
                layer="below"
            )
        fig.add_trace(go.Scatter(
            x=[0,0],
            y=[0,0],
            mode='lines',
            name='Isotope peaks',
            line=dict(color='black', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0,0],
            y=[0,0],
            mode='lines',
            name='scan width range',
            line=dict(color='lightgray', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=x_mass[:],
            y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withoutIR],
            mode='lines',
            name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withoutIR],
            line=dict(color="#1f77b4")
        ))
        fig.add_trace(go.Scatter(
            x=x_mass[:],
            y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withIR],
            mode='lines',
            name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withIR],
            line=dict(color="#ff7f0e")
        ))
        fig.add_shape(
            type="rect",
            x0=min(x_mass[baseline_range_indices]),
            y0=0,
            x1=max(x_mass[baseline_range_indices]),
            y1=mass_ymax,
            fillcolor="lightsteelblue",
            line=dict(color='rgba(0,0,0,0)'),
            opacity=0.4,
            layer="below"
        )
        fig.add_trace(go.Scatter(
            x=[0,0],
            y=[0,0],
            mode='lines',
            name='Baseline Range',
            line=dict(color='lightsteelblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[x_mass[mass_range_indices][0], x_mass[mass_range_indices][-1]],
            y=[0, 0],
            mode='lines',
            line=dict(color="lime", width=1),
            name='zero Line'
        ))
        fig.update_layout(
            xaxis=dict(range=[mass_xmin, mass_xmax]),
            yaxis=dict(range=[-0.001, mass_ymax]),
            xaxis_title="Mass (amu)",
            yaxis_title="Intensity",
            title=complex,
            legend=dict(x=0.8, y=0.9)
        )
        st.plotly_chart(fig)
        if save_output:
            html_filename = f"mass_spectra_{complex}_{int(data[0,0])}-{int(data[-1,0])}.html"
            output_html_filename = os.path.join(file_directory, html_filename)
            fig.write_html(output_html_filename, include_plotlyjs='cdn')
            st.write(f"Interactive Mass Spectra plot saved as HTML @ {output_html_filename}")
        
        st.markdown("###### *:green[Static plot with matplotlib]*")
        fig_static, ax = plt.subplots(figsize=(5, 3))
        ax.axvline(mass_complex, alpha=0.75, linestyle="solid", linewidth=2, color="green", label=f"{complex} average mass")
        for index, isotope in enumerate(list_mass_isotope):
            ax.axvline(isotope, alpha=0.75, linestyle="solid", linewidth=1, color="black")
            ax.fill_between(x_mass[list_scanwidth_isotope[index]], 0.5, color="lightgray")
        ax.axvline(0, color='black', label='Isotope peaks')
        ax.axvline(0, color='lightgray', label='scan width range')
        ax.plot(x_mass[:], compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withoutIR],
                label=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withoutIR])
        ax.plot(x_mass[:], compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withIR],
                label=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withIR])
        ax.fill_between(x_mass[baseline_range_indices], 0.2, color="lightsteelblue", label="baseline range")
        ax.hlines(0, xmin=x_mass[mass_range_indices][0], xmax=x_mass[mass_range_indices][-1], color="lime")
        ax.set_xlim(mass_xmin, mass_xmax)
        ax.set_ylim(-0.001, mass_ymax)
        ax.set_xlabel("Mass (amu)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=5)
        fig_static.tight_layout()
        st.pyplot(fig_static)
        if save_output:
            output_png_filename = os.path.join(file_directory, f"mass_spectra_{complex}_{int(data[0,0])}-{int(data[-1,0])}.png")
            fig_static.savefig(output_png_filename, dpi=300)
    
    # ---------- Tab 2: Depletion Plot ----------
    with tab2:
        st.markdown("###### *:green[Interactive plot with plotly]*")
        fig_dep = go.Figure()
        fig_dep.add_trace(go.Scatter(
            x=data[:, 0],
            y=data[:, 3],
            name=fullrange_depletion_data.columns[3],
            line=dict(color="#1f77b4")
        ))
        zero_line_depletion = 1
        fig_dep.add_trace(go.Scatter(
            x=[data[0, 0], data[-1, 0]],
            y=[zero_line_depletion, zero_line_depletion],
            mode='lines',
            line=dict(color="lime", width=1),
            name="Zero Line"
        ))
        fig_dep.update_layout(
            xaxis=dict(range=x_range_dep),
            yaxis=dict(range=[depletion_ymin, depletion_ymax]),
            xaxis_title="wavenumber (cm-1)",
            yaxis_title="Intensity",
            title=fullrange_depletion_data.columns[3] + " " + complex,
            legend=dict(x=0.8, y=0.9)
        )
        st.plotly_chart(fig_dep)
        if save_output:
            html_filename = f"depletion_{complex}_{int(data[0,0])}-{int(data[-1,0])}.html"
            output_html_filename = os.path.join(file_directory, html_filename)
            fig_dep.write_html(output_html_filename, include_plotlyjs='cdn')
            st.write(f"Interactive Depletion plot saved as HTML @ {output_html_filename}")
        
        st.markdown("###### *:green[Static plot with matplotlib]*")
        fig_dep_static, ax = plt.subplots(figsize=(21, 6))
        ax.plot(data[:, 0], data[:, 3])
        ax.scatter(data[:, 0], data[:, 3])
        ax.legend([fullrange_depletion_data.columns[3]], fontsize=5, loc="upper right")
        ax.hlines(zero_line_depletion, xmin=depletion_xmin, xmax=depletion_xmax, color="lime")
        mass_label = [round(item, 2) for item in list_mass_isotope]
        textstr = (f"Complex: {complex} \nMass peaks: {mass_label} amu \nIntegration width = {isotope_scan_width} amu\n"
                   f"Baseline reference: {baseline_reference} amu \nBaseline width = {baseline_width} amu")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.6, 0.25, textstr, transform=plt.gca().transAxes, fontsize=5,
                verticalalignment='top', bbox=props)
        ax.set_ylim(depletion_ymin, depletion_ymax)
        ax.set_xlim(depletion_xmin, depletion_xmax)
        ax.set_xlabel("wavenumber (cm-1)")
        st.pyplot(fig_dep_static)
        if save_output:
            output_png_filename = os.path.join(file_directory, f"depletion_{complex}_{int(data[0,0])}-{int(data[-1,0])}.png")
            fig_dep_static.savefig(output_png_filename, dpi=300)
    
    # ---------- Tab 3: -ln(Depletion) Plot ----------
    with tab3:
        st.markdown("###### *:green[Interactive plot with plotly]*")
        fig_ln = go.Figure()
        fig_ln.add_trace(go.Scatter(
            x=data[:, 0],
            y=data[:, 4],
            name=fullrange_depletion_data.columns[4],
            line=dict(color="#1f77b4")
        ))
        zero_line_ln = 0
        fig_ln.add_trace(go.Scatter(
            x=[data[0, 0], data[-1, 0]],
            y=[zero_line_ln, zero_line_ln],
            mode='lines',
            line=dict(color="lime", width=1),
            name="Zero Line"
        ))
        fig_ln.update_layout(
            xaxis=dict(range=x_range_dep),
            yaxis=dict(range=[ln_depletion_ymin, ln_depletion_ymax]),
            xaxis_title="wavenumber (cm-1)",
            yaxis_title="Intensity",
            title=fullrange_depletion_data.columns[4] + " " + complex,
            legend=dict(x=0.8, y=0.9)
        )
        st.plotly_chart(fig_ln)
        if save_output:
            html_filename = f"ln_depletion_{complex}_{int(data[0,0])}-{int(data[-1,0])}.html"
            output_html_filename = os.path.join(file_directory, html_filename)
            fig_ln.write_html(output_html_filename, include_plotlyjs='cdn')
            st.write(f"Interactive -ln(Depletion) plot saved as HTML @ {output_html_filename}")
        
        st.markdown("###### *:green[Static plot with matplotlib]*")
        fig_ln_static, ax = plt.subplots(figsize=(21, 6))
        ax.plot(data[:, 0], data[:, 4])
        ax.scatter(data[:, 0], data[:, 4])
        ax.legend([fullrange_depletion_data.columns[4]], fontsize=5, loc="lower right")
        ax.hlines(zero_line_ln, xmin=depletion_xmin, xmax=depletion_xmax, color="lime")
        mass_label = [round(item, 2) for item in list_mass_isotope]
        textstr = (f"Complex: {complex} \nMass peaks: {mass_label} amu \nIntegration width = {isotope_scan_width} amu\n"
                   f"Baseline reference: {baseline_reference} amu \nBaseline width = {baseline_width} amu")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=5,
                verticalalignment='top', bbox=props)
        ax.set_ylim(ln_depletion_ymin, ln_depletion_ymax)
        ax.set_xlim(depletion_xmin, depletion_xmax)
        ax.set_xlabel("wavenumber (cm-1)")
        st.pyplot(fig_ln_static)
        if save_output:
            output_png_filename = os.path.join(file_directory, f"ln_depletion_{complex}_{int(data[0,0])}-{int(data[-1,0])}.png")
            fig_ln_static.savefig(output_png_filename, dpi=300)
    
    # ---------- Tab 4: PAH Comparison ----------
    with tab4:
        st.markdown("#### PAH Comparison: Experimental vs. Theoretical IR Spectrum")
        with st.form("theory_form"):
            xml_path = st.text_input(
                "Enter path to PAH XML file",
                value=""
            )
            uid_input = st.text_input("Enter PAH UID (e.g., 18 for coronene)", value="495")
            conv_type = st.selectbox("Convolution Type", options=["Gaussian", "Lorentzian"], index=0)
            fwhm = st.number_input("FWHM for convolution (cm⁻¹)", value=15.0)
            submitted = st.form_submit_button("Load Theoretical Spectrum")
            # st.write("Theoretical data preview:", theory_df.head())

        if submitted:
            try:
                
                from amespahdbpythonsuite.amespahdb import AmesPAHdb
                # Create the database instance with desired settings
                pahdb = AmesPAHdb(filename=xml_path, check=False, cache=True)
                uid = int(uid_input)
                # Retrieve the transitions for the given UID
                transitions = pahdb.gettransitionsbyuid([uid])
                # Convolve the stick spectrum using the chosen line profile
                # Plot the emission 'stick' spectrum.
                transitions.plot(show=True)
                spectrum = transitions.convolve(fwhm=fwhm, gaussian=(conv_type == "Gaussian"), multiprocessing=False)
                # Extract frequency and intensity data from the convolved spectrum
                freq, conv_intensity = spectrum.get()  # returns arrays
                norm_conv_intensity = conv_intensity / np.max(conv_intensity)
                theory_df = pd.DataFrame({"wavenumber": freq, "norm_intensity": norm_conv_intensity})
                st.subheader("Theoretical Spectrum")
                st.dataframe(theory_df.head())
            except Exception as e:
                st.error(f"Error loading theoretical spectrum: {e}")
                theory_df = None
    #             # Create the database instance with desired settings
    #             pahdb = AmesPAHdb(filename=xml_path, check=False, cache=True)
    #             uid = int(uid_input)
    #             # Retrieve the transitions for the given UID
    #             transitions = pahdb.gettransitionsbyuid([uid])
    #             # Convolve the stick spectrum using the chosen line profile
    #             # Plot the emission 'stick' spectrum.
    #             transitions.plot(show=True)
    #             spectrum = transitions.convolve(fwhm=fwhm, gaussian=(conv_type == "Gaussian"), multiprocessing=False)
    #             # Extract frequency and intensity data from the convolved spectrum
    #             freq, conv_intensity = spectrum.get()  # returns arrays
    #             norm_conv_intensity = conv_intensity / np.max(conv_intensity)
    #             theory_df = pd.DataFrame({"wavenumber": freq, "norm_intensity": norm_conv_intensity})
    #             st.subheader("Theoretical Spectrum")
    #             st.dataframe(theory_df.head())
    #         except Exception as e:
    #             st.error(f"Error loading theoretical spectrum: {e}")
    #             theory_df = None

    #     # Generate the comparison plot if both experimental and theoretical data exist.
    #     if exp_df is not None and 'theory_df' in locals() and theory_df is not None:
    #         shift_val = st.slider("Shift Theoretical Spectrum (cm⁻¹)", min_value=-20, max_value=20, value=0)
    #         theory_df['wavenumber_shifted'] = theory_df['wavenumber'] + shift_val
    #         fig_pah = go.Figure()
    #         # Use your experimental depletion data (assumed normalized)
    #         exp_wavenumbers = data[:, 0]
    #         exp_intensity = data[:, 3] / np.max(data[:, 3])
    #         fig_pah.add_trace(go.Scatter(
    #             x=exp_wavenumbers,
    #             y=exp_intensity,
    #             mode='lines',
    #             name='Experimental'
    #         ))
    #         fig_pah.add_trace(go.Scatter(
    #             x=theory_df['wavenumber_shifted'],
    #             y=theory_df['norm_intensity'],
    #             mode='lines',
    #             name='Theoretical'
    #         ))
    #         fig_pah.update_layout(
    #             xaxis_title="Wavenumber (cm⁻¹)",
    #             yaxis_title="Normalized Intensity",
    #             title="Experimental vs. Theoretical IR Spectrum"
    #         )
    #         fig_pah.update_xaxes(autorange="reversed")  # Invert x-axis per IR convention
    #         st.plotly_chart(fig_pah, use_container_width=True)
    #     else:
    #         st.info("Insufficient data to generate a comparison plot. Please ensure that both experimental and theoretical data are loaded.")
    with tab5:
        list_mass_isotope = st.session_state["list_mass_isotope"]
    # join into a string for display
        mass_label_str = ", ".join(str(m) for m in list_mass_isotope)
 
        st.markdown(f"#### Integrated Intensity at mass {mass_label_str} vs Wavenumber")
        
        # interactive plot with plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fullrange_depletion_data.iloc[:, 0],
            y=fullrange_depletion_data.iloc[:, 1],
            mode='lines',
            name="Without IR",
            line=dict(color="black")
        ))
        fig.add_trace(go.Scatter(
            x=fullrange_depletion_data.iloc[:, 0],
            y=fullrange_depletion_data.iloc[:, 2],
            mode='lines',
            name="With IR",
            line=dict(color="red")
        ))
        fig.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Integrated Intensity",
            legend=dict(x=0.8, y=0.9)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # static plot with matplotlib
        st.markdown("###### *:green[Static plot with matplotlib]*")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(
            fullrange_depletion_data.iloc[:, 0],
            fullrange_depletion_data.iloc[:, 1],
            color="black",
            label="Without IR"
        )
        ax2.plot(
            fullrange_depletion_data.iloc[:, 0],
            fullrange_depletion_data.iloc[:, 2],
            color="red",
            label="With IR"
        )
        ax2.set_xlabel("Wavenumber (cm⁻¹)")
        ax2.set_ylabel("Integrated Intensity")
        ax2.set_title(f"Mass: {mass_label_str}")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)