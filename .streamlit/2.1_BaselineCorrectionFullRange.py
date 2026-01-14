# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from packages.BaselineCorrection import *
# import plotly.graph_objs as go


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
# check_wavenumber = st.session_state.get("check_wavenumber", None)


# if st.button("**:blue[#1]** ✨ Perform baseline correction - full range"):

#     # Initialize variables
#     baseline_corrected_data = {}
#     baseline_reference_values = {}
#     compilation_baseline_corrected_data = {}
#     st.session_state["compilation_baseline_corrected_data"] = {}
#     baseline_correction_fullrange = {}

#     baseline_correction_fullrange = baseline(baseline_reference = baseline_reference, interval = baseline_width, target_mass=x_mass)

    

#     for wavenumber in unique_wavenumbers:

#         # st.write(compiled_data[wavenumber])
#         # initialize variables
#         sum_withoutIR = {}
#         sum_withIR = {}
#         sum_baseline_corrected_withoutIR = {}
#         sum_baseline_corrected_withIR = {}
#         new_table = {}

#         # sum every other column
#         sum_withoutIR = compiled_data[wavenumber].iloc[:,0 ::2].sum(axis=1)
#         sum_withIR = compiled_data[wavenumber].iloc[:,1 ::2].sum(axis=1)

#         # calculate the mean of the baseline region
#         mean_value_withoutIR = np.mean(sum_withoutIR[baseline_range_indices])
#         mean_value_withIR = np.mean(sum_withIR[baseline_range_indices])

#         # perform baseline correction
#         sum_baseline_corrected_withoutIR = sum_withoutIR - mean_value_withoutIR
#         sum_baseline_corrected_withIR = sum_withIR - mean_value_withIR

#         # compile everything into a new table
#         # sum and baseline corrected sum for debugging purposes
#         new_table = pd.DataFrame({
#                 "sum_"+str(wavenumber)+"_withoutIR": sum_withoutIR,
#                 "sum_"+str(wavenumber)+"_withIR": sum_withIR,
#                 "sum_baseline_corrected_"+str(wavenumber)+"_withoutIR": sum_baseline_corrected_withoutIR,
#                 "sum_baseline_corrected_"+str(wavenumber)+"_withIR": sum_baseline_corrected_withIR
#             })

        
#         # merge current table containing all individual scans with new table + the sums
#         compilation_baseline_corrected_data[wavenumber] = pd.concat([compiled_data[wavenumber],new_table],axis=1)
        

#     st.session_state["compilation_baseline_corrected_data"] = compilation_baseline_corrected_data
#     st.write("Success! 😎")
#     # st.table(compilation_baseline_corrected_data[check_wavenumber].head())

#     # In case you run into the known bug, comment out `st.table ....` and uncomment the 3 lines below to see the output in the terminal
#     print("\n")
#     pd.set_option('display.max_columns', None)
#     print(compilation_baseline_corrected_data[check_wavenumber].head())


# st.markdown("#### Plot parameters")

# col1, col2, col3, col4, col5 = st.columns([0.4,0.05,0.7,0.05, 1])
# with col1:
#     st.session_state["check_wavenumber"] = float(st.text_input("Wavenumber to check plots", value = st.session_state.get("check_wavenumber",1320)))
#     st.session_state["y_max2"] = float(st.text_input("Maximum y-value", value = st.session_state.get("y_max2", 0.2)))
# with col3:
#     st.session_state["x_min2"] = float(st.text_input("Minimum x-value", value = st.session_state.get("x_min2", 0.0)))
#     st.session_state["x_max2"] = float(st.text_input("Maximum x-value", value = st.session_state.get("x_max2", 1300)))
# with col5:
#     st.session_state["plot_columnIndex_withoutIR"] = int(st.number_input("Column index for signal without IR irradiation", value = st.session_state.get("plot_columnIndex_withoutIR", -2)))
#     st.session_state["plot_columnIndex_withIR"] = int(st.number_input("Column index for signal with IR irradiation", value = st.session_state.get("plot_columnIndex_withIR", -1)))

    



# if st.button("**:blue[#2]** 🚀 Plot full range data"):

#     # Initialize variables
#     check_wavenumber = st.session_state.get("check_wavenumber", None)
#     plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
#     plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
#     compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
#     y_max2 = st.session_state.get("y_max2", None)
#     x_min2 = st.session_state.get("x_min2", None)
#     x_max2 = st.session_state.get("x_max2", None)

#     # Plot via matplotlib - static
#     # fig, ax = plt.subplots()
#     # plt.ion()
#     # ax.axvline(mass_complex,alpha=0.75,linestyle="solid",linewidth=1, color="green", label = complex)
#     # ax.plot(x_mass[mass_range_indices],compilation_baseline_corrected_data[check_wavenumber].iloc[mass_range_indices,plot_columnIndex_withoutIR], label = compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withoutIR])
#     # ax.plot(x_mass[mass_range_indices],compilation_baseline_corrected_data[check_wavenumber].iloc[mass_range_indices,plot_columnIndex_withIR], label = compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withIR])
#     # ax.fill_between(x_mass[baseline_range_indices],0.2, color = "lightgray", label = "baseline range")
#     # ax.hlines(0,xmin = x_mass[mass_range_indices][0], xmax =x_mass[mass_range_indices][-1], color="lime")
#     # ax.set_ylim(-0.001, y_max2)
#     # ax.legend()
#     # st.pyplot(fig)
#     # plt.close(fig)


#     # Plot via plotly - interactive

#     fig = go.Figure()

#     # Plot data for "without IR" signal
#     fig.add_trace(go.Scatter(
#         x=x_mass[:],
#         y=compilation_baseline_corrected_data[check_wavenumber].iloc[:, plot_columnIndex_withoutIR],
#         mode='lines',
#         name=compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withoutIR],
#         line=dict(color="#1f77b4"),
#         opacity=0.75
#     ))
    
#     # Plot data for "with IR" signal
#     fig.add_trace(go.Scatter(
#         x=x_mass[:],
#         y=compilation_baseline_corrected_data[check_wavenumber].iloc[:, plot_columnIndex_withIR],
#         mode='lines',
#         name=compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withIR],
#         line=dict(color="#ff7f0e"),
#         opacity=0.75
#     ))

#     # Plot vertical line at mass_complex
#     fig.add_trace(go.Scatter(
#         x=[mass_complex, mass_complex],
#         y=[-0.001, y_max2],
#         mode='lines',
#         line=dict(color="green", width=1, dash="solid"),
#         name=str(complex)
#     ))


#     # Add rectangle region for baseline range
#     fig.add_shape(
#         type="rect",
#         x0=min(x_mass[baseline_range_indices]),  # Starting x coordinate
#         y0=0,                                    # Starting y coordinate
#         x1=max(x_mass[baseline_range_indices]),  # Ending x coordinate
#         y1=y_max2,                               # Ending y coordinate
#         fillcolor="lightsteelblue",                   # Fill color
#         line=dict(color='rgba(0,0,0,0)'),            # Border color
#         opacity = 0.4,
#         layer = "below"
#     )
    
#     # Add an invisible scatter trace to represent the rectangle in the legend
#     fig.add_trace(go.Scatter(
#         x=[0,0],  # Use None for x to avoid displaying a line
#         y=[0,0],  # Use None for y to avoid displaying a line
#         mode='lines',  # Mode can be anything; it won't show
#         name='Baseline Range',  # This will appear in the legend
#         line=dict(color='lightsteelblue', width=2)  # Invisible line
#     ))

#     # Plot horizontal line at y=0
#     fig.add_trace(go.Scatter(
#         x=[x_mass[mass_range_indices][0], x_mass[mass_range_indices][-1]],
#         y=[0, 0],
#         mode='lines',
#         line=dict(color="lime", width=1),
#         name='zero Line'
#     ))

#     # Update layout for the plot
#     fig.update_layout(
#         yaxis=dict(range=[-0.001, y_max2]),
#         xaxis_title="Mass (amu)",
#         yaxis_title="Intensity",
#         xaxis=dict(range=[x_min2, x_max2]),
#         title=complex,
#         legend=dict(x=0.8, y=0.9)
#     )

#     # Display plot in Streamlit
#     st.plotly_chart(fig)


# st.markdown(
#     "###### *:red[Known bug:]* If there are instances of the :blue[same wavenumber per file], Streamlit will give an error in trying to display it.<br>"
#     "The problem is with Streamlit; it's fine on the terminal and does not affect the concatenated data in any way.<br>"
#     "My proposed solution is to pick a different `wavenumber` to check. See line 84 of `./streamlit/2.1_BaselineCorrectionFullRange.py`.",
#     unsafe_allow_html=True
# )
# # NEW CODE March 13 :
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import pandas as pd
# # from packages.BaselineCorrection import baseline
# # import plotly.graph_objs as go
# # import streamlit as st

# # # Import variables from session_state
# # file_directory = st.session_state.get("file_directory", None)
# # compiled_data = st.session_state.get("compiled_data", None)
# # unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
# # element1 = st.session_state.get("element1", None)
# # element2 = st.session_state.get("element2", None)
# # element3 = st.session_state.get("element3", None)
# # mass_element1 = st.session_state.get("mass_element1", None)
# # mass_element2 = st.session_state.get("mass_element2", None)
# # mass_element3 = st.session_state.get("mass_element3", None)
# # charge_state = st.session_state.get("charge_state", None)
# # x_mass = st.session_state.get("x_mass", None)
# # x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)
# # baseline_reference = st.session_state.get("baseline_reference", None)
# # baseline_width = st.session_state.get("baseline_width", None)
# # complex = st.session_state.get("complex", None)
# # mass_complex = st.session_state.get("mass_complex", None)
# # mass_range_indices = st.session_state.get("mass_range_indices", None)
# # baseline_range_indices = st.session_state.get("baseline_range_indices", None)
# # check_wavenumber = st.session_state.get("check_wavenumber", None)

# # # New: Allow user to select baseline method for full-range correction
# # st.markdown("#### Baseline Correction Method")
# # st.session_state["baseline_method"] = st.selectbox(
# #     "Select baseline correction method",
# #     options=["Mean Subtraction", "airPLS", "ASLS", "Rubberband"],
# #     index=0
# # )
# # baseline_method = st.session_state.get("baseline_method", "Mean Subtraction")

# # if st.button("**:blue[#1]** ✨ Perform baseline correction - full range"):
# #     compilation_baseline_corrected_data = {}
# #     st.session_state["compilation_baseline_corrected_data"] = {}
    
# #     # For each wavenumber, perform baseline correction on summed data
# #     for wavenumber in unique_wavenumbers:
# #         # Sum every other column
# #         sum_withoutIR = compiled_data[wavenumber].iloc[:, 0::2].sum(axis=1)
# #         sum_withIR = compiled_data[wavenumber].iloc[:, 1::2].sum(axis=1)
        
# #         # Determine the mean baseline over the baseline range indices
# #         if baseline_method == "Mean Subtraction":
# #             mean_without = np.mean(sum_withoutIR[baseline_range_indices])
# #             mean_with = np.mean(sum_withIR[baseline_range_indices])
# #             corrected_without = sum_withoutIR - mean_without
# #             corrected_with = sum_withIR - mean_with
# #         else:
# #             # Use the baseline class to perform the chosen correction on the summed data.
# #             bc = baseline(
# #                 baseline_reference=baseline_reference,
# #                 interval=baseline_width,
# #                 wavenumber=wavenumber,
# #                 column_withoutIR="sum_withoutIR",
# #                 column_withIR="sum_withIR",
# #                 data_withoutIR=sum_withoutIR,
# #                 data_withIR=sum_withIR,
# #                 target_mass=x_mass,
# #                 method=baseline_method
# #             )
# #             bc.baseline_range()
# #             bc.baseline_mean()
# #             corrected_df = bc.baseline_correction()
# #             corrected_without = corrected_df.iloc[:, 0]
# #             corrected_with = corrected_df.iloc[:, 1]
        
# #         new_table = pd.DataFrame({
# #             "sum_" + str(wavenumber) + "_withoutIR": sum_withoutIR,
# #             "sum_" + str(wavenumber) + "_withIR": sum_withIR,
# #             "sum_baseline_corrected_" + str(wavenumber) + "_withoutIR": corrected_without,
# #             "sum_baseline_corrected_" + str(wavenumber) + "_withIR": corrected_with
# #         })
        
# #         # Merge the new table with the original compiled data
# #         compilation_baseline_corrected_data[wavenumber] = pd.concat([compiled_data[wavenumber], new_table], axis=1)
    
# #     st.session_state["compilation_baseline_corrected_data"] = compilation_baseline_corrected_data
# #     st.write("Success! 😎")
# #     print("\n")
# #     pd.set_option('display.max_columns', None)
# #     print(compilation_baseline_corrected_data[check_wavenumber].head())

# # st.markdown("#### Plot parameters")

# # col1, col2, col3, col4, col5 = st.columns([0.4, 0.05, 0.7, 0.05, 1])
# # with col1:
# #     st.session_state["check_wavenumber"] = float(
# #         st.text_input("Wavenumber to check plots", value=st.session_state.get("check_wavenumber", 1320))
# #     )
# #     st.session_state["y_max2"] = float(
# #         st.text_input("Maximum y-value", value=st.session_state.get("y_max2", 0.2))
# #     )
# # with col3:
# #     st.session_state["x_min2"] = float(
# #         st.text_input("Minimum x-value", value=st.session_state.get("x_min2", 0.0))
# #     )
# #     st.session_state["x_max2"] = float(
# #         st.text_input("Maximum x-value", value=st.session_state.get("x_max2", 1300))
# #     )
# # with col5:
# #     st.session_state["plot_columnIndex_withoutIR"] = int(
# #         st.number_input("Column index for signal without IR irradiation", value=st.session_state.get("plot_columnIndex_withoutIR", -2))
# #     )
# #     st.session_state["plot_columnIndex_withIR"] = int(
# #         st.number_input("Column index for signal with IR irradiation", value=st.session_state.get("plot_columnIndex_withIR", -1))
# #     )

# # if st.button("**:blue[#2]** 🚀 Plot full range data"):
# #     check_wavenumber = st.session_state.get("check_wavenumber", None)
# #     plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
# #     plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
# #     compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
# #     y_max2 = st.session_state.get("y_max2", None)
# #     x_min2 = st.session_state.get("x_min2", None)
# #     x_max2 = st.session_state.get("x_max2", None)

# #     # Plot via Plotly - interactive
# #     fig = go.Figure()

# #     # Plot data for "without IR" signal
# #     fig.add_trace(go.Scatter(
# #         x=x_mass[:],
# #         y=compilation_baseline_corrected_data[check_wavenumber].iloc[:, plot_columnIndex_withoutIR],
# #         mode='lines',
# #         name=compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withoutIR],
# #         line=dict(color="#1f77b4"),
# #         opacity=0.75
# #     ))
    
# #     # Plot data for "with IR" signal
# #     fig.add_trace(go.Scatter(
# #         x=x_mass[:],
# #         y=compilation_baseline_corrected_data[check_wavenumber].iloc[:, plot_columnIndex_withIR],
# #         mode='lines',
# #         name=compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withIR],
# #         line=dict(color="#ff7f0e"),
# #         opacity=0.75
# #     ))

# #     # Plot vertical line at mass_complex
# #     fig.add_trace(go.Scatter(
# #         x=[mass_complex, mass_complex],
# #         y=[-0.001, y_max2],
# #         mode='lines',
# #         line=dict(color="green", width=1, dash="solid"),
# #         name=str(complex)
# #     ))

# #     # Add rectangle for baseline range
# #     fig.add_shape(
# #         type="rect",
# #         x0=min(x_mass[baseline_range_indices]),
# #         y0=0,
# #         x1=max(x_mass[baseline_range_indices]),
# #         y1=y_max2,
# #         fillcolor="lightsteelblue",
# #         line=dict(color='rgba(0,0,0,0)'),
# #         opacity=0.4,
# #         layer="below"
# #     )
    
# #     # Add an invisible trace for the rectangle legend
# #     fig.add_trace(go.Scatter(
# #         x=[0, 0],
# #         y=[0, 0],
# #         mode='lines',
# #         name='Baseline Range',
# #         line=dict(color='lightsteelblue', width=2)
# #     ))

# #     # Horizontal line at y=0
# #     fig.add_trace(go.Scatter(
# #         x=[x_mass[baseline_range_indices][0], x_mass[baseline_range_indices][-1]],
# #         y=[0, 0],
# #         mode='lines',
# #         line=dict(color="lime", width=1),
# #         name='zero Line'
# #     ))

# #     # Update layout
# #     fig.update_layout(
# #         yaxis=dict(range=[-0.001, y_max2]),
# #         xaxis_title="Mass (amu)",
# #         yaxis_title="Intensity",
# #         xaxis=dict(range=[x_min2, x_max2]),
# #         title=complex,
# #         legend=dict(x=0.8, y=0.9)
# #     )

# #     st.plotly_chart(fig)

# # st.markdown(
# #     "###### *:red[Known bug:]* If there are instances of the :blue[same wavenumber per file], "
# #     "Streamlit will give an error in trying to display it.<br>"
# #     "The problem is with Streamlit; it's fine on the terminal and does not affect the concatenated data in any way.<br>"
# #     "My proposed solution is to pick a different `wavenumber` to check. See line 84 of `./streamlit/2.1_BaselineCorrectionFullRange.py`.",
# #     unsafe_allow_html=True
# # )
# # new new code:

# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from packages.BaselineCorrection import baseline
# import plotly.graph_objs as go

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
# baseline_reference = st.session_state.get("baseline_reference", None)
# baseline_width = st.session_state.get("baseline_width", None)
# complex = st.session_state.get("complex", None)
# mass_complex = st.session_state.get("mass_complex", None)
# mass_range_indices = st.session_state.get("mass_range_indices", None)
# baseline_range_indices = st.session_state.get("baseline_range_indices", None)
# check_wavenumber = st.session_state.get("check_wavenumber", None)

# # New: Allow user to select baseline method for full-range correction
# st.markdown("#### Baseline Correction Method")
# st.session_state["baseline_method"] = st.selectbox(
#     "Select baseline correction method",
#     options=["Mean Subtraction", "arPLS"],
#     index=0
# )
# baseline_method = st.session_state.get("baseline_method", "Mean Subtraction")

# if st.button("**:blue[#1]** ✨ Perform baseline correction - full range"):
#     compilation_baseline_corrected_data = {}
#     st.session_state["compilation_baseline_corrected_data"] = {}
    
#     # For each wavenumber, perform baseline correction on summed data
#     for wavenumber in unique_wavenumbers:
#         # Sum every other column
#         sum_withoutIR = compiled_data[wavenumber].iloc[:, 0::2].sum(axis=1)
#         sum_withIR = compiled_data[wavenumber].iloc[:, 1::2].sum(axis=1)
        
#         if baseline_method == "Mean Subtraction":
#             mean_without = np.mean(sum_withoutIR[baseline_range_indices])
#             mean_with = np.mean(sum_withIR[baseline_range_indices])
#             corrected_without = sum_withoutIR - mean_without
#             corrected_with = sum_withIR - mean_with
#         elif baseline_method == "arPLS":
#             # Use the baseline class to perform the arPLS correction on summed data
#             bc = baseline(
#                 baseline_reference=baseline_reference,
#                 interval=baseline_width,
#                 wavenumber=wavenumber,
#                 column_withoutIR="sum_withoutIR",
#                 column_withIR="sum_withIR",
#                 data_withoutIR=sum_withoutIR,
#                 data_withIR=sum_withIR,
#                 target_mass=x_mass,
#                 method=baseline_method
#             )
#             bc.baseline_range()
#             bc.baseline_mean()
#             corrected_df = bc.baseline_correction()
#             corrected_without = corrected_df.iloc[:, 0]
#             corrected_with = corrected_df.iloc[:, 1]
#         else:
#             raise ValueError("Unknown baseline method selected.")
        
#         new_table = pd.DataFrame({
#             "sum_" + str(wavenumber) + "_withoutIR": sum_withoutIR,
#             "sum_" + str(wavenumber) + "_withIR": sum_withIR,
#             "sum_baseline_corrected_" + str(wavenumber) + "_withoutIR": corrected_without,
#             "sum_baseline_corrected_" + str(wavenumber) + "_withIR": corrected_with
#         })
        
#         # Merge the new table with the original compiled data
#         compilation_baseline_corrected_data[wavenumber] = pd.concat([compiled_data[wavenumber], new_table], axis=1)
    
#     st.session_state["compilation_baseline_corrected_data"] = compilation_baseline_corrected_data
#     st.write("Success! 😎")
#     print("\n")
#     pd.set_option('display.max_columns', None)
#     print(compilation_baseline_corrected_data[check_wavenumber].head())

# st.markdown("#### Plot parameters")

# col1, col2, col3, col4, col5 = st.columns([0.4, 0.05, 0.7, 0.05, 1])
# with col1:
#     st.session_state["check_wavenumber"] = float(
#         st.text_input("Wavenumber to check plots", value=st.session_state.get("check_wavenumber", 1320))
#     )
#     st.session_state["y_max2"] = float(
#         st.text_input("Maximum y-value", value=st.session_state.get("y_max2", 0.2))
#     )
# with col3:
#     st.session_state["x_min2"] = float(
#         st.text_input("Minimum x-value", value=st.session_state.get("x_min2", 0.0))
#     )
#     st.session_state["x_max2"] = float(
#         st.text_input("Maximum x-value", value=st.session_state.get("x_max2", 1300))
#     )
# with col5:
#     st.session_state["plot_columnIndex_withoutIR"] = int(
#         st.number_input("Column index for signal without IR", value=st.session_state.get("plot_columnIndex_withoutIR", -2))
#     )
#     st.session_state["plot_columnIndex_withIR"] = int(
#         st.number_input("Column index for signal with IR", value=st.session_state.get("plot_columnIndex_withIR", -1))
#     )

# if st.button("**:blue[#2]** 🚀 Plot full range data"):
#     check_wavenumber = st.session_state.get("check_wavenumber", None)
#     plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
#     plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
#     compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
#     y_max2 = st.session_state.get("y_max2", None)
#     x_min2 = st.session_state.get("x_min2", None)
#     x_max2 = st.session_state.get("x_max2", None)

#     # Plot via Plotly - interactive
#     fig = go.Figure()

#     # Plot data for "without IR" signal
#     fig.add_trace(go.Scatter(
#         x=x_mass[:],
#         y=compilation_baseline_corrected_data[check_wavenumber].iloc[:, plot_columnIndex_withoutIR],
#         mode='lines',
#         name=compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withoutIR],
#         line=dict(color="#1f77b4"),
#         opacity=0.75
#     ))
    
#     # Plot data for "with IR" signal
#     fig.add_trace(go.Scatter(
#         x=x_mass[:],
#         y=compilation_baseline_corrected_data[check_wavenumber].iloc[:, plot_columnIndex_withIR],
#         mode='lines',
#         name=compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withIR],
#         line=dict(color="#ff7f0e"),
#         opacity=0.75
#     ))

#     # Plot vertical line at mass_complex
#     fig.add_trace(go.Scatter(
#         x=[mass_complex, mass_complex],
#         y=[-0.001, y_max2],
#         mode='lines',
#         line=dict(color="green", width=1, dash="solid"),
#         name=str(complex)
#     ))

#     # Add rectangle for baseline range
#     fig.add_shape(
#         type="rect",
#         x0=min(x_mass[baseline_range_indices]),
#         y0=0,
#         x1=max(x_mass[baseline_range_indices]),
#         y1=y_max2,
#         fillcolor="lightsteelblue",
#         line=dict(color='rgba(0,0,0,0)'),
#         opacity=0.4,
#         layer="below"
#     )
    
#     # Add an invisible trace for the rectangle legend
#     fig.add_trace(go.Scatter(
#         x=[0, 0],
#         y=[0, 0],
#         mode='lines',
#         name='Baseline Range',
#         line=dict(color='lightsteelblue', width=2)
#     ))

#     # Horizontal line at y=0
#     fig.add_trace(go.Scatter(
#         x=[x_mass[baseline_range_indices][0], x_mass[baseline_range_indices][-1]],
#         y=[0, 0],
#         mode='lines',
#         line=dict(color="lime", width=1),
#         name='zero Line'
#     ))

#     # Update layout
#     fig.update_layout(
#         yaxis=dict(range=[-0.001, y_max2]),
#         xaxis_title="Mass (amu)",
#         yaxis_title="Intensity",
#         xaxis=dict(range=[x_min2, x_max2]),
#         title=complex,
#         legend=dict(x=0.8, y=0.9)
#     )

#     st.plotly_chart(fig)

# st.markdown(
#     "###### *:red[Known bug:]* If there are instances of the :blue[same wavenumber per file], "
#     "Streamlit will give an error in trying to display it.<br>"
#     "The problem is with Streamlit; it's fine on the terminal and does not affect the concatenated data in any way.<br>"
#     "My proposed solution is to pick a different `wavenumber` to check. See line 84 of `./streamlit/2.1_BaselineCorrectionFullRange.py`.",
#     unsafe_allow_html=True
# )
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from packages.BaselineCorrection import *
import plotly.graph_objs as go
from packages.utils import require_state

if not require_state(
    [
        "compiled_data",
        "unique_wavenumbers",
        "baseline_range_indices",
        "mass_range_indices",
        "x_mass",
        "complex",
        "mass_complex",
    ],
    section="2.1 Baseline correction - full range",
    hint="Run 2.0 Baseline correction first to register baseline parameters.",
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
baseline_reference = st.session_state.get("baseline_reference", None)
baseline_width = st.session_state.get("baseline_width", None)
complex = st.session_state.get("complex", None)
mass_complex = st.session_state.get("mass_complex", None)
mass_range_indices = st.session_state.get("mass_range_indices", None)
baseline_range_indices = st.session_state.get("baseline_range_indices", None)
check_wavenumber = st.session_state.get("check_wavenumber", None)

if st.button("**:blue[#1]** ✨ Perform baseline correction - full range"):
    baseline_method = st.session_state.get("baseline_method", "Mean Subtraction")
    compilation_baseline_corrected_data = {}
    
    for wavenumber in unique_wavenumbers:
        # Sum every other column to obtain total signals
        sum_withoutIR = compiled_data[wavenumber].iloc[:, 0::2].sum(axis=1)
        sum_withIR = compiled_data[wavenumber].iloc[:, 1::2].sum(axis=1)
        
        if baseline_method == "Mean Subtraction":
            mean_value_withoutIR = np.mean(sum_withoutIR[baseline_range_indices])
            mean_value_withIR = np.mean(sum_withIR[baseline_range_indices])
            corrected_without = sum_withoutIR - mean_value_withoutIR
            corrected_with = sum_withIR - mean_value_withIR
        elif baseline_method.lower() in ["airpls", "arpls"]:
            from pybaselines import Baseline
            baseline_fitter = Baseline(x_data=x_mass)
            # Removed 'itermax' argument here
            baseline_without, _ = baseline_fitter.arpls(sum_withoutIR, lam=100)
            baseline_with, _ = baseline_fitter.arpls(sum_withIR, lam=100)
            corrected_without = sum_withoutIR - baseline_without
            corrected_with = sum_withIR - baseline_with
        elif baseline_method.upper() == "ASLS":
            from pybaselines import Baseline
            baseline_fitter = Baseline(x_data=x_mass)
            baseline_without, _ = baseline_fitter.asls(sum_withoutIR, lam=1e7, p=0.02)
            baseline_with, _ = baseline_fitter.asls(sum_withIR, lam=1e7, p=0.02)
            corrected_without = sum_withoutIR - baseline_without
            corrected_with = sum_withIR - baseline_with
        elif baseline_method.upper() == "RUBBERBAND":
            from pybaselines import Baseline
            baseline_fitter = Baseline(x_data=x_mass)
            baseline_without, _ = baseline_fitter.rubberband(sum_withoutIR)
            baseline_with, _ = baseline_fitter.rubberband(sum_withIR)
            corrected_without = sum_withoutIR - baseline_without
            corrected_with = sum_withIR - baseline_with
        else:
            raise ValueError("Unsupported baseline method")
        
        new_table = pd.DataFrame({
            "sum_" + str(wavenumber) + "_withoutIR": sum_withoutIR,
            "sum_" + str(wavenumber) + "_withIR": sum_withIR,
            "sum_baseline_corrected_" + str(wavenumber) + "_withoutIR": corrected_without,
            "sum_baseline_corrected_" + str(wavenumber) + "_withIR": corrected_with
        })
        compilation_baseline_corrected_data[wavenumber] = pd.concat([compiled_data[wavenumber], new_table], axis=1)
    
    st.session_state["compilation_baseline_corrected_data"] = compilation_baseline_corrected_data
    st.write("Success! 😎")
    print(compilation_baseline_corrected_data[check_wavenumber].head())



st.markdown("#### Plot parameters")

col1, col2, col3, col4, col5 = st.columns([0.4, 0.05, 0.7, 0.05, 1])
with col1:
    available_wavenumbers = sorted(compiled_data.keys())
    default_idx = 0
    if (
        "check_wavenumber" in st.session_state
        and st.session_state["check_wavenumber"] in available_wavenumbers
    ):
        default_idx = available_wavenumbers.index(st.session_state["check_wavenumber"])
    st.session_state["check_wavenumber"] = st.selectbox(
        "Wavenumber to check plots",
        options=available_wavenumbers,
        index=default_idx,
        format_func=lambda x: f"{x:.2f}",
    )
    st.session_state["y_max2"] = float(
        st.text_input("Maximum y-value", value=st.session_state.get("y_max2", 0.2))
    )
with col3:
    st.session_state["x_min2"] = float(
        st.text_input("Minimum x-value", value=st.session_state.get("x_min2", 0.0))
    )
    st.session_state["x_max2"] = float(
        st.text_input("Maximum x-value", value=st.session_state.get("x_max2", 1300))
    )
with col5:
    st.session_state["plot_columnIndex_withoutIR"] = int(
        st.number_input("Column index for signal without IR irradiation", value=st.session_state.get("plot_columnIndex_withoutIR", -2))
    )
    st.session_state["plot_columnIndex_withIR"] = int(
        st.number_input("Column index for signal with IR irradiation", value=st.session_state.get("plot_columnIndex_withIR", -1))
    )

if st.button("**:blue[#2]** 🚀 Plot full range data"):
    # Retrieve check_wavenumber from session state
    check_wavenumber = st.session_state.get("check_wavenumber", None)

    # If the value is None or not present in compiled_data, use the first available key
    if check_wavenumber is None or check_wavenumber not in compiled_data:
        st.write("Available compiled_data keys:", list(compiled_data.keys()))
        st.write("Provided check_wavenumber:", check_wavenumber)
        check_wavenumber = list(compiled_data.keys())[0]  # Use first key as fallback

    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
    y_max2 = st.session_state.get("y_max2", None)
    x_min2 = st.session_state.get("x_min2", None)
    x_max2 = st.session_state.get("x_max2", None)

    tab_mass, tab_depletion, tab_ln = st.tabs(
        ["📈 Mass spectra", "💥 Depletion", "📉 −ln(Depletion)"]
    )

    with tab_mass:
        fig_mass = go.Figure()
        fig_mass.add_trace(
            go.Scatter(
                x=x_mass[:],
                y=compilation_baseline_corrected_data[check_wavenumber].iloc[
                    :, plot_columnIndex_withoutIR
                ],
                mode="lines",
                name=compilation_baseline_corrected_data[check_wavenumber].columns[
                    plot_columnIndex_withoutIR
                ],
                line=dict(color="#1f77b4"),
                opacity=0.75,
            )
        )
        fig_mass.add_trace(
            go.Scatter(
                x=x_mass[:],
                y=compilation_baseline_corrected_data[check_wavenumber].iloc[
                    :, plot_columnIndex_withIR
                ],
                mode="lines",
                name=compilation_baseline_corrected_data[check_wavenumber].columns[
                    plot_columnIndex_withIR
                ],
                line=dict(color="#ff7f0e"),
                opacity=0.75,
            )
        )
        fig_mass.add_trace(
            go.Scatter(
                x=[mass_complex, mass_complex],
                y=[-0.001, y_max2],
                mode="lines",
                line=dict(color="green", width=1, dash="solid"),
                name=str(complex),
            )
        )
        fig_mass.add_shape(
            type="rect",
            x0=min(x_mass[baseline_range_indices]),
            y0=0,
            x1=max(x_mass[baseline_range_indices]),
            y1=y_max2,
            fillcolor="lightsteelblue",
            line=dict(color="rgba(0,0,0,0)"),
            opacity=0.4,
            layer="below",
        )
        fig_mass.add_trace(
            go.Scatter(
                x=[x_mass[mass_range_indices][0], x_mass[mass_range_indices][-1]],
                y=[0, 0],
                mode="lines",
                line=dict(color="lime", width=1),
                name="zero line",
            )
        )
        fig_mass.update_layout(
            yaxis=dict(range=[-0.001, y_max2]),
            xaxis_title="Mass (amu)",
            yaxis_title="Intensity",
            xaxis=dict(range=[x_min2, x_max2]),
            title=f"{complex} – signals",
            legend=dict(x=0.8, y=0.9),
        )
        st.plotly_chart(fig_mass, use_container_width=True)

    with tab_depletion:
        fig_dep = go.Figure()
        fig_dep.add_trace(
            go.Scatter(
                x=st.session_state.fullrange_depletion_data.iloc[:, 0],
                y=st.session_state.fullrange_depletion_data.iloc[:, 3],
                mode="lines",
                name="Depletion",
            )
        )
        fig_dep.add_trace(
            go.Scatter(
                x=[st.session_state["depletion_xmin"], st.session_state["depletion_xmax"]],
                y=[0, 0],
                mode="lines",
                line=dict(color="lime", width=1),
                name="zero line",
            )
        )
        fig_dep.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Depletion",
            title="Full-range depletion",
        )
        st.plotly_chart(fig_dep, use_container_width=True)

    with tab_ln:
        fig_ln = go.Figure()
        fig_ln.add_trace(
            go.Scatter(
                x=st.session_state.fullrange_depletion_data.iloc[:, 0],
                y=st.session_state.fullrange_depletion_data.iloc[:, 4],
                mode="lines",
                name="-ln(Depletion)",
            )
        )
        fig_ln.add_trace(
            go.Scatter(
                x=[st.session_state["depletion_xmin"], st.session_state["depletion_xmax"]],
                y=[0, 0],
                mode="lines",
                line=dict(color="lime", width=1),
                name="zero line",
            )
        )
        fig_ln.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="-ln(Depletion)",
            title="Full-range -ln(Depletion)",
        )
        st.plotly_chart(fig_ln, use_container_width=True)
