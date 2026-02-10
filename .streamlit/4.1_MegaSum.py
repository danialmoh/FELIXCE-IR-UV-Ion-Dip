# import streamlit as st
# import numpy as np
# import pandas as pd
# from packages.FELIX_HDF5_ReadData import *
# from packages.FELIX_HDF5_ProcessData import *
# from packages.BaselineCorrection import *   
# import matplotlib.pyplot as plt
# import plotly.graph_objs as go
# import os

# # Import variables from session_state
# file_directory = st.session_state.get("file_directory", None)
# compiled_data = st.session_state.get("compiled_data", None)
# unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
# x_mass = st.session_state.get("x_mass", None)
# baseline_range_indices = st.session_state.get("baseline_range_indices", None)

# st.markdown("##### :red[Caution]: You need to run the program all the way to section 2.0 first.")

# if st.button("**:blue[#1]** ➕ Create a sum of all mass spectra + baseline correction! (wavenumber independent)"):
#     # Initialize variables
#     baseline_withoutIR = 0
#     baseline_withIR = 0

#     # Use a list to accumulate DataFrames instead of concatenating repeatedly
#     table_list = []
#     for wavenumber in unique_wavenumbers:
#         # Optionally, reduce memory usage by converting dtypes if applicable:
#         df = compiled_data[wavenumber]
#         # e.g., df = df.astype('float32')  # if your data is originally float64
#         table_list.append(df)
    
#     # Concatenate once after the loop
#     # MegaTable = pd.concat(table_list, axis=1)

#     # Sum the signals without and with IR irradiation
#     signal_withoutIR = MegaTable.iloc[:, 0::2].sum(axis=1)
#     signal_withIR = MegaTable.iloc[:, 1::2].sum(axis=1)
#     # Initialize or update the running sums
#     # if signal_withoutIR is None:
#     #     signal_withoutIR = current_withoutIR.copy()
#     #     signal_withIR = current_withIR.copy()
#     # else:
#     #     signal_withoutIR += current_withoutIR
#     #     signal_withIR += current_withIR
    
#     # Baseline correction
#     baseline_withoutIR = np.mean(signal_withoutIR[baseline_range_indices])
#     baseline_withIR = np.mean(signal_withIR[baseline_range_indices])

#     new_table = pd.DataFrame({
#         "signal_withoutIR": signal_withoutIR,
#         "signal_withIR": signal_withIR,
#         "baseline_corrected_signal_withoutIR": signal_withoutIR - baseline_withoutIR,
#         "baseline_corrected_signal_withIR": signal_withIR - baseline_withIR
#     })

#     # Optionally, if you still need the full MegaTable, consider if you can build it incrementally
#     # st.session_state["MegaSum"] = new_table  # Save the smaller, processed DataFrame
#     # st.write("MegaSum created successfully!")
#     # Save the MegaSum DataFrame in session_state
#     st.session_state["MegaSum"] = pd.concat([MegaTable, new_table], axis=1)
#     st.write("MegaSum created successfully!")



import streamlit as st
import numpy as np
import pandas as pd
from packages.FELIX_HDF5_ReadData import *
from packages.FELIX_HDF5_ProcessData import *
from packages.BaselineCorrection import *   
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from packages.ReportManager import add_plot_to_report_button, init_report_session

init_report_session()

# Import variables
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
baseline_range_indices = st.session_state.get("baseline_range_indices", None)


st.markdown("##### :red[Caution]: You need to run the program all the way to section 2.0 first.")

if st.button("**:blue[#1]** ➕ Create a sum of all mass spectra + baseline correction! (wavenumber independent)"):
    # Initialize variables
    MegaSum = {}
    MegaTable = pd.DataFrame()
    NewTable = pd.DataFrame()
    baseline_withoutIR = 0
    baseline_withIR = 0

    # assemble everything into a gigantic table
    for wavenumber in unique_wavenumbers:
        MegaTable = pd.concat([MegaTable, compiled_data[wavenumber]],axis=1)

    # sum all the signals without and with IR irradiation
    signal_withoutIR = MegaTable.iloc[: ,0: :2].sum(axis=1)
    signal_withIR = MegaTable.iloc[:, 1: :2].sum(axis=1)

    # baseline correction
    # Check if baseline_range_indices exists and is valid
    if baseline_range_indices is None:
        st.error("baseline_range_indices not found in session state. Please run baseline correction (section 2.0) first.")
        st.stop()
    
    # Ensure baseline_range_indices is integer type for .iloc indexing
    baseline_indices = np.asarray(baseline_range_indices, dtype=int)
    baseline_withoutIR = np.mean(signal_withoutIR.iloc[baseline_indices])
    baseline_withIR = np.mean(signal_withIR.iloc[baseline_indices])


    new_table = pd.DataFrame({
        "signal_withoutIR": signal_withoutIR,
        "signal_withIR": signal_withIR,
        "baseline_corrected_signal_withoutIR": signal_withoutIR - baseline_withoutIR,
        "baseline_corrected_signal_withIR": signal_withIR - baseline_withIR
    })

    # create the mega sum table
    st.session_state["MegaSum"] = pd.concat([MegaTable, new_table],axis=1)
    MegaSum = st.session_state.get("MegaSum", None)
    st.write("Success MegaSUM! 😎")
    
    # Comment out the line below if you're encountereing the same issue as section 2.1
    # st.write(MegaSum)
    # Uncomment the line below if you want to see the "MegaSum" table on your terminal
    # print(MegaSum)


if st.button("**:blue[#3]** 📈 Plot Interactive mega sum!"):
    # Retrieve necessary variables from session_state
    x_mass = st.session_state.get("x_mass", None)
    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    MegaSum_xmin = st.session_state.get("MegaSum_xmin", None)
    MegaSum_xmax = st.session_state.get("MegaSum_xmax", None)
    MegaSum_ymin = st.session_state.get("MegaSum_ymin", None)
    MegaSum_ymax = st.session_state.get("MegaSum_ymax", None)
    MegaSum = st.session_state.get("MegaSum", None)
    
    # --- Save the MegaSum Data ---
    # file_directory = st.session_state.get("file_directory", ".")
    # save_path = os.path.join(file_directory, "MegaSum.csv")
    # MegaSum.to_csv(save_path, index=False)
    # st.write(f"MegaSum data saved to: **{save_path}st**")
    
    # --- Create Interactive Plot with Plotly ---
    st.markdown("###### *:green[Interactive plot with Plotly]*")
    fig = go.Figure()
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=x_mass[0],
        x1=x_mass[-1],
        y0=0,
        y1=0,
        line=dict(color="limegreen"),
        layer="below",
        name="Zero line"
    )
    
    # Plot the signal without IR irradiation
    fig.add_trace(go.Scatter(
        x=x_mass,
        y=MegaSum.iloc[:, plot_columnIndex_withoutIR],
        mode='lines',
        name=MegaSum.columns[plot_columnIndex_withoutIR]
    ))
    
    # Set axes limits and labels
    fig.update_xaxes(range=[MegaSum_xmin, MegaSum_xmax], title="Mass (amu)")
    fig.update_yaxes(range=[MegaSum_ymin, MegaSum_ymax], title="Intensity")
    
    fig.update_layout(
        title="Interactive MegaSum Plot",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig)

# st.markdown("#### Plot parameters")
# col1,col2,col3,col4 = st.columns([1,1,1,1])

# with col1:
#     print("hi")
#     # st.session_state["max_element"] = int(st.number_input(f"Maximum size for {element}", value = st.session_state.get("max_element", 20)))
#     # st.session_state["max_messenger"] = int(st.number_input(f"Maximum size for {messenger}", value = st.session_state.get("max_messenger", 7)))

# with col2:
#     st.session_state["plot_columnIndex_withoutIR"] = int(st.number_input("Column index for signal without IR irradiation", value = st.session_state.get("plot_columnIndex_withoutIR", -2)))
#     st.session_state["plot_columnIndex_withIR"] = int(st.number_input("Column index for signal with IR irradiation", value = st.session_state.get("plot_columnIndex_withIR", -1)))

# with col3:
#     st.session_state["MegaSum_xmin"] = float(st.text_input("Minimum x-value", value = st.session_state.get("MegaSum_xmin", 0.0)))
#     st.session_state["MegaSum_xmax"] = float(st.text_input("Maximum x-value", value = st.session_state.get("MegaSum_xmax", 1300)))

# with col4:
#     st.session_state["MegaSum_ymin"] = float(st.text_input("Minimum y-value", value = st.session_state.get("MegaSum_ymin",-0.001)))
#     st.session_state["MegaSum_ymax"] = float(st.text_input("Maximum y-value", value = st.session_state.get("MegaSum_ymax",0.1))) 
    


# if st.button("**:blue[#2]** 📈 Plot mega sum!"):

#     x_mass = st.session_state.get("x_mass", None)
#     plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
#     plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
#     MegaSum_xmin = st.session_state.get("MegaSum_xmin", None)
#     MegaSum_xmax = st.session_state.get("MegaSum_xmax", None)
#     MegaSum_ymin = st.session_state.get("MegaSum_ymin", None)
#     MegaSum_ymax = st.session_state.get("MegaSum_ymax", None)
#     MegaSum = st.session_state.get("MegaSum", None)

#     st.markdown("###### *:green[Static plot with matplotlib]*")
#     fig, ax = plt.subplots()
#     ax.axhline(0, color = "limegreen")
#     # ax.fill_between(x_mass[baseline_range_indices],0.2, color = "lightsteelblue", label = "baseline range")
#     ax.plot(x_mass[:], MegaSum.iloc[:,plot_columnIndex_withoutIR], label = MegaSum.columns[plot_columnIndex_withoutIR])
#     # ax.plot(x_mass[:], MegaSum.iloc[:,plot_columnIndex_withIR], label = MegaSum.columns[plot_columnIndex_withIR])
#     ax.set_xlim(MegaSum_xmin, MegaSum_xmax)
#     ax.set_ylim(MegaSum_ymin, MegaSum_ymax)
#     ax.set_xlabel("Mass (amu)")
#     ax.set_ylabel("Intensity")
#     ax.legend(fontsize=5)
#     fig.tight_layout()

#     st.pyplot(fig)



#     # # initialize variables
#     # x_mass = st.session_state.get("x_mass", None)
#     # plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
#     # plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
#     # MegaSum_xmin = st.session_state.get("MegaSum_xmin", None)
#     # MegaSum_xmax = st.session_state.get("MegaSum_xmax", None)
#     # MegaSum_ymin = st.session_state.get("MegaSum_ymin", None)
#     # MegaSum_ymax = st.session_state.get("MegaSum_ymax", None)
#     # MegaSum = st.session_state.get("MegaSum", None)
#     # element = st.session_state.get("element", None)
#     # mass_element = st.session_state.get("mass_element", None)
#     # charge_state = st.session_state.get("charge_state", None)
#     # messenger = st.session_state.get("messenger", None)
#     # mass_messenger = st.session_state.get("mass_messenger", None)
#     # max_element = st.session_state.get("max_element", None)
#     # max_messenger = st.session_state.get("max_messenger", None)


#     # st.markdown("###### *:green[Interactive plot with plotly]*")
#     # # Create a Plotly figure
#     # fig = go.Figure()

#     # # Add horizontal line at y=0
#     # fig.add_shape(
#     #     type='line',
#     #     x0=x_mass[0], 
#     #     x1=x_mass[-1], 
#     #     y0=0, 
#     #     y1=0,
#     #     line=dict(color="limegreen"),
#     #     layer = "below",
#     #     name = "zero line"
#     # )

#     # # Plot the baseline range
#     # fig.add_trace(go.Scatter(
#     #     x=x_mass[baseline_range_indices],
#     #     y=[(MegaSum_ymax/2)] * len(baseline_range_indices),
#     #     fill='tozeroy',
#     #     mode='lines',
#     #     fillcolor='lightsteelblue',
#     #     line=dict(color='lightsteelblue'),
#     #     name='baseline range',
#     #     opacity = 0.5
#     # ))

#     # # Plot the mass spectra for the two specified columns
#     # fig.add_trace(go.Scatter(
#     #     x=x_mass,
#     #     y=MegaSum.iloc[:, plot_columnIndex_withoutIR],
#     #     mode='lines',
#     #     name=MegaSum.columns[plot_columnIndex_withoutIR]
#     # ))

#     # fig.add_trace(go.Scatter(
#     #     x=x_mass,
#     #     y=MegaSum.iloc[:, plot_columnIndex_withIR],
#     #     mode='lines',
#     #     name=MegaSum.columns[plot_columnIndex_withIR]
#     # ))

#     # # Main element annotations
#     # for i in range(1, max_element + 1):
#     #     n = i
#     #     m = 0
#     #     complex, mass_complex, _ = mass_range(n, m, element, messenger, mass_element, mass_messenger, charge_state, x_mass)
#     #     mass_index = np.where((x_mass >= mass_complex - 2) & (x_mass <= mass_complex + 2))[0]
#     #     y1 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withoutIR])
#     #     y2 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withIR])
#     #     y = max(y1, y2)

#     #     fig.add_annotation(
#     #         x=mass_complex,
#     #         y=y + 3,
#     #         text=f"({n},{m})",
#     #         showarrow=False,
#     #         arrowhead=2,
#     #         ax=0,
#     #         ay=-(y+100),
#     #         font=dict(size=20, color="red"),
#     #         textangle=90,
#     #     )

#     # # Complex annotations
#     # for i in range(1, max_element + 1):
#     #     for j in range(1, max_messenger + 1):
#     #         n = i
#     #         m = j
#     #         complex, mass_complex, _ = mass_range(n, m, element, messenger, mass_element, mass_messenger, charge_state, x_mass)
#     #         mass_index = np.where((x_mass >= mass_complex - 2) & (x_mass <= mass_complex + 2))[0]
#     #         y1 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withoutIR])
#     #         y2 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withIR])
#     #         y = max(y1, y2)

#     #         fig.add_annotation(
#     #             x=mass_complex,
#     #             y=y + (0.5*i),
#     #             text=f"({n},{m})",
#     #             showarrow=False,
#     #             arrowhead=2,
#     #             ax=0,
#     #             ay=-(y + (10 * i)),
#     #             font=dict(size=18),
#     #             textangle=90,
#     #         )

#     # # Set the x and y limits
#     # fig.update_xaxes(range=[MegaSum_xmin, MegaSum_xmax])
#     # fig.update_yaxes(range=[MegaSum_ymin, MegaSum_ymax])

#     # # Set the axis labels
#     # fig.update_layout(
#     #     xaxis_title="Mass (amu)",
#     #     yaxis_title="Intensity",
#     #     legend=dict(
#     #         font=dict(size=14),  # Font size of the legend
#     #         orientation='v',     # Set legend orientation to horizontal
#     #         yanchor='top',    # Anchor the legend to the bottom
#     #         xanchor='right',    # Center the legend horizontally
#     #     )
#     # )

#     # # Display the Plotly figure in Streamlit
#     # st.plotly_chart(fig)


#     # st.markdown("###### *:green[Static plot with matplotlib]*")
#     # fig, ax = plt.subplots()

#     # # plot the mass spectra
#     # ax.axhline(0, color = "limegreen")
#     # ax.fill_between(x_mass[baseline_range_indices],0.2, color = "lightsteelblue", label = "baseline range")
#     # ax.plot(x_mass[:], MegaSum.iloc[:,plot_columnIndex_withoutIR], label = MegaSum.columns[plot_columnIndex_withoutIR])
#     # ax.plot(x_mass[:], MegaSum.iloc[:,plot_columnIndex_withIR], label = MegaSum.columns[plot_columnIndex_withIR])

#     # # combinations of complexes
    
#     # # main element
#     # for i in range(1, max_element+1):
#     #     n = i
#     #     m = 0

#     #     complex, mass_complex, _ = mass_range(n,m,element,messenger,mass_element,mass_messenger, charge_state, x_mass)
#     #     mass_index = np.where((x_mass >= mass_complex -2) & (x_mass <= mass_complex+2))[0]
#     #     y1 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withoutIR])
#     #     y2 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withIR])
#     #     y = max(y1,y2)
#     #     ax.annotate(f"({n},{m})", (mass_complex, y), textcoords="offset points", xytext=(0,20), ha='center', fontsize=10, rotation = 90, color="red")

#     # # complex
#     # for i in range(1,max_element+1):
#     #     for j in range(1, max_messenger+1):
#     #         n=i
#     #         m = j
#     #         complex, mass_complex, _ = mass_range(n,m,element,messenger,mass_element,mass_messenger, charge_state, x_mass)
#     #         mass_index = np.where((x_mass >= mass_complex -2) & (x_mass <= mass_complex+2))[0]
#     #         y1 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withoutIR])
#     #         y2 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withIR])
#     #         y = max(y1,y2)
#     #         ax.annotate(f"({n},{m})", (mass_complex, y), textcoords="offset points", xytext=(0,y+(4*i)), ha='center', fontsize=8, rotation = 90)

#     # ax.set_xlim(MegaSum_xmin, MegaSum_xmax)
#     # ax.set_ylim(MegaSum_ymin, MegaSum_ymax)
#     # ax.set_xlabel("Mass (amu)")
#     # ax.set_ylabel("Intensity")
#     # ax.legend(fontsize=5)
#     # fig.tight_layout()

#     # st.pyplot(fig)
# import os  # Make sure this is at the top of your file along with your other imports

# if st.button("**:blue[#3]** 📈 Plot Interactive mega sum!"):
#     # Retrieve necessary variables from session_state
#     x_mass = st.session_state.get("x_mass", None)
#     plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
#     plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
#     MegaSum_xmin = st.session_state.get("MegaSum_xmin", None)
#     MegaSum_xmax = st.session_state.get("MegaSum_xmax", None)
#     MegaSum_ymin = st.session_state.get("MegaSum_ymin", None)
#     MegaSum_ymax = st.session_state.get("MegaSum_ymax", None)
#     MegaSum = st.session_state.get("MegaSum", None)
    
#     # --- Save the MegaSum Data ---
#     # Use the provided file_directory from session_state (or default to current directory)
#     file_directory = st.session_state.get("file_directory", ".")
#     save_path = os.path.join(file_directory, "MegaSum.csv")
    
#     # Save the DataFrame to CSV
#     MegaSum.to_csv(save_path, index=False)
#     st.write(f"MegaSum data saved to: **{save_path}**")
    
#     # --- Create Interactive Plot with Plotly ---
#     st.markdown("###### *:green[Interactive plot with Plotly]*")
#     fig = go.Figure()
    
#     # Add a horizontal line at y=0
#     fig.add_shape(
#         type='line',
#         x0=x_mass[0],
#         x1=x_mass[-1],
#         y0=0,
#         y1=0,
#         line=dict(color="limegreen"),
#         layer="below",
#         name="Zero line"
#     )
    
#     # Plot the signal without IR irradiation
#     fig.add_trace(go.Scatter(
#         x=x_mass,
#         y=MegaSum.iloc[:, plot_columnIndex_withoutIR],
#         mode='lines',
#         name=MegaSum.columns[plot_columnIndex_withoutIR]
#     ))
    
#     # Optionally, you can also plot the signal with IR irradiation by uncommenting below:
#     # fig.add_trace(go.Scatter(
#     #     x=x_mass,
#     #     y=MegaSum.iloc[:, plot_columnIndex_withIR],
#     #     mode='lines',
#     #     name=MegaSum.columns[plot_columnIndex_withIR]
#     # ))
    
#     # Set x and y axes limits and labels
#     fig.update_xaxes(range=[MegaSum_xmin, MegaSum_xmax], title="Mass (amu)")
#     fig.update_yaxes(range=[MegaSum_ymin, MegaSum_ymax], title="Intensity")
    
#     # Update layout for a nicer look
#     fig.update_layout(
#         title="Interactive MegaSum Plot",
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         )
#     )
    
#     # Render the interactive Plotly plot in Streamlit
#     st.plotly_chart(fig)
