import streamlit as st
import numpy as np


col1,col2,col3 = st.columns([0.5,1,1]) # col2 is just for spacing

with col1:
    # Species parameters #"What are the elements involved in your experiment?"
    st.markdown("### Species")
    st.session_state["element1"] = st.text_input("Element 1", value = st.session_state.get("element1", "C"))
    st.session_state["element2"] = st.text_input("Element 2", value = st.session_state.get("element2", "H"))
    st.session_state["element3"] = st.text_input("Element 3", value = st.session_state.get("element3", "Br"))
    
with col2:#These inputs are converted to float (for numerical calculations) and stored in session_state.
    st.markdown("### Parameters")
    st.session_state["mass_element1"] = float(st.text_input("Mass of element1 in amu", value = st.session_state.get("mass_element1",12.000)))
    st.session_state["mass_element2"] = float(st.text_input("Mass of element2 in amu", value = st.session_state.get("mass_element2",1.02)))
    st.session_state["mass_element3"] = float(st.text_input("Mass of element3 in amu", value = st.session_state.get("mass_element3",78.918336)))
    st.session_state["charge_state"] = st.text_input("Charge state", st.session_state.get("charge_state", "0"))

with col3:
    # Calibration parameters
    st.markdown("### Calibration parameters")
    st.session_state["t_off"] = st.number_input("t_off", value = st.session_state.get("t_off", 58))
    st.session_state["alpha"] = float(st.text_input("alpha", value = st.session_state.get("alpha", 7.6987e-7)))
    st.session_state["counts"] = st.number_input("length of dataset in the time axis", value = st.session_state.get("counts", 60000))


if st.button("✍️ Register inputs"):

    # Initialize variables
    mass_element = st.session_state.get("mass_element", None)
    t_off = st.session_state.get("t_off", None)
    alpha = st.session_state.get("alpha", None)
    counts = st.session_state.get("counts", None)                             

    # Generate an x-axis
    x_counts=np.linspace(1,counts,counts)
    # Calibrate spectra
    x_mass = alpha*(x_counts - t_off)**2
    # x_mass_perAtom = alpha*(x_counts - t_off)**2 / mass_element
    x_mass_perAtom = 0

    # Save the variables into memory
    st.session_state["x_mass"] = x_mass
    st.session_state["x_mass_perAtom"] = x_mass_perAtom
    st.write("Inputs registered! 😊")
