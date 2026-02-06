"""
REMPI Unique Wavelengths Page - Step 3

This page displays unique wavelengths and allows filtering by count.
Similar to 1.3_UniqueWavenumbers.py but for REMPI wavelengths.
"""

import streamlit as st
import os

# Check if data is loaded
if "rempi_file_directory" not in st.session_state:
    st.error("Please import REMPI data first (Step 8.0)", icon="🚫")
    st.stop()

if "rempi_data" not in st.session_state:
    st.error("Please import REMPI data first (Step 8.0)", icon="🚫")
    st.stop()

file_directory = st.session_state["rempi_file_directory"]
rempi_data = st.session_state["rempi_data"]

# Filter by count
st.markdown("##### Drop wavelengths with counts ≤")
wavelength_min_count = st.number_input(
    "Drop wavelengths with counts below and including:", 
    label_visibility="collapsed", 
    value=0, 
    min_value=0
)

if st.button("Apply count filter"):
    unique_wavelengths, unique_wavelengths_df = rempi_data.get_wavelengths(min_count=wavelength_min_count)
    st.session_state["rempi_unique_wavelengths"] = unique_wavelengths
    st.session_state["rempi_unique_wavelengths_df"] = unique_wavelengths_df
    st.success(f"Dropped wavelengths with counts ≤ {wavelength_min_count} 💧.")

# Try to read HTML file
html_file_path = os.path.join(file_directory, "output", "UniqueWavelengths_REMPI.html")

if os.path.exists(html_file_path):
    with open(html_file_path, 'r') as file:
        html_content = file.read()

    # Add Consolas font styling
    font_style = """
    <style>
    body, table, th, td {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    }
    th {
        font-weight: bold;
    }
    td {
        font-weight: normal;
    }
    </style>
    """

    if "<head>" in html_content:
        html_content = html_content.replace("<head>", f"<head>{font_style}")
    else:
        html_content = f"{font_style}{html_content}"

    st.components.v1.html(html_content, height=700, width=1000, scrolling=True)
else:
    # Fallback: show data from session state
    st.warning("HTML file not found. Showing data from memory.")
    
    if "rempi_unique_wavelengths_df" in st.session_state:
        st.dataframe(st.session_state["rempi_unique_wavelengths_df"])
    else:
        st.error("No unique wavelengths data found. Please import data first.")
