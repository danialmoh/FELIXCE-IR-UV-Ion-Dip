"""
REMPI Wavelengths Page - Step 2

This page displays the wavelengths per file for REMPI data.
Similar to 1.1_Wavenumbers_raw.py but for REMPI wavelengths.
"""

import streamlit as st
import os

# Check if data is loaded
if "rempi_file_directory" not in st.session_state:
    st.error("Please import REMPI data first (Step 8.0)", icon="🚫")
    st.stop()

file_directory = st.session_state["rempi_file_directory"]

# Try to read HTML file
html_file_path = os.path.join(file_directory, "output", "TableWavelengthsCheck_REMPI.html")

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
    
    if "rempi_data" in st.session_state:
        rempi_data = st.session_state["rempi_data"]
        
        st.write("### Wavelengths per File")
        for i, file_data in enumerate(rempi_data.data):
            st.write(f"**File {i+1}:** {file_data.wavelengths}")
    else:
        st.error("No REMPI data found in memory. Please import data first.")
