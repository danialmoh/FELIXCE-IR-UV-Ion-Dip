import streamlit as st
import os

# import variables
if "file_directory" in st.session_state:
    file_directory = st.session_state["file_directory"]


# Read HTML file
html_file_path = os.path.join(file_directory, "Table_Wavenumbers_raw.html")
with open(html_file_path, 'r') as file:
    html_content = file.read()

# Add Consolas font styling to the HTML content
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

# Insert the style into the HTML content
if "<head>" in html_content:
    html_content = html_content.replace("<head>", f"<head>{font_style}")
else:
    # If no head tag, add style at the beginning
    html_content = f"{font_style}{html_content}"

# Display HTML content
st.components.v1.html(html_content, height=700, width=1000, scrolling=True)