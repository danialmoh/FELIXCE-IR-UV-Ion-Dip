import streamlit as st
import os

# import variables
if "file_directory" in st.session_state:
    file_directory = st.session_state["file_directory"]
    raw_data = st.session_state["raw_data"]


st.markdown("##### Drop wavenumbers with counts ≤")
wavenumber_min_count = st.number_input("Drop wavenumbers with counts below and including:", label_visibility="collapsed", value=0, min_value=0)
if st.button("Apply count filter"):
    unique_wavenumbers = raw_data.visualize_imported_unique_wavenumbers(wavenumber_min_count)
    st.session_state["unique_wavenumbers"] = unique_wavenumbers
    st.success(f"Dropped wavenumbers with counts ≤ {wavenumber_min_count} 💧.")


    
    


# Read HTML file
html_file_path = os.path.join(file_directory, "Table_UniqueWavenumbers.html")
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