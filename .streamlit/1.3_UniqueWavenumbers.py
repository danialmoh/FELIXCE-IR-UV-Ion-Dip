import streamlit as st
import os

# import variables
if "file_directory" in st.session_state:
    file_directory = st.session_state["file_directory"]
    raw_data = st.session_state["raw_data"]


st.markdown("##### Drop wavenumbers with counts \u2264")
wavenumber_min_count = st.number_input("Drop wavenumbers with counts below and including:", label_visibility="collapsed", value=0, min_value=0)

st.markdown("##### Keep only wavenumbers with counts \u2264 (0 = no limit)")
wavenumber_max_count = st.number_input("Keep wavenumbers with counts up to:", label_visibility="collapsed", value=0, min_value=0)
if st.button("Apply count filter"):
    # Get total count before filtering
    total_before = len(raw_data.unique_wavenumbers) if hasattr(raw_data, 'unique_wavenumbers') and raw_data.unique_wavenumbers is not None else None
    
    _max_ct = wavenumber_max_count if wavenumber_max_count > 0 else None
    unique_wavenumbers = raw_data.visualize_imported_unique_wavenumbers(wavenumber_min_count, max_count=_max_ct)
    st.session_state["unique_wavenumbers"] = unique_wavenumbers
    
    # Track how many were dropped
    if total_before is not None:
        neglected = total_before - len(unique_wavenumbers)
    else:
        neglected = 0
    st.session_state["neglected_wavenumbers_count"] = neglected
    st.session_state["wavenumber_min_count_filter"] = wavenumber_min_count
    
    _msg = f"Dropped {neglected} wavenumbers with counts \u2264 {wavenumber_min_count}"
    if _max_ct is not None:
        _msg += f", and those with counts > {wavenumber_max_count}"
    st.success(_msg + " \U0001f4a7.")


    
    


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