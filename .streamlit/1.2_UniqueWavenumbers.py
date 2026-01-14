from pathlib import Path

import streamlit as st

from packages.utils import require_state


if not require_state(["file_directory"], section="1.2 Unique wavenumbers"):
    st.stop()

file_directory = Path(st.session_state["file_directory"]).expanduser()
html_file_path = file_directory / "output" / "UniqueWavenumbers.html"

if not html_file_path.exists():
    st.info(
        "No unique-wavenumber report found. Generate it from the import workflow first.",
        icon="ℹ️",
    )
else:
    html_content = html_file_path.read_text(encoding="utf-8")
    st.components.v1.html(html_content, height=700, width=1000, scrolling=True)
