"""
Report Manager Module

This module provides utilities for managing report items across the Streamlit app.
It allows users to add plots and data to a report collection that can be exported.
"""

import streamlit as st
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

__all__ = ['ReportManager', 'add_plot_to_report_button', 'init_report_session']


def init_report_session():
    """Initialize report-related session state variables"""
    if "report_plots" not in st.session_state:
        st.session_state["report_plots"] = []
    if "report_data" not in st.session_state:
        st.session_state["report_data"] = {}
    if "report_metadata" not in st.session_state:
        st.session_state["report_metadata"] = {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": []
        }


def add_plot_to_report_button(fig, title, key_suffix="", description=""):
    """
    Add a checkbox toggle that saves a matplotlib figure to the report collection.
    Wrapped in @st.fragment so toggling does NOT rerun the full page (no scroll jump).
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    title : str
        Title for the plot in the report
    key_suffix : str
        Unique suffix for the button key
    description : str
        Optional description for the plot
    """
    init_report_session()
    
    # Pre-save the figure bytes into session state so they survive fragment reruns
    safe_title = title.replace(' ', '_').replace('/', '-').replace('.', '_')
    image_cache_key = f"_report_img_{safe_title}_{key_suffix}"
    
    if image_cache_key not in st.session_state:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        st.session_state[image_cache_key] = buf.getvalue()
    
    # Call the fragment
    _report_toggle_fragment(title, key_suffix, description, image_cache_key)


@st.fragment
def _report_toggle_fragment(title, key_suffix, description, image_cache_key):
    """Fragment that renders the checkbox. Only this reruns on toggle, not the whole page."""
    safe_title = title.replace(' ', '_').replace('/', '-').replace('.', '_')
    checkbox_key = f"report_cb_{safe_title}_{key_suffix}"
    
    existing_titles = [p["title"] for p in st.session_state["report_plots"]]
    is_in_report = title in existing_titles
    
    add_to_report = st.checkbox(
        "✅ In Report" if is_in_report else "📎 Add to Report",
        value=is_in_report,
        key=checkbox_key,
        help=f"Toggle to add/remove '{title}' from the report"
    )
    
    if add_to_report and not is_in_report:
        plot_entry = {
            "title": title,
            "description": description,
            "image_bytes": st.session_state[image_cache_key],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "page": st.session_state.get("current_page", "Unknown")
        }
        st.session_state["report_plots"].append(plot_entry)
        st.toast(f"✅ Added '{title}' to report!")
    elif not add_to_report and is_in_report:
        st.session_state["report_plots"] = [
            p for p in st.session_state["report_plots"] if p["title"] != title
        ]
        st.toast(f"Removed '{title}' from report")


def add_data_to_report(data, name, description=""):
    """
    Add a DataFrame or dict to the report data collection.
    
    Parameters:
    -----------
    data : pd.DataFrame or dict
        The data to save
    name : str
        Name/key for the data
    description : str
        Optional description
    """
    init_report_session()
    
    st.session_state["report_data"][name] = {
        "data": data,
        "description": description,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


class ReportManager:
    """
    Class to manage report generation and export.
    """
    
    def __init__(self):
        init_report_session()
    
    @staticmethod
    def get_plot_count():
        """Return number of plots in report"""
        return len(st.session_state.get("report_plots", []))
    
    @staticmethod
    def get_data_count():
        """Return number of data items in report"""
        return len(st.session_state.get("report_data", {}))
    
    @staticmethod
    def clear_report():
        """Clear all report items"""
        st.session_state["report_plots"] = []
        st.session_state["report_data"] = {}
        st.session_state["report_metadata"] = {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": []
        }
    
    @staticmethod
    def add_note(note):
        """Add a note to the report"""
        init_report_session()
        st.session_state["report_metadata"]["notes"].append({
            "text": note,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    @staticmethod
    def get_plots():
        """Return list of plots in report"""
        return st.session_state.get("report_plots", [])
    
    @staticmethod
    def get_data():
        """Return dict of data items in report"""
        return st.session_state.get("report_data", {})
    
    @staticmethod
    def remove_plot(title):
        """Remove a plot by title"""
        plots = st.session_state.get("report_plots", [])
        st.session_state["report_plots"] = [p for p in plots if p["title"] != title]
    
    @staticmethod
    def remove_data(name):
        """Remove a data item by name"""
        if name in st.session_state.get("report_data", {}):
            del st.session_state["report_data"][name]
