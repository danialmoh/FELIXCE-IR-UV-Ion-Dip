"""
Results Summary Page

This page provides a comprehensive summary of the analysis session including:
- Wavenumbers per file (CSV)
- Unique wavenumber parameters
- Baseline corrected mass spectra
- All plots from mass data analysis (mass spectra, depletion, -ln(depletion), intensity vs wavenumber)
- MegaSum analysis
- Peak detection results
- User-added plots via "Add to Report" buttons
- Download as PDF report with parameters, plots, and data summaries
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
from pathlib import Path
import tempfile
import io

# Try to import PDF generation library
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# Initialize report session state
from packages.ReportManager import init_report_session, ReportManager
init_report_session()

st.markdown("# 📋 Results Summary & Export")
st.markdown("---")

# Get current date/time
analysis_date = datetime.now().strftime("%Y-%m-%d")
analysis_time = datetime.now().strftime("%H:%M:%S")

col_date1, col_date2 = st.columns(2)
with col_date1:
    st.markdown(f"**Analysis Date:** {analysis_date}")
with col_date2:
    st.markdown(f"**Analysis Time:** {analysis_time}")

# ============================================================================
# SECTION 1: Report Queue (plots added via "Add to Report" buttons)
# ============================================================================

st.markdown("---")
st.markdown("## 📎 Report Queue")

report_plots = ReportManager.get_plots()
report_data = ReportManager.get_data()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Plots in Queue", len(report_plots))
with col2:
    st.metric("Data Files in Queue", len(report_data))
with col3:
    if st.button("🗑️ Clear Report Queue"):
        ReportManager.clear_report()
        st.rerun()

if report_plots:
    with st.expander(f"📊 View {len(report_plots)} Queued Plots", expanded=False):
        for i, plot in enumerate(report_plots):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{i+1}. {plot['title']}**")
                if plot.get('description'):
                    st.caption(plot['description'])
            with col2:
                if st.button("❌ Remove", key=f"remove_plot_{i}"):
                    ReportManager.remove_plot(plot['title'])
                    st.rerun()

# ============================================================================
# SECTION 2: Session Overview
# ============================================================================

st.markdown("---")
st.markdown("## 📊 Session Data Overview")

# Check what data is available
ir_uv_active = "compiled_data" in st.session_state
has_wavenumbers_table = "wavenumbers_table" in st.session_state
has_unique_wavenumbers = "unique_wavenumbers" in st.session_state
has_baseline_corrected = "compilation_baseline_corrected_data" in st.session_state
has_depletion_data = "fullrange_depletion_data" in st.session_state
has_megasum = "MegaSum" in st.session_state
has_peak_detection = "detected_mz" in st.session_state or "candidates_df" in st.session_state
rempi_active = "rempi_compiled_data" in st.session_state

# Status indicators
status_items = [
    ("Compiled Data", ir_uv_active),
    ("Wavenumbers Table", has_wavenumbers_table),
    ("Unique Wavenumbers", has_unique_wavenumbers),
    ("Baseline Corrected Data", has_baseline_corrected),
    ("Depletion Data", has_depletion_data),
    ("MegaSum", has_megasum),
    ("Peak Detection", has_peak_detection),
    ("REMPI Data", rempi_active),
]

cols = st.columns(4)
for i, (name, available) in enumerate(status_items):
    with cols[i % 4]:
        if available:
            st.success(f"✅ {name}")
        else:
            st.info(f"⏳ {name}")

if not any([ir_uv_active, rempi_active]):
    st.error("No analysis data found. Please run the analysis pipeline first.")
    st.stop()

# ============================================================================
# SECTION 3: Data to Export
# ============================================================================

st.markdown("---")
st.markdown("## 📁 Data Files to Export")

# 2. Unique wavenumber parameters
if has_unique_wavenumbers:
    with st.expander("2️⃣ Unique Wavenumber Parameters", expanded=False):
        unique_wn = st.session_state.get("unique_wavenumbers", [])
        step_size = st.session_state.get("step_size", "N/A")
        min_count_filter = st.session_state.get("wavenumber_min_count_filter", 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Unique Wavenumbers", len(unique_wn))
        with col2:
            st.metric("Step Size", step_size)
        with col3:
            st.metric("Min Count Filter", min_count_filter)
        
        st.dataframe(pd.DataFrame({"Wavenumber (cm⁻¹)": unique_wn}))
        st.session_state["export_unique_wavenumbers"] = True
        st.success("✅ Will be included in export")

# 3. Baseline corrected mass spectra
if has_baseline_corrected:
    with st.expander("3️⃣ Baseline Corrected Mass Spectra (Full Range)", expanded=False):
        bc_data = st.session_state.get("compilation_baseline_corrected_data", {})
        st.write(f"Available wavenumbers with baseline correction: {len(bc_data)}")
        
        # Show sample
        if bc_data:
            sample_wn = list(bc_data.keys())[0]
            st.write(f"Sample data for {sample_wn} cm⁻¹:")
            # Make column names unique before displaying (PyArrow requires unique columns)
            sample_df = bc_data[sample_wn].head(20).copy()
            cols = list(sample_df.columns)
            seen = {}
            unique_cols = []
            for c in cols:
                if c in seen:
                    seen[c] += 1
                    unique_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    unique_cols.append(c)
            sample_df.columns = unique_cols
            st.dataframe(sample_df)
        
        st.session_state["export_baseline_corrected"] = True
        st.success("✅ Will be included in export")

# 4. Depletion data
if has_depletion_data:
    with st.expander("4️⃣ Depletion Data", expanded=False):
        depletion_data = st.session_state.get("fullrange_depletion_data", None)
        if depletion_data is not None:
            st.dataframe(depletion_data)
            st.session_state["export_depletion_data"] = True
            st.success("✅ Will be included in export")

# 5. MegaSum data
if has_megasum:
    with st.expander("5️⃣ MegaSum Data", expanded=False):
        megasum = st.session_state.get("MegaSum", None)
        if megasum is not None:
            st.write(f"MegaSum shape: {megasum.shape}")
            st.dataframe(megasum.head(20))
            st.session_state["export_megasum"] = True
            st.success("✅ Will be included in export")

# 6. Peak detection results
if has_peak_detection:
    with st.expander("6️⃣ Peak Detection Results", expanded=False):
        detected_mz = st.session_state.get("detected_mz", None)
        candidates_df = st.session_state.get("candidates_df", None)
        
        if detected_mz is not None:
            st.write(f"Detected peaks: {len(detected_mz)}")
            st.dataframe(pd.DataFrame({"m/z": detected_mz}))
        
        if candidates_df is not None:
            st.write("Candidate formulas:")
            st.dataframe(candidates_df)
        
        st.session_state["export_peak_detection"] = True
        st.success("✅ Will be included in export")

# REMPI data
if rempi_active:
    with st.expander("7️⃣ REMPI Data", expanded=False):
        rempi_df = st.session_state.get("rempi_compiled_dataframe", None)
        rempi_bc = st.session_state.get("rempi_baseline_corrected", None)
        
        if rempi_df is not None:
            st.write("REMPI Compiled Data:")
            st.dataframe(rempi_df.head(20))
        
        if rempi_bc is not None:
            st.write("REMPI Baseline Corrected:")
            st.dataframe(rempi_bc.head(20))
        
        st.session_state["export_rempi"] = True
        st.success("✅ Will be included in export")

# ============================================================================
# SECTION 4: Plots to Export
# ============================================================================

st.markdown("---")
st.markdown("## 📈 Plots to Export")

st.info("💡 **Tip:** You can add plots to the report from other pages using the '📎 Add to Report' button that appears next to plots.")

def generate_mass_spectra_plot():
    """Generate mass spectra plot for specified wavenumber"""
    if not has_baseline_corrected:
        return None
    
    bc_data = st.session_state.get("compilation_baseline_corrected_data", {})
    x_mass = st.session_state.get("x_mass", None)
    plot_wn = st.session_state.get("plot_wavenumber", None)
    mass_complex = st.session_state.get("mass_complex", 100)
    complex_name = st.session_state.get("complex", "")
    
    if not bc_data or x_mass is None or plot_wn is None:
        return None
    
    if plot_wn not in bc_data:
        plot_wn = list(bc_data.keys())[0]
    
    df = bc_data[plot_wn]
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    # Plot last two columns (withoutIR and withIR)
    if len(df.columns) >= 2:
        ax.plot(x_mass, df.iloc[:, -2], 'b-', linewidth=0.5, label='Without IR')
        ax.plot(x_mass, df.iloc[:, -1], 'r-', linewidth=0.5, label='With IR')
    
    ax.axvline(mass_complex, color='green', linestyle='--', alpha=0.7, 
              label=f'{complex_name} ({mass_complex:.1f} amu)')
    ax.axhline(0, color='lime', linewidth=1)
    ax.set_xlim(mass_complex - 10, mass_complex + 10)
    ax.set_xlabel('Mass (amu)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(f'Mass Spectra - {plot_wn} cm⁻¹')
    ax.legend(fontsize=8)
    plt.tight_layout()
    
    return fig, f"Mass Spectra - {plot_wn} cm⁻¹"

def generate_depletion_plot():
    """Generate depletion full range plot"""
    if not has_depletion_data:
        return None
    
    depletion_data = st.session_state.get("fullrange_depletion_data", None)
    complex_name = st.session_state.get("complex", "")
    
    if depletion_data is None:
        return None
    
    data = np.array(depletion_data)
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(data[:, 0], data[:, 3], 'b-', linewidth=1)
    ax.scatter(data[:, 0], data[:, 3], s=10)
    ax.axhline(0, color='lime', linewidth=1)
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Depletion')
    ax.set_title(f'Depletion - Full Range - {complex_name}')
    plt.tight_layout()
    
    return fig, f"Depletion Full Range - {complex_name}"

def generate_ln_depletion_plot():
    """Generate -ln(depletion) full range plot"""
    if not has_depletion_data:
        return None
    
    depletion_data = st.session_state.get("fullrange_depletion_data", None)
    complex_name = st.session_state.get("complex", "")
    
    if depletion_data is None:
        return None
    
    data = np.array(depletion_data)
    
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(data[:, 0], data[:, 4], 'b-', linewidth=1)
    ax.scatter(data[:, 0], data[:, 4], s=10)
    ax.axhline(0, color='lime', linewidth=1)
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('-ln(Depletion)')
    ax.set_title(f'-ln(Depletion) - Full Range - {complex_name}')
    plt.tight_layout()
    
    return fig, f"-ln(Depletion) Full Range - {complex_name}"

def generate_megasum_plot():
    """Generate MegaSum plot"""
    if not has_megasum:
        return None
    
    megasum = st.session_state.get("MegaSum", None)
    x_mass = st.session_state.get("x_mass", None)
    
    if megasum is None or x_mass is None:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    # Plot baseline corrected signals
    if "baseline_corrected_signal_withoutIR" in megasum.columns:
        ax.plot(x_mass, megasum["baseline_corrected_signal_withoutIR"], 'b-', linewidth=0.5, label='Without IR')
    if "baseline_corrected_signal_withIR" in megasum.columns:
        ax.plot(x_mass, megasum["baseline_corrected_signal_withIR"], 'r-', linewidth=0.5, label='With IR')
    
    ax.axhline(0, color='lime', linewidth=1)
    ax.set_xlabel('Mass (amu)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title('MegaSum - All Wavenumbers Combined')
    ax.legend(fontsize=8)
    plt.tight_layout()
    
    return fig, "MegaSum - All Wavenumbers"

# Generate available plots
available_plots = []

if has_baseline_corrected:
    result = generate_mass_spectra_plot()
    if result:
        available_plots.append(result)

if has_depletion_data:
    result = generate_depletion_plot()
    if result:
        available_plots.append(result)
    
    result = generate_ln_depletion_plot()
    if result:
        available_plots.append(result)

if has_megasum:
    result = generate_megasum_plot()
    if result:
        available_plots.append(result)

# Display available plots
if available_plots:
    st.write(f"**{len(available_plots)} auto-generated plots available:**")
    
    for fig, title in available_plots:
        with st.expander(f"📊 {title}", expanded=False):
            st.pyplot(fig)
            plt.close(fig)

# Also show user-added plots
if report_plots:
    st.write(f"**{len(report_plots)} user-added plots in queue:**")
    for plot in report_plots:
        st.write(f"- {plot['title']}")

# ============================================================================
# SECTION 5: Parameters Summary
# ============================================================================

st.markdown("---")
st.markdown("## ⚙️ Parameters Summary")

with st.expander("View All Parameters", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### IR-UV-Ion-Dip Parameters")
        params_ir = {
            "Element 1": st.session_state.get('element1', 'N/A'),
            "Element 2": st.session_state.get('element2', 'N/A'),
            "Element 3": st.session_state.get('element3', 'N/A'),
            "Mass Element 1": st.session_state.get('mass_element1', 'N/A'),
            "Mass Element 2": st.session_state.get('mass_element2', 'N/A'),
            "Mass Element 3": st.session_state.get('mass_element3', 'N/A'),
            "Complex": st.session_state.get('complex', 'N/A'),
            "Mass Complex": st.session_state.get('mass_complex', 'N/A'),
            "Charge State": st.session_state.get('charge_state', 'N/A'),
            "t_off": st.session_state.get('t_off', 'N/A'),
            "alpha": st.session_state.get('alpha', 'N/A'),
            "Baseline Reference": st.session_state.get('baseline_reference', 'N/A'),
            "Baseline Width": st.session_state.get('baseline_width', 'N/A'),
            "Baseline Method": st.session_state.get('baseline_method', 'N/A'),
            "n_element1": st.session_state.get('n_element1', 'N/A'),
            "n_element2": st.session_state.get('n_element2', 'N/A'),
            "n_element3": st.session_state.get('n_element3', 'N/A'),
        }
        for k, v in params_ir.items():
            st.write(f"- **{k}:** {v}")
    
    with col2:
        st.markdown("### REMPI Parameters")
        params_rempi = {
            "Molecule Name": st.session_state.get('rempi_molecule_name', 'N/A'),
            "Molecule Mass": st.session_state.get('rempi_molecule_mass', 'N/A'),
            "t_off": st.session_state.get('rempi_t_off', 'N/A'),
            "alpha": st.session_state.get('rempi_alpha', 'N/A'),
            "Baseline Reference": st.session_state.get('rempi_baseline_reference', 'N/A'),
            "Baseline Width": st.session_state.get('rempi_baseline_width', 'N/A'),
        }
        for k, v in params_rempi.items():
            st.write(f"- **{k}:** {v}")

# ============================================================================
# SECTION 6: Notes
# ============================================================================

st.markdown("---")
st.markdown("## 📝 Notes")

if "report_notes" not in st.session_state:
    st.session_state["report_notes"] = ""

st.session_state["report_notes"] = st.text_area(
    "Add notes to include in the report",
    value=st.session_state["report_notes"],
    height=150,
    placeholder="e.g. Sample preparation details, observations, experiment conditions..."
)

# ============================================================================
# SECTION 7: Generate and Download Report
# ============================================================================

st.markdown("---")
st.markdown("## 📥 Generate & Download Report")

def _sanitize_pdf_text(text):
    """Replace Unicode characters not supported by Helvetica with ASCII equivalents."""
    text = str(text)
    text = text.replace("\u207b", "-").replace("\u00b9", "1").replace("\u2081", "1")
    text = text.replace("\u00b2", "2").replace("\u00b3", "3")
    text = text.replace("\u2013", "-").replace("\u2014", "--").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text

def create_pdf_report(plots_dir, params_ir, params_rempi):
    """Create a PDF report with all session information"""
    
    if not HAS_FPDF:
        return None
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.cell(0, 20, 'FELIX Data Analysis Report', ln=True, align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f'Generated: {analysis_date} at {analysis_time}', ln=True, align='C')
    pdf.ln(20)
    
    # IR-UV-Ion-Dip Parameters
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'IR-UV-Ion-Dip Parameters', ln=True)
    pdf.set_font('Helvetica', '', 10)
    for k, v in params_ir.items():
        if v != 'N/A':
            pdf.cell(0, 6, _sanitize_pdf_text(f"  - {k}: {v}"), ln=True)
    pdf.ln(10)
    
    # REMPI Parameters
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'REMPI Parameters', ln=True)
    pdf.set_font('Helvetica', '', 10)
    for k, v in params_rempi.items():
        if v != 'N/A':
            pdf.cell(0, 6, _sanitize_pdf_text(f"  - {k}: {v}"), ln=True)
    pdf.ln(10)
    
    # Notes
    notes = st.session_state.get("report_notes", "").strip()
    if notes:
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, 'Notes', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, _sanitize_pdf_text(notes))
        pdf.ln(10)
    
    # Add plots
    plot_files = list(Path(plots_dir).glob("*.png"))
    if plot_files:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, 'Plots', ln=True)
        pdf.ln(5)
        
        for plot_file in plot_files:
            pdf.set_font('Helvetica', 'B', 12)
            # Sanitize Unicode characters not supported by Helvetica
            plot_label = plot_file.stem.replace("_", " ")
            plot_label = plot_label.replace("\u207b", "-").replace("\u00b9", "1").replace("\u2081", "1").replace("\u00b2", "2")
            plot_label = _sanitize_pdf_text(plot_label)
            pdf.cell(0, 8, plot_label, ln=True)
            try:
                pdf.image(str(plot_file), w=180)
            except:
                pdf.cell(0, 8, _sanitize_pdf_text(f"[Could not embed: {plot_file.name}]"), ln=True)
            pdf.ln(5)
    
    return pdf

def create_data_summary_for_pdf(pdf):
    """Add data file summaries to the PDF report instead of exporting CSVs."""
    
    # Unique wavenumbers summary
    if has_unique_wavenumbers:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, 'Unique Wavenumber Parameters', ln=True)
        pdf.set_font('Helvetica', '', 10)
        unique_wn = st.session_state.get("unique_wavenumbers", [])
        step_size = st.session_state.get("step_size", "N/A")
        min_count_filter = st.session_state.get("wavenumber_min_count_filter", 0)
        pdf.cell(0, 6, _sanitize_pdf_text(f"  Total Unique Wavenumbers: {len(unique_wn)}"), ln=True)
        pdf.cell(0, 6, _sanitize_pdf_text(f"  Step Size: {step_size}"), ln=True)
        pdf.cell(0, 6, _sanitize_pdf_text(f"  Min Count Filter: {min_count_filter}"), ln=True)
        pdf.ln(3)
        # List wavenumbers in compact form
        wn_str = ", ".join(str(w) for w in unique_wn)
        pdf.set_font('Helvetica', '', 8)
        pdf.multi_cell(0, 4, _sanitize_pdf_text(f"  Wavenumbers: {wn_str}"))
        pdf.ln(5)
    
    # Baseline corrected data summary
    if has_baseline_corrected:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Baseline Corrected Mass Spectra', ln=True)
        pdf.set_font('Helvetica', '', 10)
        bc_data = st.session_state.get("compilation_baseline_corrected_data", {})
        pdf.cell(0, 6, _sanitize_pdf_text(f"  Wavenumbers with baseline correction: {len(bc_data)}"), ln=True)
        pdf.cell(0, 6, _sanitize_pdf_text(f"  Wavenumber range: {list(bc_data.keys())[0]} - {list(bc_data.keys())[-1]} cm-1"), ln=True)
        pdf.ln(5)
    
    # Depletion data summary
    if has_depletion_data:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Depletion Data', ln=True)
        pdf.set_font('Helvetica', '', 10)
        dep_data = st.session_state.get("fullrange_depletion_data", None)
        if dep_data is not None:
            pdf.cell(0, 6, _sanitize_pdf_text(f"  Columns: {', '.join(dep_data.columns.tolist())}"), ln=True)
            pdf.cell(0, 6, _sanitize_pdf_text(f"  Data points: {len(dep_data)}"), ln=True)
            pdf.cell(0, 6, _sanitize_pdf_text(f"  Wavenumber range: {dep_data.iloc[0, 0]:.1f} - {dep_data.iloc[-1, 0]:.1f} cm-1"), ln=True)
        pdf.ln(5)
    
    # MegaSum summary
    if has_megasum:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'MegaSum Data', ln=True)
        pdf.set_font('Helvetica', '', 10)
        megasum = st.session_state.get("MegaSum", None)
        if megasum is not None:
            pdf.cell(0, 6, _sanitize_pdf_text(f"  Columns: {', '.join(megasum.columns.tolist())}"), ln=True)
            pdf.cell(0, 6, _sanitize_pdf_text(f"  Data points: {len(megasum)}"), ln=True)
        pdf.ln(5)
    
    # Peak detection summary
    if has_peak_detection:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Peak Detection Results', ln=True)
        pdf.set_font('Helvetica', '', 10)
        detected_mz = st.session_state.get("detected_mz", None)
        candidates_df = st.session_state.get("candidates_df", None)
        if detected_mz is not None:
            pdf.cell(0, 6, _sanitize_pdf_text(f"  Detected peaks: {len(detected_mz)}"), ln=True)
            mz_str = ", ".join(f"{m:.2f}" for m in detected_mz[:20])
            if len(detected_mz) > 20:
                mz_str += f" ... (+{len(detected_mz) - 20} more)"
            pdf.set_font('Helvetica', '', 8)
            pdf.multi_cell(0, 4, _sanitize_pdf_text(f"  m/z values: {mz_str}"))
            pdf.set_font('Helvetica', '', 10)
        if candidates_df is not None:
            pdf.cell(0, 6, _sanitize_pdf_text(f"  Candidate formulas: {len(candidates_df)}"), ln=True)
            # Show top candidates
            for _, row in candidates_df.head(10).iterrows():
                pdf.set_font('Helvetica', '', 8)
                pdf.cell(0, 4, _sanitize_pdf_text(f"    m/z {row.get('m/z', 'N/A')}: {row.get('formula', row.get('Formula', 'N/A'))}"), ln=True)
            if len(candidates_df) > 10:
                pdf.cell(0, 4, _sanitize_pdf_text(f"    ... (+{len(candidates_df) - 10} more)"), ln=True)
        pdf.ln(5)
    
    # REMPI summary
    if rempi_active:
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'REMPI Data', ln=True)
        pdf.set_font('Helvetica', '', 10)
        rempi_df = st.session_state.get("rempi_compiled_dataframe", None)
        if rempi_df is not None:
            pdf.cell(0, 6, _sanitize_pdf_text(f"  Data points: {len(rempi_df)}"), ln=True)
            pdf.cell(0, 6, _sanitize_pdf_text(f"  Columns: {', '.join(rempi_df.columns.tolist()[:5])}"), ln=True)
        pdf.ln(5)

if st.button("📄 Generate & Download Report (PDF)", use_container_width=True, type="primary"):
    
    if not HAS_FPDF:
        st.error("PDF generation requires `fpdf2`. Install with: `pip install fpdf2`")
    else:
        with st.spinner("Generating PDF report..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                plots_dir = tmpdir / "plots"
                plots_dir.mkdir()
                
                # ===== SAVE PLOTS as PNGs for embedding in PDF =====
                plot_idx = 1
                for fig, title in available_plots:
                    safe_title = title.replace(" ", "_").replace("/", "-")
                    fig.savefig(plots_dir / f"{plot_idx:02d}_{safe_title}.png", dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    plot_idx += 1
                
                # User-added plots from report queue
                for plot in report_plots:
                    safe_title = plot['title'].replace(" ", "_").replace("/", "-")
                    img_path = plots_dir / f"{plot_idx:02d}_{safe_title}.png"
                    with open(img_path, 'wb') as f:
                        f.write(plot['image_bytes'])
                    plot_idx += 1
                
                # ===== CREATE PDF =====
                params_ir = {
                    "Element 1": st.session_state.get('element1', 'N/A'),
                    "Element 2": st.session_state.get('element2', 'N/A'),
                    "Element 3": st.session_state.get('element3', 'N/A'),
                    "Complex": st.session_state.get('complex', 'N/A'),
                    "Mass Complex": st.session_state.get('mass_complex', 'N/A'),
                    "t_off": st.session_state.get('t_off', 'N/A'),
                    "alpha": st.session_state.get('alpha', 'N/A'),
                    "Baseline Reference": st.session_state.get('baseline_reference', 'N/A'),
                    "Baseline Width": st.session_state.get('baseline_width', 'N/A'),
                    "Step Size": st.session_state.get('step_size', 'N/A'),
                    "Min Count Filter": st.session_state.get('wavenumber_min_count_filter', 0),
                    "Unique Wavenumbers": len(st.session_state.get('unique_wavenumbers', [])),
                }
                params_rempi = {
                    "Molecule Name": st.session_state.get('rempi_molecule_name', 'N/A'),
                    "Molecule Mass": st.session_state.get('rempi_molecule_mass', 'N/A'),
                    "t_off": st.session_state.get('rempi_t_off', 'N/A'),
                    "alpha": st.session_state.get('rempi_alpha', 'N/A'),
                }
                
                pdf = create_pdf_report(plots_dir, params_ir, params_rempi)
                if pdf:
                    # Add data summaries to PDF
                    create_data_summary_for_pdf(pdf)
                    
                    # Save PDF to buffer
                    pdf_buffer = io.BytesIO()
                    pdf.output(pdf_buffer)
                    pdf_buffer.seek(0)
                    
                    complex_name = st.session_state.get('complex', 'Analysis')
                    pdf_filename = f"FELIX_Report_{complex_name}_{analysis_date.replace('-', '')}.pdf"
                    
                    st.success("✅ PDF report generated successfully!")
                    
                    st.download_button(
                        label="⬇️ Download Report (PDF)",
                        data=pdf_buffer,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True
                    )

# ============================================================================
# SECTION 7: Debug View
# ============================================================================

with st.expander("🔧 Debug: View All Session State Variables", expanded=False):
    st.markdown("**All session state keys:**")
    keys = sorted(st.session_state.keys())
    for key in keys:
        value = st.session_state[key]
        if isinstance(value, (pd.DataFrame, np.ndarray)):
            st.write(f"- `{key}`: {type(value).__name__} (shape: {getattr(value, 'shape', 'N/A')})")
        elif isinstance(value, dict):
            st.write(f"- `{key}`: dict ({len(value)} keys)")
        elif isinstance(value, list):
            st.write(f"- `{key}`: list ({len(value)} items)")
        else:
            st.write(f"- `{key}`: {value}")
