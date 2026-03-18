import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import os
from palettable.wesanderson import Darjeeling2_5, Moonrise5_6

# Set color palettes
plotly_colors = Darjeeling2_5.hex_colors
colors = Darjeeling2_5.hex_colors
color_palette = Moonrise5_6.hex_colors

# Page configuration
st.set_page_config(page_title="Depletion Comparison", page_icon="📊", layout="wide")

# Title
st.title("📊 Depletion Spectrum Comparison")
st.markdown("""
Compare multiple depletion spectra from Section 3.0 analysis.  
Upload 2 or more CSV files containing depletion data to visualize and compare them side-by-side.
""")

st.markdown("---")

# File upload
st.header("📁 Upload Depletion Files")
st.info("💡 Upload 2 or more depletion CSV files from Section 3.0 to compare them.")

uploaded_files = st.file_uploader(
    "Select CSV files (must contain: wavenumber, depletion, -ln(depletion))",
    type=["csv"],
    accept_multiple_files=True,
    help="Choose multiple files from Section 3.0 output"
)

if uploaded_files and len(uploaded_files) >= 2:
    try:
        all_data = []
        available_columns = set()
        
        # Load and validate all files
        with st.spinner("Loading files..."):
            for file in uploaded_files:
                df = pd.read_csv(file)
                if "wavenumber" not in df.columns:
                    st.error(f"❌ {file.name} is missing 'wavenumber' column")
                    st.stop()
                
                df.sort_values("wavenumber", inplace=True)
                df["filename"] = os.path.basename(file.name)
                all_data.append(df)
                available_columns.update(df.columns)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Show file summary
        st.success(f"✅ Successfully loaded {len(uploaded_files)} files!")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**📋 File Summary**")
            file_info = pd.DataFrame({
                "File": [f.name for f in uploaded_files],
                "Rows": [len(all_data[i]) for i in range(len(all_data))],
                "Size (KB)": [f"{f.size / 1024:.1f}" for f in uploaded_files]
            })
            st.dataframe(file_info, use_container_width=True, hide_index=True)
        
        with col2:
            with st.expander("👁️ Preview Combined Data", expanded=False):
                st.dataframe(combined_df.head(15), use_container_width=True)
        
        # Determine plottable columns
        plot_columns = []
        if "depletion" in available_columns:
            plot_columns.append("depletion")
        if "-ln(depletion)" in available_columns:
            plot_columns.append("-ln(depletion)")
        
        if not plot_columns:
            st.error("❌ No plottable columns found. Files must contain 'depletion' or '-ln(depletion)' columns.")
        else:
            st.markdown("---")
            st.header("📈 Comparison Plots")
            
            # Plot controls
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                selected_columns = st.multiselect(
                    "Select data to compare:",
                    options=plot_columns,
                    default=plot_columns,
                    help="Choose which columns to plot"
                )
            with col2:
                show_interactive = st.checkbox("Interactive Plots", value=True)
            with col3:
                plot_style = st.selectbox("Line Style", ["Lines", "Lines + Markers"], index=0)
            
            if selected_columns:
                # Generate plots for each selected column
                for idx, col in enumerate(selected_columns):
                    st.markdown(f"### {col.title()}")
                    
                    # Interactive Plotly plot
                    if show_interactive:
                        st.markdown("###### *:green[Interactive Plot (Plotly)]*")
                        
                        mode = "lines+markers" if plot_style == "Lines + Markers" else "lines"
                        
                        fig = px.line(
                            combined_df,
                            x="wavenumber",
                            y=col,
                            color="filename",
                            title=f"Comparison: {col}",
                            labels={
                                "wavenumber": "Wavenumber (cm⁻¹)",
                                col: col,
                                "filename": "File"
                            },
                            color_discrete_sequence=color_palette
                        )
                        
                        # Update traces to add markers if needed
                        if plot_style == "Lines + Markers":
                            fig.update_traces(mode='lines+markers', marker=dict(size=4))
                        
                        fig.update_layout(
                            hovermode='x unified',
                            legend=dict(
                                x=0.02, 
                                y=0.98,
                                bgcolor='rgba(255,255,255,0.9)',
                                bordercolor='rgba(0,0,0,0.2)',
                                borderwidth=1
                            ),
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Static Matplotlib plot
                    st.markdown("###### *:green[Static Plot (Matplotlib)]*")
                    fig_static, ax = plt.subplots(figsize=(14, 6))
                    
                    for i, file_name in enumerate(combined_df["filename"].unique()):
                        df_file = combined_df[combined_df["filename"] == file_name].sort_values("wavenumber")
                        
                        if plot_style == "Lines + Markers":
                            ax.plot(df_file["wavenumber"], df_file[col], 
                                   label=file_name, 
                                   color=color_palette[i % len(color_palette)],
                                   linewidth=2, alpha=0.8, marker='o', markersize=3, markevery=5)
                        else:
                            ax.plot(df_file["wavenumber"], df_file[col], 
                                   label=file_name, 
                                   color=color_palette[i % len(color_palette)],
                                   linewidth=2, alpha=0.8)
                    
                    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=13, fontweight='bold')
                    ax.set_ylabel(col, fontsize=13, fontweight='bold')
                    ax.set_title(f"Comparison: {col} vs. Wavenumber", fontsize=15, fontweight='bold')
                    ax.legend(fontsize=10, loc='best', framealpha=0.9)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
                    fig_static.tight_layout()
                    st.pyplot(fig_static)
                    
                    # Download button
                    buf = BytesIO()
                    fig_static.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                    buf.seek(0)
                    st.download_button(
                        label=f"⬇️ Download {col} Plot (High-Res PNG)",
                        data=buf,
                        file_name=f"comparison_{col.replace('(', '').replace(')', '').replace('-', '')}.png",
                        mime="image/png",
                        key=f"download_{col}_{idx}"
                    )
                    
                    if idx < len(selected_columns) - 1:
                        st.markdown("---")
                
                # Export combined data
                st.markdown("---")
                st.subheader("💾 Export Combined Data")
                
                csv_data = combined_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download Combined CSV",
                    data=csv_data,
                    file_name="combined_depletion_comparison.csv",
                    mime="text/csv"
                )
                
            else:
                st.warning("⚠️ Please select at least one column to plot.")
                
    except Exception as e:
        st.error(f"❌ Error processing files: {e}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

elif uploaded_files and len(uploaded_files) < 2:
    st.warning("⚠️ Please upload at least 2 files for comparison.")
else:
    # Show helpful instructions when no files uploaded
    st.info("👆 **Get Started:** Upload your depletion CSV files above")
    
    with st.expander("ℹ️ How to use this page"):
        st.markdown("""
        ### Quick Start Guide
        
        1. **Upload Files**: Click the upload button and select 2 or more CSV files from Section 3.0
        2. **Select Columns**: Choose which data to compare (depletion and/or -ln(depletion))
        3. **View Plots**: Interactive and static comparison plots will appear automatically
        4. **Download**: Save high-resolution plots and combined data
        
        ### Required File Format
        
        Your CSV files must contain these columns:
        - `wavenumber` - IR wavenumber in cm⁻¹
        - `depletion` - Depletion values
        - `-ln(depletion)` - Natural log of depletion (IR yield)
        
        Files from Section 3.0 analysis already have the correct format.
        
        ### Features
        
        - 📊 **Interactive Plots**: Hover, zoom, pan with Plotly
        - 🖼️ **Static Plots**: Publication-ready Matplotlib figures
        - 💾 **Export**: Download high-res PNG plots (300 DPI)
        - 📁 **Data Export**: Save combined CSV data
        """)
