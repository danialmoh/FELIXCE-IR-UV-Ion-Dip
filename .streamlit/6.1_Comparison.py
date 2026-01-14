# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import matplotlib.pyplot as plt
# from io import BytesIO
# import os
# from palettable.wesanderson import Darjeeling2_5, Moonrise5_6

# # Set color palettes
# plotly_colors = Darjeeling2_5.hex_colors  # Used for Plotly plots in method comparison
# colors = Darjeeling2_5.hex_colors         # Used for Matplotlib plots in method comparison
# color_palette = Moonrise5_6.hex_colors      # Used for the multi-file comparison section

# st.title("Compare Depletion: Method 1, Method 2, and Method 3")

# # -------------------------------
# # 1. Upload Multiple Individual Scan CSVs
# # -------------------------------
# st.header("1. Upload Multiple Individual Scan CSVs")
# uploaded_files = st.file_uploader(
#     "Upload multiple CSV files (with columns: wavenumber, sum_withoutIR, sum_withIR, depletion, -ln(depletion))",
#     type=["csv"],
#     accept_multiple_files=True
# )

# if uploaded_files:
#     # Read each CSV and tag rows with the source file name
#     dfs = []
#     for file in uploaded_files:
#         df = pd.read_csv(file)
#         df["source_file"] = file.name
#         dfs.append(df)
    
#     # Concatenate all individual DataFrames
#     big_df = pd.concat(dfs, ignore_index=True)
    
#     # Create a DataFrame with a list of unique contributing files for each wavenumber
#     files_df = (
#         big_df.groupby("wavenumber")["source_file"]
#         .apply(lambda x: list(x.unique()))
#         .reset_index()
#         .rename(columns={"source_file": "files_contributed"})
#     )
    
#     # Also, count the number of unique files per wavenumber
#     counts_df = (
#         big_df.groupby("wavenumber")["source_file"]
#         .nunique()
#         .reset_index()
#         .rename(columns={"source_file": "num_files_contributed"})
#     )
    
#     # -------------------------------
#     # Compute Methods 1 and 2 from individual scans
#     # -------------------------------
#     # Aggregate data by wavenumber: averaging sum_withoutIR, sum_withIR, and depletion
#     avg_df = (
#         big_df.groupby("wavenumber", as_index=False)
#         .agg({
#             "sum_withoutIR": "mean",
#             "sum_withIR": "mean",
#             "depletion": "mean"
#         })
#     )
#     avg_df.rename(
#         columns={
#             "sum_withoutIR": "sum_withoutIR_avg",
#             "sum_withIR": "sum_withIR_avg",
#             "depletion": "depletion_method2"  # Direct average of depletion values
#         },
#         inplace=True
#     )
#     # Method 1: computed as the ratio of averaged sum_withIR to sum_withoutIR
#     avg_df["depletion_method1"] = avg_df["sum_withIR_avg"] / avg_df["sum_withoutIR_avg"]
    
#     # Merge in file count and the actual file names for each wavenumber
#     avg_df = avg_df.merge(counts_df, on="wavenumber", how="left")
#     avg_df = avg_df.merge(files_df, on="wavenumber", how="left")
    
#     st.subheader("Averaged Data (From Individual Scans)")
#     st.write(
#         """
#         **Columns**:
#         - **wavenumber**
#         - **sum_withoutIR_avg**, **sum_withIR_avg** (averaged from individual scans)
#         - **depletion_method2** (average of 'depletion' from all scans)
#         - **depletion_method1** (ratio: averaged withIR / averaged withoutIR)
#         - **num_files_contributed** (number of files that contained this wavenumber)
#         - **files_contributed** (list of file names contributing to this wavenumber)
#         """
#     )
#     st.dataframe(avg_df.head())
    
#     # -------------------------------
#     # 2. Upload Single Summed CSV (Method 3)
#     # -------------------------------
#     st.header("2. Upload Single Summed CSV")
#     summed_file = st.file_uploader(
#         "Upload the summed CSV (with columns: wavenumber, sum_withoutIR, sum_withIR, depletion, -ln(depletion))",
#         type=["csv"],
#         accept_multiple_files=False
#     )
    
#     if summed_file:
#         sum_df = pd.read_csv(summed_file)
#         # Rename columns to avoid confusion with the averaged data
#         sum_df.rename(
#             columns={
#                 "sum_withoutIR": "sum_withoutIR_sum",
#                 "sum_withIR": "sum_withIR_sum",
#                 "depletion": "depletion_method3"  # Directly from the Summed CSV
#             },
#             inplace=True
#         )
#         # Merge the summed CSV data with the averaged data based on wavenumber
#         compare_df = pd.merge(
#             avg_df,
#             sum_df[["wavenumber", "sum_withoutIR_sum", "sum_withIR_sum", "depletion_method3"]],
#             on="wavenumber",
#             how="outer"
#         )
        
#         st.subheader("Comparison of Methods 1, 2, and 3")
#         st.write(
#             """
#             **Columns**:
#             - **wavenumber**
#             - **sum_withoutIR_avg**, **sum_withIR_avg** (from individual scans)
#             - **depletion_method1** = (sum_withIR_avg / sum_withoutIR_avg)
#             - **depletion_method2** = average(depletion from all scans)
#             - **num_files_contributed** = number of files with this wavenumber
#             - **files_contributed** = list of file names contributing
#             - **sum_withoutIR_sum**, **sum_withIR_sum** (from Summed CSV)
#             - **depletion_method3** = depletion from Summed CSV (no extra calculation)
#             """
#         )
#         st.dataframe(compare_df.head())
        
#         st.download_button(
#             label="Download Comparison CSV",
#             data=compare_df.to_csv(index=False),
#             file_name="comparison.csv",
#             mime="text/csv"
#         )
        
#         # -------------------------------
#         # 3. Plotting
#         # -------------------------------
#         st.subheader("Plot: Depletion vs. Wavenumber (Interactive Plotly)")
#         # Define colors for the different methods
#         plotly_colors = [colors[3], colors[1], colors[2]]
#         # Create an interactive Plotly line plot with hover data showing both the file count and names
#         fig = px.line(
#             compare_df,
#             x="wavenumber",
#             y=["depletion_method1", "depletion_method2", "depletion_method3"],
#             labels={
#                 "value": "Depletion",
#                 "variable": "Method",
#                 "wavenumber": "Wavenumber",
#                 "num_files_contributed": "# Files"
#             },
#             title="Comparison of Depletion Methods",
#             color_discrete_sequence=plotly_colors,
#             hover_data=["num_files_contributed", "files_contributed"]
#         )
#         st.plotly_chart(fig)
        
#         html_str = fig.to_html()
#         st.download_button(
#             label="Download Plot as HTML",
#             data=html_str,
#             file_name="comparison_plot.html",
#             mime="text/html"
#         )
        
#         st.subheader("Static Plot (Matplotlib)")
#         fig_static, ax = plt.subplots(figsize=(13, 4))
#         ax.plot(compare_df["wavenumber"], compare_df["depletion_method1"],
#                 label="Method 1; (sum_withIR_avg / sum_withoutIR_avg)", color=colors[3])
#         ax.plot(compare_df["wavenumber"], compare_df["depletion_method2"],
#                 label="Method 2; average(depletion from all scans)", color=colors[1])
#         ax.plot(compare_df["wavenumber"], compare_df["depletion_method3"],
#                 label="Method 3; depletion from Summed CSV", color=colors[2])
#         ax.set_xlabel("Wavenumber")
#         ax.set_ylabel("Depletion")
#         ax.set_title("Comparison of Depletion Methods")
#         ax.legend()
#         ax.grid(True)
#         buf = BytesIO()
#         fig_static.savefig(buf, format="png", bbox_inches="tight")
#         buf.seek(0)
#         st.image(buf, caption="Static Plot", use_column_width=True)
#         st.download_button(
#             label="Download Static Plot as PNG",
#             data=buf,
#             file_name="comparison_plot.png",
#             mime="image/png"
#         )
        
#         # -------------------------------
#         # 4. Statistical Report
#         # -------------------------------
#         st.subheader("Statistical Report")
        
#         # Basic metrics: total number of files and unique wavenumbers
#         n_files = len(uploaded_files)
#         n_wavenumbers = compare_df.shape[0]
        
#         # Compare depletion values from Method 2 and Method 3
#         same_mask = np.isclose(
#             compare_df["depletion_method2"],
#             compare_df["depletion_method3"],
#             rtol=1e-5, atol=1e-8
#         )
#         n_same = same_mask.sum()
#         n_diff = n_wavenumbers - n_same
        
#         report_data = {
#             "Metric": [
#                 "Total Files Averaged",
#                 "Total Unique Wavenumbers",
#                 "Depletion values same (Method 2 vs. Method 3)",
#                 "Depletion values different (Method 2 vs. Method 3)"
#             ],
#             "Value": [n_files, n_wavenumbers, n_same, n_diff]
#         }
#         report_df = pd.DataFrame(report_data)
#         st.dataframe(report_df)
        
#         st.write("**Wavenumbers where depletion (Method 2 vs. Method 3) are the same along with contributing files:**")
#         same_df = compare_df.loc[same_mask, ["wavenumber", "files_contributed"]]
#         st.dataframe(same_df)
        
#         st.write("**Wavenumbers where depletion (Method 2 vs. Method 3) are different along with contributing files:**")
#         diff_df = compare_df.loc[~same_mask, ["wavenumber", "files_contributed"]]
#         st.dataframe(diff_df)
        
#         st.download_button(
#             label="Download Statistical Report CSV",
#             data=report_df.to_csv(index=False),
#             file_name="statistical_report.csv",
#             mime="text/csv"
#         )

# # =============================================================================
# # NEW SECTION: Compare Different Summed CSV Files (by Baseline Correction Method)
# # =============================================================================
# st.title("Compare Depletion vs. Wavenumber for Multiple CSV Files")
# files = st.file_uploader(
#     "Upload up to 5 CSV files (with columns 'wavenumber' and 'depletion')",
#     type=["csv"],
#     accept_multiple_files=True
# )
# if files:
#     all_data = []
#     for file in files:
#         df = pd.read_csv(file)
#         df.sort_values("wavenumber", inplace=True)
#         df["filename"] = os.path.basename(file.name)
#         all_data.append(df)
#     combined_df = pd.concat(all_data, ignore_index=True)
#     st.subheader("Combined Data Preview")
#     st.dataframe(combined_df.head())
    
#     fig = px.line(
#         combined_df,
#         x="wavenumber",
#         y="depletion",
#         color="filename",
#         title="Comparison of Depletion by File",
#         labels={
#             "wavenumber": "Wavenumber",
#             "depletion": "Depletion",
#             "filename": "File"
#         },
#         color_discrete_sequence=color_palette
#     )
#     st.plotly_chart(fig)
    
#     st.subheader("Static Plot: Depletion vs. Wavenumber")
#     fig_static, ax = plt.subplots(figsize=(14, 6))
#     for i, file_name in enumerate(combined_df["filename"].unique()):
#         df_file = combined_df[combined_df["filename"] == file_name].sort_values("wavenumber")
#         ax.plot(df_file["wavenumber"], df_file["depletion"], label=file_name, color=color_palette[i % len(color_palette)])
#     ax.set_xlabel("Wavenumber")
#     ax.set_ylabel("Depletion")
#     ax.set_title("Static Plot: Depletion vs. Wavenumber by File")
#     ax.legend()
#     ax.grid(True)
#     st.pyplot(fig_static)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import os
from palettable.wesanderson import Darjeeling2_5, Moonrise5_6

# Set color palettes
plotly_colors = Darjeeling2_5.hex_colors  # For Plotly plots
colors = Darjeeling2_5.hex_colors         # For Matplotlib plots
color_palette = Moonrise5_6.hex_colors      # For multi-file comparison

# Expected columns for validations
EXPECTED_INDIVIDUAL_COLS = {"wavenumber", "sum_withoutIR", "sum_withIR", "depletion"}
EXPECTED_SUMMED_COLS = {"wavenumber", "sum_withoutIR", "sum_withIR", "depletion"}

def validate_csv(df, expected_cols):
    """Validate if the dataframe contains the required columns."""
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return True

# Initialize session state variables for data sharing between sections
if 'individual_files' not in st.session_state:
    st.session_state.individual_files = None
if 'summed_file' not in st.session_state:
    st.session_state.summed_file = None
if 'big_df' not in st.session_state:
    st.session_state.big_df = None
if 'avg_df' not in st.session_state:
    st.session_state.avg_df = None
if 'compare_df' not in st.session_state:
    st.session_state.compare_df = None

# Sidebar Navigation for UX/UI improvement
nav_option = st.sidebar.radio("Navigation", 
                              ["Data Upload", "Data Processing", "Visualization", "Reporting", "Multi-file Comparison"])

# ------------------------ Data Upload Section ------------------------
if nav_option == "Data Upload":
    st.title("Data Upload")
    st.write("Upload your CSV files below. For individual scans, ensure the file includes columns: **wavenumber**, **sum_withoutIR**, **sum_withIR**, and **depletion**.")
    
    st.subheader("Upload Multiple Individual Scan CSVs")
    individual_files = st.file_uploader(
        "Upload multiple CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="individual_upload"
    )
    if individual_files:
        valid_files = []
        errors = []
        with st.spinner("Validating individual scan files..."):
            for file in individual_files:
                try:
                    df = pd.read_csv(file)
                    validate_csv(df, EXPECTED_INDIVIDUAL_COLS)
                    df["source_file"] = file.name
                    valid_files.append(df)
                except Exception as e:
                    errors.append(f"Error in {file.name}: {e}")
        if errors:
            for err in errors:
                st.error(err)
        if valid_files:
            st.session_state.individual_files = valid_files
            st.success("Individual scan files uploaded and validated successfully!")
    
    st.subheader("Upload Single Summed CSV")
    summed_file = st.file_uploader(
        "Upload the summed CSV file",
        type=["csv"],
        accept_multiple_files=False,
        key="summed_upload"
    )
    if summed_file:
        try:
            df_sum = pd.read_csv(summed_file)
            validate_csv(df_sum, EXPECTED_SUMMED_COLS)
            st.session_state.summed_file = df_sum
            st.success("Summed CSV uploaded and validated successfully!")
        except Exception as e:
            st.error(f"Error in summed CSV: {e}")

# ------------------------ Data Processing Section ------------------------
elif nav_option == "Data Processing":
    st.title("Data Processing")
    st.write("Processing the uploaded data...")
    if st.session_state.individual_files is None:
        st.error("Please upload individual scan CSV files in the Data Upload section.")
    else:
        try:
            with st.spinner("Aggregating individual scan data..."):
                # Concatenate individual CSVs
                big_df = pd.concat(st.session_state.individual_files, ignore_index=True)
                st.session_state.big_df = big_df
                
                # Create a list of contributing file names for each wavenumber
                files_df = (
                    big_df.groupby("wavenumber")["source_file"]
                    .apply(lambda x: list(x.unique()))
                    .reset_index()
                    .rename(columns={"source_file": "files_contributed"})
                )
                
                # Count the number of unique files per wavenumber
                counts_df = (
                    big_df.groupby("wavenumber")["source_file"]
                    .nunique()
                    .reset_index()
                    .rename(columns={"source_file": "num_files_contributed"})
                )
                
                # Aggregate data: compute averages for sum_withoutIR, sum_withIR, and depletion
                avg_df = (
                    big_df.groupby("wavenumber", as_index=False)
                    .agg({
                        "sum_withoutIR": "mean",
                        "sum_withIR": "mean",
                        "depletion": "mean"
                    })
                )
                avg_df.rename(
                    columns={
                        "sum_withoutIR": "sum_withoutIR_avg",
                        "sum_withIR": "sum_withIR_avg",
                        "depletion": "depletion_method2"
                    },
                    inplace=True
                )
                # Method 1: ratio of averaged values
                avg_df["depletion_method1"] = avg_df["sum_withIR_avg"] / avg_df["sum_withoutIR_avg"]
                
                # Merge file contribution details
                avg_df = avg_df.merge(counts_df, on="wavenumber", how="left")
                avg_df = avg_df.merge(files_df, on="wavenumber", how="left")
                
                st.session_state.avg_df = avg_df
                st.success("Data processing for individual scans completed!")
                st.write("Preview of Aggregated Data:")
                st.dataframe(avg_df.head())
        except Exception as e:
            st.error(f"Error during data processing: {e}")
    
    # Process the summed CSV if it has been uploaded
    if st.session_state.summed_file is not None:
        try:
            with st.spinner("Processing summed CSV data..."):
                sum_df = st.session_state.summed_file.copy()
                sum_df.rename(
                    columns={
                        "sum_withoutIR": "sum_withoutIR_sum",
                        "sum_withIR": "sum_withIR_sum",
                        "depletion": "depletion_method3"
                    },
                    inplace=True
                )
                if st.session_state.avg_df is not None:
                    compare_df = pd.merge(
                        st.session_state.avg_df,
                        sum_df[["wavenumber", "sum_withoutIR_sum", "sum_withIR_sum", "depletion_method3"]],
                        on="wavenumber",
                        how="outer"
                    )
                    st.session_state.compare_df = compare_df
                    st.success("Summed CSV data processed and merged successfully!")
                    st.write("Preview of Merged Data:")
                    st.dataframe(compare_df.head())
                else:
                    st.warning("Please process individual scan data first before merging with summed CSV.")
        except Exception as e:
            st.error(f"Error processing summed CSV: {e}")

# ------------------------ Visualization Section ------------------------
elif nav_option == "Visualization":
    st.title("Visualization")
    if st.session_state.compare_df is None:
        st.error("Please complete data processing (upload and process files) before visualization.")
    else:
        try:
            with st.spinner("Generating interactive plot..."):
                # Create interactive Plotly plot with hover data showing file contributions
                plotly_colors_custom = [colors[3], colors[1], colors[2]]
                fig = px.line(
                    st.session_state.compare_df,
                    x="wavenumber",
                    y=["depletion_method1", "depletion_method2", "depletion_method3"],
                    labels={
                        "value": "Depletion",
                        "variable": "Method",
                        "wavenumber": "Wavenumber",
                        "num_files_contributed": "# Files"
                    },
                    title="Comparison of Depletion Methods",
                    color_discrete_sequence=plotly_colors_custom,
                    hover_data=["num_files_contributed", "files_contributed"]
                )
                st.plotly_chart(fig)
                html_str = fig.to_html()
                st.download_button(
                    label="Download Interactive Plot as HTML",
                    data=html_str,
                    file_name="comparison_plot.html",
                    mime="text/html"
                )
            with st.spinner("Generating static plot..."):
                fig_static, ax = plt.subplots(figsize=(13, 4))
                ax.plot(st.session_state.compare_df["wavenumber"], st.session_state.compare_df["depletion_method1"],
                        label="Method 1; (sum_withIR_avg / sum_withoutIR_avg)", color=colors[3])
                ax.plot(st.session_state.compare_df["wavenumber"], st.session_state.compare_df["depletion_method2"],
                        label="Method 2; average(depletion from scans)", color=colors[1])
                ax.plot(st.session_state.compare_df["wavenumber"], st.session_state.compare_df["depletion_method3"],
                        label="Method 3; depletion from Summed CSV", color=colors[2])
                ax.set_xlabel("Wavenumber")
                ax.set_ylabel("Depletion")
                ax.set_title("Static Comparison of Depletion Methods")
                ax.legend()
                ax.grid(True)
                buf = BytesIO()
                fig_static.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                st.image(buf, caption="Static Plot", use_column_width=True)
                st.download_button(
                    label="Download Static Plot as PNG",
                    data=buf,
                    file_name="comparison_plot.png",
                    mime="image/png"
                )
        except Exception as e:
            st.error(f"Error during visualization: {e}")

# ------------------------ Reporting Section ------------------------
elif nav_option == "Reporting":
    st.title("Reporting")
    if st.session_state.compare_df is None:
        st.error("Please complete data processing before generating the report.")
    else:
        try:
            with st.spinner("Generating statistical report..."):
                compare_df = st.session_state.compare_df
                n_files = len(st.session_state.individual_files) if st.session_state.individual_files is not None else 0
                n_wavenumbers = compare_df.shape[0]
                same_mask = np.isclose(
                    compare_df["depletion_method2"],
                    compare_df["depletion_method3"],
                    rtol=1e-5, atol=1e-8
                )
                n_same = same_mask.sum()
                n_diff = n_wavenumbers - n_same
                report_data = {
                    "Metric": [
                        "Total Files Averaged",
                        "Total Unique Wavenumbers",
                        "Depletion values same (Method 2 vs. Method 3)",
                        "Depletion values different (Method 2 vs. Method 3)"
                    ],
                    "Value": [n_files, n_wavenumbers, n_same, n_diff]
                }
                report_df = pd.DataFrame(report_data)
                st.dataframe(report_df)
                
                st.write("**Wavenumbers where depletion (Method 2 vs. Method 3) are the same along with contributing files:**")
                same_df = compare_df.loc[same_mask, ["wavenumber", "files_contributed"]]
                st.dataframe(same_df)
                
                st.write("**Wavenumbers where depletion (Method 2 vs. Method 3) are different along with contributing files:**")
                diff_df = compare_df.loc[~same_mask, ["wavenumber", "files_contributed"]]
                st.dataframe(diff_df)
                
                st.download_button(
                    label="Download Statistical Report CSV",
                    data=report_df.to_csv(index=False),
                    file_name="statistical_report.csv",
                    mime="text/csv"
                )
            st.success("Statistical report generated successfully!")
        except Exception as e:
            st.error(f"Error during report generation: {e}")

# ------------------------ Multi-file Comparison Section ------------------------
elif nav_option == "Multi-file Comparison":
    st.title("Multi-file Comparison")
    st.write("Upload up to 5 CSV files for multi-file comparison. Each file should include the columns **wavenumber** and **depletion**.")
    multi_files = st.file_uploader(
        "Upload CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="multi_file_upload"
    )
    if multi_files:
        try:
            all_data = []
            for file in multi_files:
                df = pd.read_csv(file)
                if "wavenumber" not in df.columns or "depletion" not in df.columns:
                    raise ValueError(f"{file.name} is missing required columns (wavenumber, depletion).")
                df.sort_values("wavenumber", inplace=True)
                df["filename"] = os.path.basename(file.name)
                all_data.append(df)
            combined_df = pd.concat(all_data, ignore_index=True)
            st.subheader("Combined Data Preview")
            st.dataframe(combined_df.head())
            
            with st.spinner("Generating multi-file interactive plot..."):
                fig = px.line(
                    combined_df,
                    x="wavenumber",
                    y="depletion",
                    color="filename",
                    title="Comparison of Depletion by File",
                    labels={
                        "wavenumber": "Wavenumber",
                        "depletion": "Depletion",
                        "filename": "File"
                    },
                    color_discrete_sequence=color_palette
                )
                st.plotly_chart(fig)
            
            with st.spinner("Generating multi-file static plot..."):
                fig_static, ax = plt.subplots(figsize=(14, 6))
                for i, file_name in enumerate(combined_df["filename"].unique()):
                    df_file = combined_df[combined_df["filename"] == file_name].sort_values("wavenumber")
                    ax.plot(df_file["wavenumber"], df_file["depletion"], label=file_name, color=color_palette[i % len(color_palette)])
                ax.set_xlabel("Wavenumber")
                ax.set_ylabel("Depletion")
                ax.set_title("Static Plot: Depletion vs. Wavenumber by File")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig_static)
        except Exception as e:
            st.error(f"Error in multi-file comparison: {e}")
