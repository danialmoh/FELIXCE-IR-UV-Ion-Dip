import streamlit as st
import re
import pandas as pd
import plotly.graph_objs as go
import io
import matplotlib.pyplot as plt

st.title("Library-Peak Overlay from Files or Custom Data")

# 1) Choose input method
data_input = st.radio(
    "Select data source:",
    options=["Upload MassBank files", "Custom peaks"],
    index=0
)

# 1a) Database name (for legends)
default_db_name = "MassBank Europe (2025.05.01)" if data_input == "Upload MassBank files" else "Custom Database"
db_name = st.text_input("Database name for overlay legends:", value=default_db_name)

# 2) Gather library data
all_lib = []
if data_input == "Upload MassBank files":
    uploaded_files = st.file_uploader(
        "Upload MassBank text export(s) (.txt)",
        type="txt",
        accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded in uploaded_files:
            lines = uploaded.getvalue().decode('utf-8').splitlines()
            peaks = []
            in_block = False
            for line in lines:
                if line.startswith('PK$PEAK:'):
                    in_block = True
                    continue
                if in_block:
                    if not line.strip() or line.strip().startswith('//'):
                        break
                    parts = re.split(r"\s+", line.strip())
                    if len(parts) >= 3:
                        mz = float(parts[0])
                        val = float(parts[2])
                        peaks.append((mz, val, uploaded.name))
            if peaks:
                df = pd.DataFrame(peaks, columns=['mz', 'val', 'label'])
                all_lib.append(df)
            else:
                st.warning(f"No peaks found in {uploaded.name}")
else:
    st.markdown("Enter custom peaks (one per line: m/z intensity [label]):")
    custom_block = st.text_area("Custom peaks:")
    if custom_block.strip():
        peaks = []
        for line in custom_block.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    val = float(parts[1])
                    label = parts[2] if len(parts) > 2 else 'Custom'
                    peaks.append((mz, val, label))
                except ValueError:
                    st.warning(f"Could not parse line: {line}")
        if peaks:
            df = pd.DataFrame(peaks, columns=['mz', 'val', 'label'])
            all_lib.append(df)

# proceed only if library data exists
if all_lib:
    # Let user override labels for each dataset
    custom_labels = []
    for idx, df in enumerate(all_lib):
        default_label = df['label'].iloc[0]
        new_label = st.text_input(f"Label for source #{idx+1} (default: {default_label}):", value=default_label)
        custom_labels.append(new_label)

    # 3) Scaling options
    scale_col = st.selectbox(
        "Scale by which column?",
        options=["rel.int (3rd)", "int. (2nd)"],
        help="‘rel.int’ normalizes relative intensity, ‘int.’ uses absolute intensity"
    )
    db_scale = st.number_input(
        "Database scale factor:",
        value=1.0, step=0.1,
        help="Multiply intensities after normalization"
    )
    x_shift = st.number_input(
        "Shift theoretical m/z by:",
        value=0.0, step=0.01, format="%.4f",
        help="Offset theoretical m/z values"
    )
    st.markdown("### Axis Limits")
    x_min = st.number_input("X-axis min (m/z):", value=0.0)
    x_max = st.number_input("X-axis max (m/z):", value=1000.0)
    y_min = st.number_input("Y-axis min (Intensity):", value=0.0)
    y_max = st.number_input("Y-axis max (Intensity):", value=1.0)
    st.markdown("### Custom Annotation")
    custom_mz = st.number_input("Annotate this mass (m/z):", value=0.0, step=0.1, format="%.4f")
    custom_text = st.text_input("Annotation text:", value="Annotation")
    include_avg = st.checkbox("Include average overlay", value=True)

    if st.button("Load & Overlay"):
        # experimental data
        if "x_mass" not in st.session_state or "signal" not in st.session_state:
            st.error("No experimental data. Run Peak Detection first.")
            st.stop()
        x_exp = st.session_state["x_mass"]
        y_exp = st.session_state["signal"]
        mask = (x_exp >= x_min) & (x_exp <= x_max)
        if not mask.any():
            st.error("No experimental data within x-range.")
            st.stop()
        exp_max = y_exp[mask].max()

        # prepare colors
        colors = ['blue','green','orange','purple','brown','cyan']
        avg_color = 'red'

        # scale library data
        scaled_list = []
        for df in all_lib:
            df2 = df.copy()
            df2['mz_shifted'] = df2['mz'] + x_shift
            df2['scaled'] = df2['val'] / df2['val'].max()
            df2['scaled_int'] = df2['scaled'] * exp_max * db_scale
            scaled_list.append(df2[['mz_shifted','scaled_int']])

        # average overlay
        avg_df = None
        if include_avg and scaled_list:
            avg_df = pd.concat(scaled_list).groupby('mz_shifted')['scaled_int'].mean().reset_index()

        # Interactive full overlay
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_exp, y=y_exp, mode='lines', name='Experimental', line=dict(color='black')))
        for i, df2 in enumerate(scaled_list):
            fig.add_trace(go.Bar(x=df2['mz_shifted'], y=df2['scaled_int'], width=0.1,
                                 name=custom_labels[i],
                                 marker=dict(color=colors[i % len(colors)], opacity=0.5)))
        if include_avg and avg_df is not None:
            fig.add_trace(go.Bar(x=avg_df['mz_shifted'], y=avg_df['scaled_int'], width=0.1,
                                 name=f'Library overlay ({db_name})',
                                 marker=dict(color=avg_color, opacity=0.7)))
        fig.update_layout(title='Experimental + Library Overlays', xaxis_title='m/z', yaxis_title='Intensity',
                          xaxis_range=[x_min,x_max], yaxis_range=[y_min,y_max])
        if custom_text and custom_mz>0:
            fig.add_annotation(x=custom_mz, y=y_max, text=custom_text, showarrow=True, arrowhead=2)
        st.plotly_chart(fig, use_container_width=True)

        # Interactive experimental + average only
        if include_avg and avg_df is not None:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x_exp, y=y_exp, mode='lines', name='Experimental', line=dict(color='black')))
            fig2.add_trace(go.Bar(x=avg_df['mz_shifted'], y=avg_df['scaled_int'], width=0.1,
                                  name=f'Library overlay ({db_name})',
                                  marker=dict(color=avg_color, opacity=0.7)))
            fig2.update_layout(title='Experimental + Average Overlay', xaxis_title='m/z', yaxis_title='Intensity',
                               xaxis_range=[x_min,x_max], yaxis_range=[y_min,y_max])
            if custom_text and custom_mz>0:
                fig2.add_annotation(x=custom_mz, y=y_max, text=custom_text, showarrow=True, arrowhead=2)
            st.plotly_chart(fig2, use_container_width=True)

        # Static full overlay
        buf = io.BytesIO()
        fig3, ax = plt.subplots(figsize=(10,6))
        ax.plot(x_exp, y_exp, color='black', label='Experimental')
        for i, df2 in enumerate(scaled_list):
            ax.bar(df2['mz_shifted'], df2['scaled_int'], width=0.1,
                   color=colors[i % len(colors)], alpha=0.5,
                   label=custom_labels[i])
        if include_avg and avg_df is not None:
            ax.bar(avg_df['mz_shifted'], avg_df['scaled_int'], width=0.1,
                   color=avg_color, alpha=0.7,
                   label=f'Library overlay ({db_name})')
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        if custom_text and custom_mz>0:
            ax.annotate(custom_text, xy=(custom_mz, y_max),
                        xytext=(custom_mz, y_max*0.9), arrowprops=dict(arrowstyle='->', color='blue'))
        ax.set_xlabel('m/z'); ax.set_ylabel('Intensity'); ax.legend(); plt.tight_layout()
        fig3.savefig(buf, format='jpg'); buf.seek(0)
        st.download_button(label='Download static JPG', data=buf,
                           file_name=f"overlay_{custom_text.replace(' ','_')}.jpg",
                           mime='image/jpeg')
        st.pyplot(fig3)

        # Static experimental + average only
        if include_avg and avg_df is not None:
            buf2 = io.BytesIO()
            fig4, ax4 = plt.subplots(figsize=(10,6))
            ax4.plot(x_exp, y_exp, color='black', label='Experimental')
            ax4.bar(avg_df['mz_shifted'], avg_df['scaled_int'], width=0.1,
                    color=avg_color, alpha=0.7,
                    label=f'Library overlay ({db_name})')
            if custom_text and custom_mz>0:
                ax4.annotate(custom_text, xy=(custom_mz, y_max),
                             xytext=(custom_mz, y_max*0.9), arrowprops=dict(arrowstyle='->', color='blue'))
            ax4.set_xlim(x_min, x_max); ax4.set_ylim(y_min, y_max)
            ax4.set_xlabel('m/z'); ax4.set_ylabel('Intensity'); ax4.legend(); plt.tight_layout()
            fig4.savefig(buf2, format='jpg'); buf2.seek(0)
            st.download_button(label='Download exp+avg JPG', data=buf2,
                               file_name=f"exp_avg_overlay_{custom_text.replace(' ','_')}.jpg",
                               mime='image/jpeg')
            st.pyplot(fig4)
