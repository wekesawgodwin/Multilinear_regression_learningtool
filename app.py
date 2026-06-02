"""
Streamlit Learning Tool: General Linear Regression Explorer
Main application that wires together data loading, analysis and UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_uploaded_file
from analysis import (
    correlation_heatmap,
    univariate_regression,
    multiple_regression,
    model_diagnostics,
    suggest_model
)


def main():
    st.set_page_config(page_title="Regression Explorer", layout="wide")
    st.title("📈 General Linear Regression Explorer")
    st.markdown("""
    This tool helps you understand **Multiple Linear Regression, Nonlinear Regression, Interactions, 
    Transformations, and Indicator Variables** by exploring your own data.
    Upload a dataset, choose variables, and discover relationships through statistics and visual diagnostics.
    """)

    uploaded_file = st.file_uploader("Upload CSV, Excel, or ZIP file", type=['csv', 'xlsx', 'xls', 'zip'])
    if uploaded_file is None:
        st.info("Please upload a file to begin.")
        return

    with st.spinner("Loading data..."):
        df = load_uploaded_file(uploaded_file)
    if df is None or df.empty:
        st.error("Could not load data. Check file format.")
        return

    st.success("Data loaded successfully!")
    st.write("### Data Preview", df.head())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns for regression analysis.")
        return

    y_col = st.selectbox("Select target variable (Y)", numeric_cols)
    remaining = [c for c in numeric_cols if c != y_col]
    x_cols = st.multiselect("Select predictor variables (X)", remaining,
                            default=remaining[:min(2, len(remaining))])
    if not x_cols:
        st.warning("Select at least one predictor.")
        return

    st.markdown("---")
    st.subheader("Exploratory Analysis")

    # Correlation heatmap
    st.write("#### Correlation Matrix")
    heatmap_fig = correlation_heatmap(df, [y_col] + x_cols)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Univariate analyses
    st.write("#### Univariate Regressions")
    for x in x_cols:
        with st.expander(f"Y ~ {x}"):
            try:
                univ_summary, scatter_fig, resid_fig, qq_fig = univariate_regression(df, y_col, x)
                st.plotly_chart(scatter_fig, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R²", f"{univ_summary['r_squared']:.4f}")
                    st.metric("p-value (slope)", f"{univ_summary['p_value_slope']:.4f}")
                with col2:
                    st.write("**Significance:**",
                             "✅ Significant" if univ_summary['significant'] else "❌ Not significant (p≥0.05)")
                st.plotly_chart(resid_fig, use_container_width=True)
                st.plotly_chart(qq_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not perform regression for {x}: {e}")

    # Multiple regression
    st.write("#### Multiple Linear Regression Model")
    if len(x_cols) > 1:
        try:
            model, vif_df, f_pvalue = multiple_regression(df, y_col, x_cols)
            st.text(model.summary())
            st.write(f"**F-test p-value:** {f_pvalue:.6f} (overall model significance)")
            st.write("**Variance Inflation Factors (VIF):**")
            st.dataframe(vif_df.style.highlight_max(subset=['VIF'], color='orange'))

            resid_multi_fig, qq_multi_fig, hist_multi_fig = model_diagnostics(model, df, x_cols)
            st.plotly_chart(resid_multi_fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(qq_multi_fig, use_container_width=True)
            with col2:
                st.plotly_chart(hist_multi_fig, use_container_width=True)

            st.subheader("🔍 Model Improvement Suggestions")
            advice_list = suggest_model(model, vif_df)
            for line in advice_list:
                st.markdown(line)

            # Model enhancement form
            st.markdown("---")
            st.subheader("⚙️ Build an Improved Model")
            st.write("Modify your model using the options below, then re-train.")

            with st.form("model_options"):
                transform_y = st.selectbox("Transformation on Y", ['None', 'log', 'sqrt', 'square'])
                transforms_x = {}
                for x in x_cols:
                    transforms_x[x] = st.selectbox(f"Transformation for {x}",
                                                   ['None', 'log', 'sqrt', 'square'], key=x)

                poly_x = st.selectbox("Add polynomial degree for", ['None'] + x_cols)
                poly_degree = st.slider("Polynomial degree", min_value=2, max_value=5, value=2) if poly_x != 'None' else None

                interaction_pairs = []
                if len(x_cols) >= 2:
                    st.write("Include interactions (product terms):")
                    for i in range(len(x_cols)):
                        for j in range(i+1, len(x_cols)):
                            if st.checkbox(f"{x_cols[i]} × {x_cols[j]}", key=f"int_{i}_{j}"):
                                interaction_pairs.append((x_cols[i], x_cols[j]))

                submitted = st.form_submit_button("Train Improved Model")

            if submitted:
                df_trans = df[[y_col] + x_cols].dropna().copy()
                y_name = y_col
                if transform_y == 'log':
                    df_trans['y_log'] = np.log(df_trans[y_col])
                    y_name = 'y_log'
                elif transform_y == 'sqrt':
                    df_trans['y_sqrt'] = np.sqrt(df_trans[y_col])
                    y_name = 'y_sqrt'
                elif transform_y == 'square':
                    df_trans['y_sq'] = df_trans[y_col] ** 2
                    y_name = 'y_sq'

                new_x_names = []
                for x in x_cols:
                    tx = transforms_x[x]
                    if tx == 'log':
                        df_trans[f'{x}_log'] = np.log(df_trans[x])
                        new_x_names.append(f'{x}_log')
                    elif tx == 'sqrt':
                        df_trans[f'{x}_sqrt'] = np.sqrt(df_trans[x])
                        new_x_names.append(f'{x}_sqrt')
                    elif tx == 'square':
                        df_trans[f'{x}_sq'] = df_trans[x] ** 2
                        new_x_names.append(f'{x}_sq')
                    else:
                        new_x_names.append(x)

                if poly_x != 'None' and poly_degree:
                    for d in range(2, poly_degree+1):
                        col_name = f'{poly_x}_pow{d}'
                        df_trans[col_name] = df_trans[poly_x] ** d
                        new_x_names.append(col_name)

                for (x1, x2) in interaction_pairs:
                    col_name = f'{x1}:{x2}'
                    df_trans[col_name] = df_trans[x1] * df_trans[x2]
                    new_x_names.append(col_name)

                try:
                    model_new, vif_new, f_pval_new = multiple_regression(df_trans, y_name, new_x_names)
                    st.success("Improved model trained!")
                    st.text(model_new.summary())
                    st.write(f"**F-test p-value:** {f_pval_new:.6f}")
                    st.write("**VIF:**")
                    st.dataframe(vif_new.style.highlight_max(subset=['VIF'], color='orange'))

                    res_fig_new, qq_fig_new, hist_fig_new = model_diagnostics(model_new, df_trans, new_x_names)
                    st.plotly_chart(res_fig_new, use_container_width=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(qq_fig_new, use_container_width=True)
                    with col2:
                        st.plotly_chart(hist_fig_new, use_container_width=True)

                    csv = df_trans.to_csv(index=False)
                    st.download_button("Download transformed dataset as CSV",
                                       data=csv, file_name="transformed_data.csv")
                except Exception as e:
                    st.error(f"Failed to fit improved model: {e}")
        except Exception as e:
            st.error(f"Multiple regression failed: {e}")
    else:
        st.info("Select at least two predictor variables to run multiple linear regression and see diagnostics.")

    st.markdown("---")
    st.markdown("Made with ❤️ for Moringa School | Learning Tool for Regression Concepts")


if __name__ == "__main__":
    main()