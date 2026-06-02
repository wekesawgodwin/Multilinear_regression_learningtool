"""
Streamlit Learning Tool: General Linear Regression Explorer
----------------------------------------------------------
Helps students understand Multiple Linear Regression, Nonlinear Regression,
Interactions, Transformations, and Indicator Variables by exploring user-uploaded datasets.

Features:
    - Data loading from CSV, Excel, or ZIP files (with multiple inner files)
    - Interactive column selection (Y and Xs)
    - Statistical inference: OLS summaries, p-values, VIF, diagnostics
    - Residual plots and model quality advice
    - Model enhancement: transformations, polynomials, interactions
    - Download transformed data

Architecture: Factory pattern for file loaders; modular functions for analysis, plotting, and advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import io
import zipfile
from abc import ABC, abstractmethod

# ---------------------------- DATA LOADING (FACTORY PATTERN) ----------------------------
class DataLoader(ABC):
    """Abstract base class for data loaders."""
    @abstractmethod
    def load(self, file) -> pd.DataFrame:
        pass

class CSVLoader(DataLoader):
    def load(self, file) -> pd.DataFrame:
        return pd.read_csv(file)

class ExcelLoader(DataLoader):
    def load(self, file) -> pd.DataFrame:
        return pd.read_excel(file, engine='openpyxl')

class DataLoaderFactory:
    """Factory that returns the appropriate loader based on file extension."""
    @staticmethod
    def get_loader(file_extension: str) -> DataLoader:
        ext = file_extension.lower()
        if ext in ['.csv', '.txt']:
            return CSVLoader()
        elif ext in ['.xlsx', '.xls']:
            return ExcelLoader()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load an uploaded file (CSV, Excel, or ZIP) into a DataFrame.
    For ZIP archives, the user selects one inner file from a list.
    """
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file) as zf:
            inner_files = [f for f in zf.namelist() if not f.endswith('/')]
            if not inner_files:
                st.error("ZIP archive contains no files.")
                return None
            # Let user choose the inner file
            selected_inner = st.selectbox("Select a file from the ZIP archive:", inner_files)
            with zf.open(selected_inner) as inner_file:
                # Determine inner extension and use factory
                inner_ext = '.' + selected_inner.rsplit('.', 1)[-1] if '.' in selected_inner else '.csv'
                loader = DataLoaderFactory.get_loader(inner_ext)
                return loader.load(inner_file)
    else:
        # Single file
        ext = '.' + uploaded_file.name.rsplit('.', 1)[-1] if '.' in uploaded_file.name else '.csv'
        loader = DataLoaderFactory.get_loader(ext)
        return loader.load(uploaded_file)

# ---------------------------- ANALYSIS FUNCTIONS ---------------------------------------
def correlation_heatmap(df: pd.DataFrame, columns: list):
    """Create a Plotly correlation heatmap for selected columns."""
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(title="Correlation Matrix", width=600, height=500)
    return fig

def univariate_regression(df: pd.DataFrame, y_col: str, x_col: str):
    """
    Perform simple linear regression between one X and Y.
    Returns a dictionary with regression results, a Plotly scatter+line figure, and diagnostics figures.
    """
    X = sm.add_constant(df[x_col])
    y = df[y_col]
    model = sm.OLS(y, X, missing='drop').fit()
    summary_dict = {
        'intercept': model.params['const'],
        'slope': model.params[x_col],
        'r_squared': model.rsquared,
        'p_value_slope': model.pvalues[x_col],
        'significant': model.pvalues[x_col] < 0.05
    }
    # Scatter + regression line
    scatter_fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", title=f"{y_col} vs {x_col}")
    scatter_fig.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text=f"y = {summary_dict['intercept']:.3f} + {summary_dict['slope']:.3f}*x<br>R² = {summary_dict['r_squared']:.3f}, p = {summary_dict['p_value_slope']:.4f}",
        showarrow=False, font=dict(size=12), bgcolor="white"
    )
    # Residual diagnostics
    residuals = model.resid
    fitted = model.fittedvalues
    # Residual vs Fitted
    resid_fig = px.scatter(x=fitted, y=residuals, labels={'x': 'Fitted values', 'y': 'Residuals'})
    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
    resid_fig.update_layout(title="Residuals vs Fitted")
    # Q-Q plot
    qq = stats.probplot(residuals, dist="norm")
    qq_fig = px.scatter(x=qq[0][0], y=qq[0][1], labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
    # add line
    slope, intercept, r = qq[1]
    line_x = np.array([qq[0][0].min(), qq[0][0].max()])
    line_y = slope * line_x + intercept
    qq_fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='red', dash='dash'), name='Normal line'))
    qq_fig.update_layout(title="Q-Q Plot of Residuals")
    return summary_dict, scatter_fig, resid_fig, qq_fig

def multiple_regression(df: pd.DataFrame, y_col: str, x_cols: list):
    """Fit multiple linear regression and return statsmodels results and VIF DataFrame."""
    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    model = sm.OLS(y, X, missing='drop').fit()
    # VIF
    vif_data = pd.DataFrame({
        'Variable': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    # F-test p-value
    f_pvalue = model.f_pvalue
    return model, vif_data, f_pvalue

def model_diagnostics(model, df, x_cols):
    """Generate residual diagnostic plots for a fitted OLS model."""
    residuals = model.resid
    fitted = model.fittedvalues
    resid_fig = px.scatter(x=fitted, y=residuals, labels={'x': 'Fitted values', 'y': 'Residuals'})
    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
    resid_fig.update_layout(title="Residuals vs Fitted")
    qq = stats.probplot(residuals, dist="norm")
    qq_fig = px.scatter(x=qq[0][0], y=qq[0][1], labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
    slope, intercept, r = qq[1]
    line_x = np.array([qq[0][0].min(), qq[0][0].max()])
    line_y = slope * line_x + intercept
    qq_fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='red', dash='dash')))
    qq_fig.update_layout(title="Q-Q Plot")
    # Histogram of residuals
    hist_fig = px.histogram(residuals, nbins=20, marginal='box', title="Residuals Distribution")
    return resid_fig, qq_fig, hist_fig

def suggest_model(model, vif_df, resid_fig):
    """
    Analyze model diagnostics and return textual advice for improvements.
    """
    advice = []
    # Check for curvature in residual vs fitted (simple heuristic: lowess smooth? We'll check using residual plot shape)
    # Instead, we can check significance of quadratic terms if added? Simpler: check if mean of residuals at ends differs.
    # For educational purpose, we provide generic guidance based on p-values and VIF.
    # Check multicollinearity
    high_vif = vif_df[vif_df['Variable'] != 'const']
    if not high_vif.empty and high_vif['VIF'].max() > 10:
        advice.append("- **Multicollinearity detected** (VIF > 10). Consider removing or combining highly correlated predictors.")
    # Check p-values of coefficients
    pvalues = model.pvalues.drop('const', errors='ignore')
    insignificant = pvalues[pvalues > 0.05]
    if not insignificant.empty:
        advice.append(f"- **Insignificant predictors** (p > 0.05): {', '.join(insignificant.index)}. Consider dropping them.")
    # Check R-squared (if low)
    if model.rsquared < 0.3:
        advice.append("- **Low R²** – the linear model explains little variance. Consider nonlinear terms or interactions.")
    # Suggest transformations if residual variance seems non-constant? We can't test perfectly, but we can mention.
    # Always add general guidance
    advice.append("- To capture **nonlinear relationships**, try adding polynomial terms (e.g., X²).")
    advice.append("- To model **interactions**, include product terms (X1*X2).")
    advice.append("- If residual variance changes with fitted values, apply a **transformation** (log, sqrt) to Y.")
    return advice

# ---------------------------- STREAMLIT APP --------------------------------------------
def main():
    st.set_page_config(page_title="Regression Explorer", layout="wide")
    st.title("📈 General Linear Regression Explorer")
    st.markdown("""
    This tool helps you understand **Multiple Linear Regression, Nonlinear Regression, Interactions, 
    Transformations, and Indicator Variables** by exploring your own data.
    Upload a dataset, choose variables, and discover relationships through statistics and visual diagnostics.
    """)

    # Step 1: Upload
    uploaded_file = st.file_uploader("Upload CSV, Excel, or ZIP file", type=['csv', 'xlsx', 'xls', 'zip'])
    if uploaded_file is None:
        st.info("Please upload a file to begin.")
        return

    # Load data
    with st.spinner("Loading data..."):
        df = load_uploaded_file(uploaded_file)
    if df is None or df.empty:
        st.error("Could not load data. Check file format.")
        return

    st.success("Data loaded successfully!")
    st.write("### Data Preview", df.head())

    # Step 2: Column selection
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns for regression analysis.")
        return

    y_col = st.selectbox("Select target variable (Y)", numeric_cols)
    remaining = [c for c in numeric_cols if c != y_col]
    x_cols = st.multiselect("Select predictor variables (X)", remaining, default=remaining[:min(2, len(remaining))])
    if not x_cols:
        st.warning("Select at least one predictor.")
        return

    st.markdown("---")
    st.subheader("Exploratory Analysis")

    # Correlation heatmap
    st.write("#### Correlation Matrix")
    heatmap_fig = correlation_heatmap(df, [y_col] + x_cols)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    # Univariate analysis for each X (always shown for insight)
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
                    st.write("**Significance:**", "✅ Significant" if univ_summary['significant'] else "❌ Not significant (p≥0.05)")
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

            # Diagnostics
            resid_multi_fig, qq_multi_fig, hist_multi_fig = model_diagnostics(model, df, x_cols)
            st.plotly_chart(resid_multi_fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(qq_multi_fig, use_container_width=True)
            with col2:
                st.plotly_chart(hist_multi_fig, use_container_width=True)

            # Suggestions
            st.subheader("🔍 Model Improvement Suggestions")
            advice_list = suggest_model(model, vif_df, resid_multi_fig)
            for line in advice_list:
                st.markdown(line)

            # Interactive model enhancement
            st.markdown("---")
            st.subheader("⚙️ Build an Improved Model")
            st.write("Modify your model using the options below, then re-train.")

            with st.form("model_options"):
                transform_y = st.selectbox("Transformation on Y", ['None', 'log', 'sqrt', 'square'])
                transforms_x = {}
                for x in x_cols:
                    trans = st.selectbox(f"Transformation for {x}", ['None', 'log', 'sqrt', 'square'], key=x)
                    transforms_x[x] = trans

                # Polynomial degree (for a single X selected by user)
                poly_x = st.selectbox("Add polynomial degree for", ['None'] + x_cols)
                poly_degree = st.slider("Polynomial degree", min_value=2, max_value=5, value=2) if poly_x != 'None' else None

                # Interaction terms
                interaction_pairs = []
                if len(x_cols) >= 2:
                    st.write("Include interactions (product terms):")
                    for i in range(len(x_cols)):
                        for j in range(i+1, len(x_cols)):
                            if st.checkbox(f"{x_cols[i]} × {x_cols[j]}", key=f"int_{i}_{j}"):
                                interaction_pairs.append((x_cols[i], x_cols[j]))

                submitted = st.form_submit_button("Train Improved Model")

            if submitted:
                # Build transformed dataset
                df_trans = df[[y_col] + x_cols].dropna().copy()
                # Apply Y transformation
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

                # Apply X transformations
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

                # Add polynomial terms
                if poly_x != 'None' and poly_degree:
                    for d in range(2, poly_degree+1):
                        col_name = f'{poly_x}_pow{d}'
                        df_trans[col_name] = df_trans[poly_x] ** d
                        new_x_names.append(col_name)

                # Add interactions
                for (x1, x2) in interaction_pairs:
                    col_name = f'{x1}:{x2}'
                    df_trans[col_name] = df_trans[x1] * df_trans[x2]
                    new_x_names.append(col_name)

                # Fit new model
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

                    # Download transformed data
                    csv = df_trans.to_csv(index=False)
                    st.download_button("Download transformed dataset as CSV", data=csv, file_name="transformed_data.csv")
                except Exception as e:
                    st.error(f"Failed to fit improved model: {e}")

        except Exception as e:
            st.error(f"Multiple regression failed: {e}")
    else:
        st.info("Select at least two predictor variables to run multiple linear regression and see diagnostics.")

    st.markdown("---")
    st.markdown("Made with ❤️ by Godwin Wekesa | Learning Tool for Regression Concepts")

if __name__ == "__main__":
    main()