"""
Analysis and plotting functions for the Regression Explorer.
Uses statsmodels and plotly for statistical summaries and visualisations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats


def correlation_heatmap(df: pd.DataFrame, columns: list):
    """Create a Plotly correlation heatmap for selected columns."""
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(title="Correlation Matrix", width=600, height=500)
    return fig


def univariate_regression(df: pd.DataFrame, y_col: str, x_col: str):
    """
    Perform simple linear regression between one X and Y.
    Returns a dict with results, scatter plot, residual plot, Q-Q plot.
    """
    X = sm.add_constant(df[x_col])
    y = df[y_col]
    model = sm.OLS(y, X, missing='drop').fit()

    summary = {
        'intercept': model.params['const'],
        'slope': model.params[x_col],
        'r_squared': model.rsquared,
        'p_value_slope': model.pvalues[x_col],
        'significant': model.pvalues[x_col] < 0.05
    }

    # Scatter with regression line
    scatter_fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                             title=f"{y_col} vs {x_col}")
    scatter_fig.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text=(f"y = {summary['intercept']:.3f} + {summary['slope']:.3f}*x<br>"
              f"R² = {summary['r_squared']:.3f}, p = {summary['p_value_slope']:.4f}"),
        showarrow=False, font=dict(size=12), bgcolor="white"
    )

    # Residual diagnostics
    residuals = model.resid
    fitted = model.fittedvalues

    resid_fig = px.scatter(x=fitted, y=residuals,
                           labels={'x': 'Fitted values', 'y': 'Residuals'})
    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
    resid_fig.update_layout(title="Residuals vs Fitted")

    qq = stats.probplot(residuals, dist="norm")
    qq_fig = px.scatter(x=qq[0][0], y=qq[0][1],
                        labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
    slope, intercept, r = qq[1]
    line_x = np.array([qq[0][0].min(), qq[0][0].max()])
    line_y = slope * line_x + intercept
    qq_fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                line=dict(color='red', dash='dash'), name='Normal line'))
    qq_fig.update_layout(title="Q-Q Plot of Residuals")

    return summary, scatter_fig, resid_fig, qq_fig


def multiple_regression(df: pd.DataFrame, y_col: str, x_cols: list):
    """Fit multiple linear regression and return model, VIF DataFrame, F-test p-value."""
    X = sm.add_constant(df[x_cols])
    y = df[y_col]
    model = sm.OLS(y, X, missing='drop').fit()

    vif_data = pd.DataFrame({
        'Variable': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })

    return model, vif_data, model.f_pvalue


def model_diagnostics(model, df, x_cols):
    """Generate residual diagnostic plots for a fitted OLS model."""
    residuals = model.resid
    fitted = model.fittedvalues

    resid_fig = px.scatter(x=fitted, y=residuals,
                           labels={'x': 'Fitted values', 'y': 'Residuals'})
    resid_fig.add_hline(y=0, line_dash="dash", line_color="red")
    resid_fig.update_layout(title="Residuals vs Fitted")

    qq = stats.probplot(residuals, dist="norm")
    qq_fig = px.scatter(x=qq[0][0], y=qq[0][1],
                        labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'})
    slope, intercept, r = qq[1]
    line_x = np.array([qq[0][0].min(), qq[0][0].max()])
    line_y = slope * line_x + intercept
    qq_fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines',
                                line=dict(color='red', dash='dash'), name='Normal line'))
    qq_fig.update_layout(title="Q-Q Plot")

    hist_fig = px.histogram(residuals, nbins=20, marginal='box',
                            title="Residuals Distribution")

    return resid_fig, qq_fig, hist_fig


def suggest_model(model, vif_df, resid_fig=None):
    """
    Analyse model diagnostics and return textual advice for improvements.
    """
    advice = []
    high_vif = vif_df[vif_df['Variable'] != 'const']
    if not high_vif.empty and high_vif['VIF'].max() > 10:
        advice.append("- **Multicollinearity detected** (VIF > 10). Consider removing or combining highly correlated predictors.")

    pvalues = model.pvalues.drop('const', errors='ignore')
    insignificant = pvalues[pvalues > 0.05]
    if not insignificant.empty:
        advice.append(f"- **Insignificant predictors** (p > 0.05): {', '.join(insignificant.index)}. Consider dropping them.")

    if model.rsquared < 0.3:
        advice.append("- **Low R²** – the linear model explains little variance. Consider nonlinear terms or interactions.")

    advice.append("- To capture **nonlinear relationships**, try adding polynomial terms (e.g., X²).")
    advice.append("- To model **interactions**, include product terms (X1*X2).")
    advice.append("- If residual variance changes with fitted values, apply a **transformation** (log, sqrt) to Y.")

    return advice