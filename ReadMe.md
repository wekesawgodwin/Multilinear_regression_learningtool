# 📈 General Linear Regression Explorer

A Streamlit-based interactive learning tool that helps students understand **Multiple Linear Regression**, **Nonlinear Regression**, **Interactions**, **Transformations**, and **Indicator Variables**.  
Designed to complement the Moringa School Phase 4 curriculum (Week 4, Day 20 – Canvas module).

Upload your own dataset (CSV, Excel, or ZIP) and explore statistical relationships through scatter plots, regression summaries, significance tests, and model diagnostics. Get automatic advice on how to improve your model and apply transformations, polynomials, or interactions right in the browser.

---

## 🎯 What This Tool Teaches

This tool brings to life the key concepts from the Canvas pages:

- **Multiple Linear Regression** – modeling a target variable with several predictors.
- **Nonlinear Regression** – capturing curvature by adding polynomial terms.
- **Interactions** – when the effect of one predictor depends on another.
- **Transformations** – using log, sqrt, or square to improve model fit.
- **Model diagnostics** – residuals, VIF, p-values, and significance testing.

Every analysis includes plain‑language summaries and visualizations so you can **see** the statistics, not just read them.

---

## 🚀 Features 

- **Upload multiple file types**: CSV, Excel (`.xlsx`, `.xls`), or ZIP archives (containing one or more data files).
- **Interactive variable selection** – choose your target (Y) and predictors (X).
- **Correlation heatmap** – instantly spot strong/weak relationships.
- **Univariate regressions** – for every predictor, get:
  - Scatter plot with regression line
  - Equation, R², and p‑value
  - Residual vs fitted plot
  - Q‑Q plot (normality check)
- **Multiple linear regression**:
  - Full `statsmodels` OLS summary
  - Variance Inflation Factors (VIF) for multicollinearity
  - Overall F‑test p‑value
  - Residual diagnostics (fitted vs residuals, Q‑Q, histogram)
- **Model improvement suggestions** – intelligent tips based on your data:
  - Insignificant predictors
  - High multicollinearity (VIF > 10)
  - Low R² → try nonlinear terms
  - Non‑constant variance → consider transformations
- **Interactive model builder**:
  - Apply log, sqrt, or square transformations to Y or any X
  - Add polynomial terms (X², X³, …)
  - Include interaction terms (X₁ × X₂)
  - Re‑train the model and compare results side‑by‑side
- **Download transformed dataset** as CSV for further analysis.

---

## 📦 Installation & Dependencies

Make sure you have Python 3.8+ installed. Then install the required packages:

```bash
pip install -r requirements.txt