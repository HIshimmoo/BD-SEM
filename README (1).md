# Boundary-Density–Resolved Modeling (CPIR-BD) — README

This repository contains small, single-purpose Python scripts for **component-wise, physics-informed regression** under **boundary density** (BD). It supports:

- **Exploratory screening** with single-predictor OLS + LOO-CV across several transforms.
- **Mechanistic fits** to estimate transport lengths \( \lambda \) and related parameters for
  kinetics (KIN), high-frequency ohmic part (RHF), and catalyst-layer transport (RCL).
- **Metrics**: in-sample $R^2$ and predictive $Q^2$ (leave-one-out) on the **original response scale**.

> **Note on formulas**: GitHub renders equations inside `$...$` (inline) and `$$...$$` (display).  
> Below, each important equation appears once as rendered math, followed by a small **LaTeX code block** for copy–paste.

---

## Quick start

### 1) Environment

```bash
python -m pip install --upgrade numpy pandas scipy openpyxl
```

Python ≥ 3.9 is recommended.

### 2) Run any script independently

Place the corresponding input Excel next to the script and run:

```bash
python screening_singleX_ols.py
python fit_kin_nls.py
python fit_rhf_ols.py
python fit_rcl_nls.py
```

Each command creates one Excel file named `result_<script>.xlsx` with sheets described below.

---

## Repository layout

```
fit_kin_nls.py                  # NLS for λ_kin (kinetic component)
fit_rcl_nls.py                  # NLS for λ_RCL and R_CL,s (catalyst-layer component)
fit_rhf_ols.py                  # OLS for R_HF vs 1/BD, λ_RHF via delta method (ohmic HF)
screening_singleX_ols.py        # OLS + LOO-CV screening across transformation families

input_fit_kin_nls.xlsx          # (BD, eta_kin)  two columns, no header
input_fit_rcl_nls.xlsx          # (BD, eta_RCL)  two columns, no header
input_fit_rhf_ols.xlsx          # (BD, eta_RHF)  two columns, no header
input_screening_singleX_ols.xlsx# x1,x2,x3,y1..y5 with headers (see below)

result_fit_kin_nls.xlsx         # output from fit_kin_nls.py
result_fit_rcl_nls.xlsx         # output from fit_rcl_nls.py
result_fit_rhf_ols.xlsx         # output from fit_rhf_ols.py
result_screening_singleX_ols.xlsx
```

---

## Inputs and outputs

### A) `screening_singleX_ols.py` (exploratory OLS + LOO-CV)

**Input**: `input_screening_singleX_ols.xlsx` with **headers**

```
x1  x2  x3  y1  y2  y3  y4  y5
```

**Families (per $(y,x)$):**
- Linear: $$y=\beta_0+\beta_1 x$$
- Logarithmic in $x$: $$y=\beta_0+\beta_1 \log_{10} x \quad (x>0)$$
- Reciprocal in $x$: $$y=\beta_0+\beta_1 \,(1/x) \quad (x\neq 0)$$
- Exponential in $y$: $$\ln y=\beta_0+\beta_1 x\;\;\Rightarrow\;\;\hat y=\exp(\beta_0+\beta_1 x)\quad (y>0)$$

**Output**: `result_screening_singleX_ols.xlsx`

- `all_fits`: one row per $(y,x,\text{family})$ with $\hat\beta_0,\hat\beta_1$, **95% CIs**, $R^2$, $Q^2$, $n$, formula.
- `best_single_Q2`: best-$Q^2$ model per response $y$.
- `Q2_pivot`: pivot table of $Q^2$ with rows $y$ and columns `x|family`.
- `settings`: list of transform families and input path.

### B) `fit_kin_nls.py` (kinetic $\lambda_{\mathrm{kin}}$)

**Input**: `input_fit_kin_nls.xlsx` (two columns, **no header**): col1 = $B_D$, col2 = $\eta_{\mathrm{kin}}$ (V)

**Model (rendered)**
$$
\eta_{\mathrm{kin}}(B_D;\lambda)=
b\log_{10}\!\left(\frac{j}{2j_0}\right)
+b\log_{10}\!\left(1+\frac{1}{\lambda B_D}\right).
$$

**LaTeX (copy–paste)**
```tex
\eta_{\mathrm{kin}}(B_D;\lambda)=
b\log_{10}\!\left(\frac{j}{2j_0}\right)
+b\log_{10}\!\left(1+\frac{1}{\lambda B_D}\right)
```

**Output**: `result_fit_kin_nls.xlsx`
- `params`: $\hat\lambda_{\mathrm{kin}}$, SE, 95% CI; constants $b, j_0, j$.
- `data_fit`: observed vs fitted, residuals.
- `metrics`: $R^2$ and **LOO-CV $Q^2$** on the $\eta_{\mathrm{kin}}$ scale.
- `settings`: `use_relative_residuals` flag.

### C) `fit_rhf_ols.py` (ohmic HF, $\lambda_{\mathrm{RHF}}$)

**Input**: `input_fit_rhf_ols.xlsx` (two columns, **no header**): col1 = $B_D$, col2 = $\eta_{\mathrm{RHF}}$ (V). Internally $R_{\mathrm{HF}}=\eta_{\mathrm{RHF}}/j$.

**Model (rendered)**
$$
R_{\mathrm{HF}}(B_D)=a+\frac{b}{B_D},\qquad
\lambda_{\mathrm{RHF}}=\frac{a-R_{\mathrm{HF},0}}{b}\,.
$$

**LaTeX (copy–paste)**
```tex
R_{\mathrm{HF}}(B_D)=a+\frac{b}{B_D},\qquad
\lambda_{\mathrm{RHF}}=\frac{a-R_{\mathrm{HF},0}}{b}
```

**Output**: `result_fit_rhf_ols.xlsx`
- `params`: $\hat a,\hat b$ (SE, 95% CI), $\hat\lambda_{\mathrm{RHF}}$ with **delta-method** CI; constants $j, R_{\mathrm{HF},0}$.
- `data_fit`: $R_{\mathrm{HF}}$ observed vs fitted, residuals.
- `metrics`: $R^2$ and **LOO-CV $Q^2$** on the $R_{\mathrm{HF}}$ scale.

**Delta method (rendered)**
$$
\lambda=\frac{a-R_{\mathrm{HF},0}}{b},\qquad
\nabla\lambda=\begin{bmatrix}1/b\\ -\lambda/b\end{bmatrix},\qquad
\mathrm{Var}(\hat\lambda)\approx \nabla\lambda^\top\,\widehat{\mathrm{Cov}}(a,b)\,\nabla\lambda.
$$

**LaTeX (copy–paste)**
```tex
\lambda=\frac{a-R_{\mathrm{HF},0}}{b},\qquad
\nabla\lambda=\begin{bmatrix}1/b\\ -\lambda/b\end{bmatrix},\qquad
\mathrm{Var}(\hat\lambda)\approx \nabla\lambda^\top\,\widehat{\mathrm{Cov}}(a,b)\,\nabla\lambda
```

### D) `fit_rcl_nls.py` (catalyst layer, $\lambda_{\mathrm{RCL}}, R_{\mathrm{CL},s}$)

**Input**: `input_fit_rcl_nls.xlsx` (two columns, **no header**): col1 = $B_D$, col2 = $\eta_{\mathrm{RCL}}$ (V).

**Model (rendered)**
$$
\eta_{\mathrm{RCL}}(B_D;\lambda,R_s)=
\frac{b}{\alpha}\log_{10}\!\left\{
1+\left[\frac{j\ln 10}{2b}\left(\frac{R_s}{2}\left(1+\frac{1}{B_D\lambda}\right)\right)\right]^{\!\alpha}\right\}.
$$

**LaTeX (copy–paste)**
```tex
\eta_{\mathrm{RCL}}(B_D;\lambda,R_s)=
\frac{b}{\alpha}\log_{10}\!\left\{
1+\left[\frac{j\ln 10}{2b}\left(\frac{R_s}{2}\left(1+\frac{1}{B_D\lambda}\right)\right)\right]^{\alpha}\right\}
```

**Output**: `result_fit_rcl_nls.xlsx`
- `params`: $\hat\lambda_{\mathrm{RCL}}, \hat R_{\mathrm{CL},s}$ (SE, 95% CI) and constants $b,\alpha,j$.
- `data_fit`: observed vs fitted, residuals.
- `metrics`: $R^2$ and **LOO-CV $Q^2$** on the $\eta_{\mathrm{RCL}}$ scale.
- `settings`: `use_relative_residuals` flag.

---

## Metrics (definitions)

**Rendered**
$$
R^2 \;=\; 1-\frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2},\qquad
Q^2 \;=\; 1-\frac{\sum_i (y_i-\hat y_{(-i)})^2}{\sum_i (y_i-\bar y)^2}\,.
$$

**LaTeX (copy–paste)**
```tex
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2},\qquad
Q^2 = 1 - \frac{\sum_i (y_i - \hat y_{(-i)})^2}{\sum_i (y_i - \bar y)^2}
```

- $\hat y_{(-i)}$ is the prediction for point $i$ from a model refit on the other $n-1$ points (true LOO-CV).
- All $Q^2$ values here are computed on the **native response** scale (V or $\Omega\cdot\mathrm{cm}^2$).

---

## Constants and knobs

Constants live at the top of each script to keep files independent:

- `b` (V/dec), `j0` (A/cm²), `j` (A/cm²), `alpha` (dimensionless), `R_HF0` (Ω·cm²).
- `use_relative_residuals`: `False` by default. If `True`, residuals are divided by $|y|$ in both full fit and LOO fits.

You can safely change these values to match your experiment; re-run the script to propagate to the result workbook.

---

## Data requirements & tips

- For NLS models (`fit_kin_nls.py`, `fit_rcl_nls.py`), ensure **$B_D>0$** and, where required, **$y>0$**.
- Excel inputs for the three λ-fits are **two columns with no header**. Screening input **must** use the exact column names shown.
- If an NLS fit fails to converge, adjust initial guesses at the top of the script (`lam0`, `R_s0`).

---

## Interpretation checklist

- **Point estimate ± 95% CI**: see the `params` sheet.
- **Goodness of fit**: `R2` in the `metrics` sheet.
- **Predictive value**: `Q2` in the `metrics` sheet (LOO-CV). Values near 1 indicate strong out-of-sample predictability; $Q^2<0$ means worse than predicting by the mean.
- For RHF, the linear $R_{\mathrm{HF}}$ vs $1/B_D$ fit supports the reciprocal BD dependence predicted by theory.

---

## Reproducibility & design choices

- Each script is a **single file** with explicit constants and no inter-file imports.
- Outputs are never merged; each run overwrites its own `result_<script>.xlsx` file.
- No figures are generated, by design (SI-friendly tabular outputs).
- No license or citation files are included per project preference.
