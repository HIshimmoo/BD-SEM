# Boundary-Density–Resolved Modeling (CPIR-BD) — README

Small, single-purpose Python scripts for **component-wise, physics‑informed regression** under **boundary density** (BD).

- **Exploratory screening**: single‑predictor OLS + LOO‑CV across several transforms.
- **Mechanistic fits**: estimate transport lengths (λ) for kinetics (KIN), ohmic HF (RHF), and catalyst‑layer (RCL).
- **Metrics**: in‑sample $R^2$ and predictive $Q^2$ (leave‑one‑out) computed on the **original response scale**.

> **Math rendering**: Equations below use GitHub’s `$...$` (inline) and `$$...$$` (display) math. If you view this file in an IDE preview that doesn’t support GitHub math, the formulas will appear as plain text. Open on **github.com** to see rendered equations.

---

## Quick start

```bash
python -m pip install --upgrade numpy pandas scipy openpyxl
```

Run scripts independently (each reads its own input and writes one result workbook):

```bash
python screening_singleX_ols.py
python fit_kin_nls.py
python fit_rhf_ols.py
python fit_rcl_nls.py
```

---

## Files

```
fit_kin_nls.py
fit_rcl_nls.py
fit_rhf_ols.py
screening_singleX_ols.py

input_fit_kin_nls.xlsx
input_fit_rcl_nls.xlsx
input_fit_rhf_ols.xlsx
input_screening_singleX_ols.xlsx

result_fit_kin_nls.xlsx
result_fit_rcl_nls.xlsx
result_fit_rhf_ols.xlsx
result_screening_singleX_ols.xlsx
```

---

## Inputs & outputs

### A) `screening_singleX_ols.py` (exploratory OLS + LOO‑CV)

**Input** (`input_screening_singleX_ols.xlsx` with headers): `x1, x2, x3, y1, y2, y3, y4, y5`

**Families (per $(y,x)$):**

$$y=eta_0+eta_1 x$$

$$y=eta_0+eta_1 \log_{10} x \quad (x>0)$$

$$y=eta_0+eta_1 \,(1/x) \quad (x\neq 0)$$

$$\ln y=eta_0+eta_1 x \;\;\Rightarrow\;\;\hat y=\exp(\beta_0+\beta_1 x) \quad (y>0)$$

**Output** (`result_screening_singleX_ols.xlsx`): sheets `all_fits`, `best_single_Q2`, `Q2_pivot`, `settings`.

---

### B) `fit_kin_nls.py` (kinetic $\lambda_{\mathrm{kin}}$)

**Input** (`input_fit_kin_nls.xlsx`, two columns, no header): col1 $B_D$, col2 $\eta_{\mathrm{kin}}$ (V).

**Model**

$$
\eta_{\mathrm{kin}}(B_D;\lambda)=
b\log_{10}\!\left(\frac{j}{2j_0}\right)+
b\log_{10}\!\left(1+\frac{1}{\lambda B_D}\right).
$$

**Output** (`result_fit_kin_nls.xlsx`): `params`, `data_fit`, `metrics`, `settings`.

---

### C) `fit_rhf_ols.py` (ohmic HF, $\lambda_{\mathrm{RHF}}$)

**Input** (`input_fit_rhf_ols.xlsx`, two columns, no header): col1 $B_D$, col2 $\eta_{\mathrm{RHF}}$ (V). Internally $R_{\mathrm{HF}}=\eta_{\mathrm{RHF}}/j$.

**Model**

$$
R_{\mathrm{HF}}(B_D)=a+\frac{b}{B_D},\qquad
\lambda_{\mathrm{RHF}}=\frac{a-R_{\mathrm{HF},0}}{b}.
$$

**Output** (`result_fit_rhf_ols.xlsx`): `params` (including delta‑method CI for $\lambda_{\mathrm{RHF}}$), `data_fit`, `metrics`.

**Delta method**

$$
\lambda=\frac{a-R_{\mathrm{HF},0}}{b},\qquad
\nabla\lambda=\begin{bmatrix}1/b\\ -\lambda/b\end{bmatrix},\qquad
\operatorname{Var}(\hat\lambda)\approx \nabla\lambda^\top\,\widehat{\operatorname{Cov}}(a,b)\,\nabla\lambda.
$$

---

### D) `fit_rcl_nls.py` (catalyst layer, $\lambda_{\mathrm{RCL}}, R_{\mathrm{CL},s}$)

**Input** (`input_fit_rcl_nls.xlsx`, two columns, no header): col1 $B_D$, col2 $\eta_{\mathrm{RCL}}$ (V).

**Model**

$$
\eta_{\mathrm{RCL}}(B_D;\lambda,R_s)=
\frac{b}{\alpha}\log_{10}\!\left(
1+\left[\frac{j\,\ln 10}{2b}\left(\frac{R_s}{2}\left(1+\frac{1}{B_D\lambda}\right)\right)\right]^{\alpha}
\right).
$$

**Output** (`result_fit_rcl_nls.xlsx`): `params`, `data_fit`, `metrics`, `settings`.

---

## Metrics

**Rendered**

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2},
\qquad
Q^2 = 1 - \frac{\sum_i (y_i - \hat y_{(-i)})^2}{\sum_i (y_i - \bar y)^2}.
$$

**LaTeX for copy‑paste**

```tex
R^2 = 1 - rac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - ar y)^2},
\qquad
Q^2 = 1 - rac{\sum_i (y_i - \hat y_{(-i)})^2}{\sum_i (y_i - ar y)^2}.
```

---

## Knobs

- Per‑script constants: `b`, `j0`, `j`, `alpha`, `R_HF0`.
- Residuals: `use_relative_residuals = False` by default (applies to full fit and LOO).

## Tips

- Ensure $B_D>0$ and, where required, $y>0$.
- Two‑column, **no‑header** inputs for the λ‑fits; named‑column input for screening.
- If NLS struggles, tweak initial guesses at the top of the file.
