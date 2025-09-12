# Boundary-Density–Resolved Modeling (CPIR-BD)

Small, single-purpose Python scripts for **component-wise, physics‑informed regression** under **boundary density** (BD). The toolbox supports:

- **Exploratory screening** with single‑predictor OLS and leave‑one‑out cross‑validation (LOO‑CV).
- **Mechanistic fits** to estimate transport lengths (\(\lambda\)) and related parameters for kinetics (KIN), high‑frequency ohmic part (RHF), and catalyst layer (RCL).
- **Metrics**: in‑sample $R^2$ and predictive $Q^2$ (LOO‑CV) on the **original response scale**.

Each script reads its **own** Excel input (`input_<script>.xlsx`) and writes **one** Excel output (`result_<script>.xlsx`). No figures are produced.

---

## Quick start

### Environment
```bash
python -m pip install --upgrade numpy pandas scipy openpyxl
```

### Run (each script is independent)
```bash
python screening_singleX_ols.py
python fit_kin_nls.py
python fit_rhf_ols.py
python fit_rcl_nls.py
```
Each command creates a single Excel file named `result_<script>.xlsx`.

---

## Repository layout

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

## Inputs and outputs

### A) `screening_singleX_ols.py` — OLS + LOO‑CV screening

**Input**: `input_screening_singleX_ols.xlsx` with headers

```
x1  x2  x3  y1  y2  y3  y4  y5
```

For each $(y,x)$ we fit four families:

- `linear`: $$y = \beta_0 + \beta_1 x$$
- `log10x` (requires $x>0$): $$y = \beta_0 + \beta_1 \log_{10}(x)$$
- `reciprocalx` (requires $x\neq 0$): $$y = \beta_0 + \beta_1\,x^{-1}$$
- `expY_linearx` (requires $y>0$): $$\ln y = \beta_0 + \beta_1 x \quad\Rightarrow\quad \hat y = e^{\beta_0+\beta_1 x}$$

**Output**: `result_screening_singleX_ols.xlsx`  
Sheets:
- `all_fits`: one row per $(y,x,\text{family})$ with $\hat\beta_0,\hat\beta_1$, 95% CIs, $R^2$, $Q^2$, $n$, formula.
- `best_single_Q2`: best‑$Q^2$ model per response $y$.
- `Q2_pivot`: pivot of $Q^2$ (rows $y$, columns `x|family`).
- `settings`: transform list and input path.

### B) `fit_kin_nls.py` — kinetic $\lambda_{\text{kin}}$

**Input**: `input_fit_kin_nls.xlsx` (two columns, **no header**):  
col1 = $B_D$, col2 = $\eta_{\text{kin}}$ (V).

**Model**
$$
\eta_{\text{kin}}(B_D;\lambda)=
b\,\log_{10}\!\Big(\tfrac{j}{2j_0}\Big)
+ b\,\log_{10}\!\Big(1+\tfrac{1}{\lambda B_D}\Big).
$$

**Output**: `result_fit_kin_nls.xlsx`  
- `params`: $\hat\lambda_{\text{kin}}$, SE, 95% CI; constants $b, j_0, j$.  
- `data_fit`: observed vs fitted, residuals.  
- `metrics`: $R^2$ and **LOO‑CV $Q^2$** on the $\eta_{\text{kin}}$ scale.  
- `settings`: `use_relative_residuals` flag.

### C) `fit_rhf_ols.py` — ohmic HF, $\lambda_{\text{RHF}}$

**Input**: `input_fit_rhf_ols.xlsx` (two columns, **no header**):  
col1 = $B_D$, col2 = $\eta_{\text{RHF}}$ (V). Internally $R_{\text{HF}}=\eta_{\text{RHF}}/j$.

**Model**
$$
R_{\text{HF}}(B_D)=a+\frac{b}{B_D},\qquad
\lambda_{\text{RHF}}=\frac{a-R_{\text{HF},0}}{b}.
$$

**Output**: `result_fit_rhf_ols.xlsx`  
- `params`: $\hat a,\hat b$ (SE, 95% CI), $\hat\lambda_{\text{RHF}}$ with **delta‑method** CI; constants $j, R_{\text{HF},0}$.  
- `data_fit`: $R_{\text{HF}}$ observed vs fitted, residuals.  
- `metrics`: $R^2$ and **LOO‑CV $Q^2$** on the $R_{\text{HF}}$ scale.

**Delta‑method (variance of $\lambda$)**  
Let $\lambda=(a-R_{\text{HF},0})/b$. Gradient:
$$
\nabla\lambda = \big[\,1/b,\; -\lambda/b\,\big]^\top,\quad
\operatorname{Var}(\hat\lambda) \approx \nabla\lambda^\top\,\widehat{\operatorname{Cov}}(a,b)\,\nabla\lambda.
$$

### D) `fit_rcl_nls.py` — catalyst layer, $\lambda_{\text{RCL}},\;R_{\text{CL},s}$

**Input**: `input_fit_rcl_nls.xlsx` (two columns, **no header**):  
col1 = $B_D$, col2 = $\eta_{\text{RCL}}$ (V).

**Model**
$$
\eta_{\text{RCL}}(B_D;\lambda,R_s)=
\frac{b}{\alpha}\,\log_{10}\!\Bigg(1+\Big[\,\frac{j\ln 10}{2b}\,
\Big(\frac{R_s}{2}\big(1+\tfrac{1}{B_D\lambda}\big)\Big)\,\Big]^{\!\alpha}\Bigg).
$$

**Output**: `result_fit_rcl_nls.xlsx`  
- `params`: $\hat\lambda_{\text{RCL}},\;\hat R_{\text{CL},s}$ (SE, 95% CI); constants $b,\alpha,j$.  
- `data_fit`: observed vs fitted, residuals.  
- `metrics`: $R^2$ and **LOO‑CV $Q^2$** on the $\eta_{\text{RCL}}$ scale.  
- `settings`: `use_relative_residuals` flag.

---

## Metrics

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2},\qquad
Q^2 = 1 - \frac{\sum_i (y_i - \hat y_{(-i)})^2}{\sum_i (y_i - \bar y)^2},
$$
where $\hat y_{(-i)}$ is the prediction for point $i$ from a model refit on the other $n-1$ points (true LOO‑CV). All $Q^2$ values here are computed on the **native response** scale (V or Ohm·cm²).

---

## Constants and knobs

Each script keeps its own constants at the top (files are independent):

- $b$ (V/dec), $j_0$ (A/cm²), $j$ (A/cm²), $\alpha$ (dimensionless), $R_{\text{HF},0}$ (Ohm·cm²).
- `use_relative_residuals` (`False` by default): if `True`, residuals are divided by $|y|$ in both full and LOO fits.

---

## Data requirements

- For NLS models, ensure $B_D>0$ and, where required, $y>0$ (because of logs inside the model).
- Excel inputs for the three λ‑fits are **two columns with no header**. Screening input **must** use the exact column names shown above.

---

## Troubleshooting

- **ImportError: openpyxl** → `pip install openpyxl`  
- **NLS convergence** → adjust initial guesses near the top of the script (e.g., `lam0`, `R_s0`); verify finiteness/positivity of inputs.  
- **$Q^2$ is NaN** → too few valid LOO predictions, or $\sum_i (y_i-\bar y)^2=0$.
