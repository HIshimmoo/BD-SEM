# Boundary-Density–Resolved Modeling (CPIR-BD)

Small, single-purpose Python scripts for **component-wise, physics-informed regression** under **boundary density** (BD).

- Exploratory screening with single-predictor OLS and leave-one-out cross-validation (LOO-CV).
- Mechanistic fits to estimate transport lengths (λ) and related parameters for kinetics (KIN), high-frequency ohmic part (RHF), and catalyst layer (RCL).
- Metrics: in-sample $R^2$ and predictive $Q^2$ (LOO-CV) on the original response scale.

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

---

## Inputs and outputs

### A) `screening_singleX_ols.py` — OLS + LOO-CV screening

**Input**: `input_screening_singleX_ols.xlsx` with headers
```
x1  x2  x3  y1  y2  y3  y4  y5
```
Families per $(y,x)$:
- Linear: $$y = \beta_0 + \beta_1 x$$
- Log10 $x$ ($x>0$): $$y = \beta_0 + \beta_1 \log_{10}(x)$$
- Reciprocal $x$ ($x\neq 0$): $$y = \beta_0 + \beta_1 x^{-1}$$
- Exp $y$ linear $x$ ($y>0$): $$\ln y = \beta_0 + \beta_1 x \quad\Rightarrow\quad \hat y = e^{\beta_0+\beta_1 x}$$

**Output**: `result_screening_singleX_ols.xlsx` with sheets `all_fits`, `best_single_Q2`, `Q2_pivot`, `settings`.

### B) `fit_kin_nls.py` — kinetic $\lambda_{kin}$

**Input**: `input_fit_kin_nls.xlsx` (two columns, no header): col1 $B_D$, col2 $\eta_{kin}$ (V).

**Model**

$$\eta_{kin}(B_D;\lambda)= b\,\log_{10}\left(\frac{j}{2j_0}\right) + b\,\log_{10}\left(1+\frac{1}{\lambda B_D}\right)$$

**Output**: `result_fit_kin_nls.xlsx` with sheets `params`, `data_fit`, `metrics`, `settings`.

### C) `fit_rhf_ols.py` — ohmic HF, $\lambda_{RHF}$

**Input**: `input_fit_rhf_ols.xlsx` (two columns, no header): col1 $B_D$, col2 $\eta_{RHF}$ (V). Internally $R_{HF}=\eta_{RHF}/j$.

**Model**

$$R_{HF}(B_D)=a+\frac{b}{B_D}, \qquad \lambda_{RHF}=\frac{a-R_{HF,0}}{b}$$

**Delta-method (variance of $\lambda_{RHF}$)**

$$\nabla\lambda = \begin{bmatrix} 1/b \\ -\lambda/b \end{bmatrix}, \qquad$$
$$\{Var}(\hat\lambda) \approx (\nabla\lambda)^{\top}\widehat{\{Cov}}(a,b)(\nabla\lambda)$$

**Output**: `result_fit_rhf_ols.xlsx` with sheets `params`, `data_fit`, `metrics`.

### D) `fit_rcl_nls.py` — catalyst layer, $\lambda_{RCL}, R_{CL,s}$

**Input**: `input_fit_rcl_nls.xlsx` (two columns, no header): col1 $B_D$, col2 $\eta_{RCL}$ (V).

**Model**

$$\eta_{RCL}(B_D;\lambda,R_s)= \frac{b}{\alpha}\,\log_{10}\left( 1 + \left[\frac{j\,\ln 10}{2b}\left(\frac{R_s}{2}\left(1+\frac{1}{B_D\lambda}\right)\right)\right]^{\alpha} \right)$$

**Output**: `result_fit_rcl_nls.xlsx` with sheets `params`, `data_fit`, `metrics`, `settings`.

---

## Metrics

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}, \qquad
Q^2 = 1 - \frac{\sum_i (y_i - \hat y_{(-i)})^2}{\sum_i (y_i - \bar y)^2}.
$$

All $Q^2$ values are computed on the native response scale (V or Ohm·cm^2).

---

## Constants and knobs

Each script keeps its own constants at the top (files are independent): $b$, $j_0$, $j$, $\alpha$, $R_{HF,0}$, and the `use_relative_residuals` flag.
