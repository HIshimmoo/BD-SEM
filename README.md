# Boundary-Density Semi-Empirical Modeling (BD-SEM)

Small, single-purpose Python scripts for component-wise, **semi-empirical** regression under **boundary density** (BD). This release switches from estimating a separate transport length for each component to fitting parameters of semi-empirical models while using a **single, fixed** benchmark transport length \$\lambda\$ derived from morphology. The exploratory **single-predictor OLS screening** is unchanged (now also reporting 95% CIs).

* Semi-empirical fits for \$\eta\_{\mathrm{kin}},\ \eta\_{\mathrm{RHF}},\ \eta\_{\mathrm{RCL}},\ \eta\_{\mathrm{res}}\$ with one shared \$\lambda\$.
* Metrics on the native response scale: in-sample \$R^2\$ and predictive \$Q^2\$ (LOO-CV with refitting).
* Each script reads a single Excel input and writes a single Excel output.

---
## Semi-empirical models

Let \$B\_D\$ be in \$\mathrm{mm}^{-1}\$ and \$\lambda\$ in mm. We use \$f(B\_D;\lambda)=1-\exp(-\lambda B\_D)\$ and \$1/f\approx\frac12+\frac{1}{\lambda B\_D}\$.

* **Kinetics** (fit \$b,j\_0\$; \$\lambda\$ fixed): \$\eta\_{\mathrm{kin}}(B\_D;b,j\_0)=b\log\_{10}(j/j\_0)+b\log\_{10}(1/2+1/(\lambda B\_D))\$
  Bounds used in fitting: \$0.10\<b<0.15\$ and \$5\times 10^{-5}\<j\_0<5\times 10^{-4}\$.

* **High-frequency ohmic** (fit \$R\_{\mathrm{HF},0},R\_{\mathrm{HF},s}\$; \$\lambda\$ fixed; work on \$R\_{\mathrm{HF}}=\eta\_{\mathrm{RHF}}/j\$): \$R\_{\mathrm{HF}}(B\_D)=R\_{\mathrm{HF},0}+R\_{\mathrm{HF},s}(1/2+1/(\lambda B\_D))=(R\_{\mathrm{HF},0}+R\_{\mathrm{HF},s}/2)+(,R\_{\mathrm{HF},s}/\lambda,),B\_D^{-1}\$

* **Catalyst-layer** (fit \$R\_{\mathrm{CL},s}\$; \$\lambda\$ fixed; use \$b\$ from kinetics): \$\eta\_{\mathrm{RCL}}(B\_D;R\_{\mathrm{CL},s})=(b/\alpha)\log\_{10}{1+\[(j\ln 10)/(2b)\cdot R\_{\mathrm{CL},s}(1/2+1/(\lambda B\_D))]^{\alpha}}\$

* **Residual term** (fit \$\eta\_{\mathrm{con}},k\_\theta\$; use \$b\$ from kinetics): \$\eta\_{\mathrm{res}}(B\_D;\eta\_{\mathrm{con}},k\_\theta)=\eta\_{\mathrm{con}}+b\log\_{10}(B\_D/(B\_D-jk\_\theta))\$ with a data-driven bound \$0\<k\_\theta<(1-\varepsilon)\min\_i B\_{D,i}/j\$ to ensure \$B\_D-jk\_\theta>0\$ for all points (default \$\varepsilon\approx 0.05\$).

---

## Metrics

\$R^2=1-\frac{\sum\_i(y\_i-\hat y\_i)^2}{\sum\_i(y\_i-\bar y)^2}\$, \$Q^2=1-\frac{\sum\_i(y\_i-\hat y\_{(-i)})^2}{\sum\_i(y\_i-\bar y)^2}\$

All \$Q^2\$ scores are computed on the native response scale (V or \$\Omega\cdot\mathrm{cm}^2\$) using leave-one-out refitting.

---

## Repository contents

* `screening_singleX_ols.py` — single-predictor OLS across four transformation families (linear, log10\$x\$, reciprocal \$x\$, exp\$,y\$–linear \$x\$), with LOO-CV \$Q^2\$, in-sample \$R^2\$, and 95% CIs for \$\beta\_0,\beta\_1\$.
  **Input:** `input_screening_singleX_ols.xlsx` with columns `x1 x2 x3 y1 y2 y3 y4 y5`
  **Output:** `result_screening_singleX_ols.xlsx` with sheets `all_fits`, `best_single_Q2`, `Q2_pivot`, `settings`.

* `fit_kin_nls.py` — NLS for \$b,j\_0\$ under the kinetics model above; \$\lambda\$ fixed; bounded box constraints for \$(b,j\_0)\$.
  **Input:** `input_fit_kin_nls.xlsx` (two columns: \$B\_D\$, \$\eta\_{\mathrm{kin}}\$)
  **Output:** `result_fit_kin_nls.xlsx` with `params`, `data_fit`, `metrics`, `settings`.

* `fit_rhf_ols.py` — OLS on \$R\_{\mathrm{HF}}\$ vs \$1/B\_D\$ to recover \$R\_{\mathrm{HF},0},R\_{\mathrm{HF},s}\$; \$\lambda\$ fixed; reports delta-method uncertainties for derived parameters.
  **Input:** `input_fit_rhf_ols.xlsx` (two columns: \$B\_D\$, \$\eta\_{\mathrm{RHF}}\$)
  **Output:** `result_fit_rhf_ols.xlsx` with `params`, `data_fit`, `metrics`.

* `fit_rcl_nls.py` — NLS for \$R\_{\mathrm{CL},s}\$ in the catalyst-layer model; \$\lambda\$ fixed; uses \$b\$ from the kinetics fit.
  **Input:** `input_fit_rcl_nls.xlsx` (two columns: \$B\_D\$, \$\eta\_{\mathrm{RCL}}\$)
  **Output:** `result_fit_rcl_nls.xlsx` with `params`, `data_fit`, `metrics`, `settings`.

* `fit_res_nls.py` — NLS for \$\eta\_{\mathrm{con}},k\_\theta\$ with a positivity bound on \$k\_\theta\$; uses \$b\$ from the kinetics fit.
  **Input:** `input_fit_res_nls.xlsx` (two columns: \$B\_D\$, \$\eta\_{\mathrm{res}}\$)
  **Output:** `result_fit_res_nls.xlsx` with `params`, `data_fit`, `metrics`, `settings`.

Each script writes exactly one Excel workbook named `result_<script>.xlsx` and produces no figures.

---

## Installation

```bash
python -m pip install --upgrade numpy pandas scipy openpyxl
```

---

## Usage

```bash
# exploratory screening (unchanged design; now with 95% CIs)
python screening_singleX_ols.py

# semi-empirical component fits (λ fixed; each script independent)
python fit_kin_nls.py
python fit_rhf_ols.py
python fit_rcl_nls.py
python fit_res_nls.py
```

## Conventions and knobs

* **Units:** use \$B\_D\$ in \$\mathrm{mm}^{-1}\$ and \$\lambda\$ in mm so \$1/(\lambda B\_D)\$ is dimensionless.
* **Shared λ:** default \$\lambda=0.0105\$ mm (10.05 μm) derived from the active-fraction/contact benchmark; editable at the top of each fit script.
* **Kinetics bounds:** \$0.10\<b<0.15\$, \$5\times 10^{-5}\<j\_0<5\times 10^{-4}\$.
* **Residual positivity:** \$k\_\theta\$ is bounded by \$0\<k\_\theta<(1-\varepsilon)\min\_i B\_{D,i}/j\$ to enforce \$B\_D-jk\_\theta>0\$.
* **Independence:** each script carries its own constants (e.g., \$j,\lambda,\alpha\$) and a `use_relative_residuals` toggle.

