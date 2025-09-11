#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-parameter NLS for λ_RCL and R_CL_s.

Input : 'input_fit_rcl_nls.xlsx' (two cols: BD, eta_RCL; no header)
Output: 'result_fit_rcl_nls.xlsx' with params, data_fit, metrics, settings
"""
import pandas as pd
import numpy as np
from scipy.optimize import least_squares

INPUT_PATH  = "input_fit_rcl_nls.xlsx"
RESULT_PATH = "result_fit_rcl_nls.xlsx"
b      = 0.11025
alpha  = 1.1982
j      = 4.0
use_relative_residuals = False
LN10 = np.log(10.0)

def eta_rcl_model(BD, lam, R_s):
    inner = (j * LN10 / (2.0*b)) * ( (R_s/2.0) * (1.0 + 1.0/(BD*lam)) )
    return (b/alpha) * np.log10(1.0 + np.power(inner, alpha))

def fit_params(BD, y, x0=(0.02, 0.05)):
    def resid_fun(params):
        lam, R_s = params
        if lam <= 0 or R_s <= 0:
            return 1e6*np.ones_like(y)
        yhat = eta_rcl_model(BD, lam, R_s)
        r = y - yhat
        return r/np.maximum(np.abs(y),1e-9) if use_relative_residuals else r
    return least_squares(resid_fun, x0=np.array(x0),
                         bounds=(np.array([1e-9,1e-9]), np.array([np.inf,np.inf])),
                         method="trf")

def main():
    df = pd.read_excel(INPUT_PATH, header=None).dropna()
    if df.shape[1] < 2:
        raise ValueError("Expect two columns: BD, eta_RCL.")
    BD = df.iloc[:,0].astype(float).to_numpy()
    y  = df.iloc[:,1].astype(float).to_numpy()
    n = len(y)

    # Full-data fit
    sol = fit_params(BD, y, x0=(0.02, 0.05))
    lam_hat, R_s_hat = map(float, sol.x)

    # Covariance & SE
    p = 2
    r_abs = y - eta_rcl_model(BD, lam_hat, R_s_hat)
    dof = max(n-p, 1)
    sigma2 = float(np.dot(r_abs, r_abs)) / dof
    J = sol.jac
    try:
        JTJ_inv = np.linalg.inv(J.T @ J)
        cov = sigma2 * JTJ_inv
        se = np.sqrt(np.diag(cov))
        se_lam, se_Rs = float(se[0]), float(se[1])
    except np.linalg.LinAlgError:
        cov = np.full((2,2), np.nan)
        se_lam, se_Rs = np.nan, np.nan

    # 95% CIs
    tval = 1.96
    ci_lam = (lam_hat - tval*se_lam, lam_hat + tval*se_lam)
    ci_Rs  = (R_s_hat - tval*se_Rs,  R_s_hat + tval*se_Rs)

    # R² on original scale
    yhat = eta_rcl_model(BD, lam_hat, R_s_hat)
    R2 = 1.0 - float(np.sum((y - yhat)**2))/float(np.sum((y - np.mean(y))**2))

    # Q² via LOO re-fitting
    yhat_loo = np.full(n, np.nan)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        BD_tr, y_tr = BD[idx], y[idx]
        try:
            # warm-start from full fit
            sol_i = fit_params(BD_tr, y_tr, x0=(lam_hat, R_s_hat))
            lam_i, Rs_i = map(float, sol_i.x)
            yhat_loo[i] = eta_rcl_model(np.array([BD[i]]), lam_i, Rs_i)[0]
        except Exception:
            yhat_loo[i] = np.nan
    ok = np.isfinite(yhat_loo)
    if ok.sum() >= 3:
        ybar = np.nanmean(y)
        ss_res_loo = float(np.nansum((y - yhat_loo)**2))
        ss_tot = float(np.nansum((y - ybar)**2))
        Q2 = 1.0 - ss_res_loo/ss_tot if ss_tot > 0 else np.nan
    else:
        Q2 = np.nan

    # Save
    with pd.ExcelWriter(RESULT_PATH, engine="openpyxl") as writer:
        pd.DataFrame({
            "param": ["lambda_RCL","R_CL_s","b","alpha","j"],
            "value": [lam_hat, R_s_hat, b, alpha, j],
            "se":    [se_lam, se_Rs, np.nan, np.nan, np.nan],
            "ci_low":[ci_lam[0], ci_Rs[0], np.nan, np.nan, np.nan],
            "ci_high":[ci_lam[1], ci_Rs[1], np.nan, np.nan, np.nan],
            "note":  ["NLS (95% CI)","NLS (95% CI)","","",""]
        }).to_excel(writer, sheet_name="params", index=False)

        pd.DataFrame({
            "BD": BD, "eta_RCL_obs": y, "eta_RCL_fit": yhat, "residual": y - yhat
        }).to_excel(writer, sheet_name="data_fit", index=False)

        pd.DataFrame({
            "metric": ["R2","Q2","n_used","loo_preds_ok"],
            "value":  [R2, Q2, n, int(ok.sum())]
        }).to_excel(writer, sheet_name="metrics", index=False)

        pd.DataFrame({
            "setting": ["use_relative_residuals"], "value":[use_relative_residuals]
        }).to_excel(writer, sheet_name="settings", index=False)

if __name__ == "__main__":
    main()
