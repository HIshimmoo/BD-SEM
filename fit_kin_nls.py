#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-parameter NLS for λ_kin using:
η_kin(BD; λ) = b*log10(j/(2*j0)) + b*log10(1 + 1/(λ*BD))

Input : 'input_fit_kin_nls.xlsx' (two columns: BD, eta_kin; no header)
Output: 'result_fit_kin_nls.xlsx' with params, data_fit, metrics, settings
"""
import pandas as pd
import numpy as np
from scipy.optimize import least_squares

INPUT_PATH  = "input_fit_kin_nls.xlsx"
RESULT_PATH = "result_fit_kin_nls.xlsx"
# constants
b   = 0.11025   # V/dec
j0  = 0.000305  # A/cm^2
j   = 4.0       # A/cm^2
use_relative_residuals = False  # also used inside LOO re-fits

def eta_kin_model(BD, lam):
    return b*np.log10(j/(2.0*j0)) + b*np.log10(1.0 + 1.0/(lam*BD))

def fit_lambda(BD, y, lam0=0.02):
    def resid_fun(lam_arr):
        lam = lam_arr[0]
        if lam <= 0:
            return 1e6*np.ones_like(y)
        yhat = eta_kin_model(BD, lam)
        r = y - yhat
        return r/np.maximum(np.abs(y),1e-9) if use_relative_residuals else r
    sol = least_squares(resid_fun, x0=np.array([lam0]), bounds=(1e-9, np.inf), method="trf")
    return sol

def main():
    df = pd.read_excel(INPUT_PATH, header=None).dropna()
    if df.shape[1] < 2:
        raise ValueError("Expect at least two columns: BD, eta_kin.")
    BD  = df.iloc[:,0].astype(float).to_numpy()
    y   = df.iloc[:,1].astype(float).to_numpy()

    # Full-data fit
    sol = fit_lambda(BD, y, lam0=0.02)
    lam_hat = float(sol.x[0])

    # Covariance & SE
    n, p = len(y), 1
    r_abs = y - eta_kin_model(BD, lam_hat)
    dof = max(n-p, 1)
    sigma2 = float(np.dot(r_abs, r_abs)) / dof
    J = sol.jac
    try:
        JTJ_inv = np.linalg.inv(J.T @ J)
        cov = sigma2 * JTJ_inv
        se_lam = float(np.sqrt(cov[0,0]))
    except np.linalg.LinAlgError:
        cov = np.array([[np.nan]]); se_lam = np.nan

    # 95% CI
    tval = 1.96
    ci_low = lam_hat - tval*se_lam
    ci_high= lam_hat + tval*se_lam

    # R² on original scale
    yhat = eta_kin_model(BD, lam_hat)
    R2 = 1.0 - float(np.sum((y - yhat)**2))/float(np.sum((y - np.mean(y))**2))

    # Q² via LOO re-fitting
    yhat_loo = np.full(n, np.nan)
    for i in range(n):
        idx = np.ones(n, dtype=bool); idx[i] = False
        BD_tr, y_tr = BD[idx], y[idx]
        try:
            sol_i = fit_lambda(BD_tr, y_tr, lam0=lam_hat)  # warm-start
            lam_i = float(sol_i.x[0])
            yhat_loo[i] = eta_kin_model(np.array([BD[i]]), lam_i)[0]
        except Exception:
            yhat_loo[i] = np.nan
    num_loo = np.count_nonzero(np.isfinite(yhat_loo))
    if num_loo >= 3:
        ybar = np.nanmean(y)
        ss_res_loo = float(np.nansum((y - yhat_loo)**2))
        ss_tot = float(np.nansum((y - ybar)**2))
        Q2 = 1.0 - ss_res_loo/ss_tot if ss_tot > 0 else np.nan
    else:
        Q2 = np.nan

    # Save
    with pd.ExcelWriter(RESULT_PATH, engine="openpyxl") as writer:
        pd.DataFrame({
            "param": ["lambda_kin", "b", "j0", "j"],
            "value": [lam_hat, b, j0, j],
            "se":    [se_lam, np.nan, np.nan, np.nan],
            "ci_low":[ci_low, np.nan, np.nan, np.nan],
            "ci_high":[ci_high, np.nan, np.nan, np.nan],
            "note": ["NLS (95% CI)", "", "", ""]
        }).to_excel(writer, sheet_name="params", index=False)

        pd.DataFrame({
            "BD": BD, "eta_kin_obs": y, "eta_kin_fit": yhat, "residual": y - yhat
        }).to_excel(writer, sheet_name="data_fit", index=False)

        pd.DataFrame({
            "metric": ["R2","Q2","n_used","loo_preds_ok"],
            "value":  [R2, Q2, n, int(num_loo)]
        }).to_excel(writer, sheet_name="metrics", index=False)

        pd.DataFrame({
            "setting": ["use_relative_residuals"], "value": [use_relative_residuals]
        }).to_excel(writer, sheet_name="settings", index=False)

if __name__ == "__main__":
    main()
