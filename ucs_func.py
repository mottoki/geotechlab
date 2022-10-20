import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import curve_fit
# import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from numpy import sin, cos, tan, arcsin, arccos, arctan, radians, degrees, sqrt, diag, log10, log, exp


def objective(x, a, b):
    # row['Sigma_3'] + sigci*math.sqrt((mi * row['Sigma_3'] / sigci) + 1)
    # x = Sigma3, a = sigci, b = mi
    return x + a*np.sqrt((b * x / a) + 1)

def calc_hoek(sumx, sumy, sumxy, sumxsq, sumysq, n):
    calc_sigci = sqrt((sumy/n)-((sumxy-(sumx*sumy/n))/(sumxsq-(sumx**2/n)))*sumx/n)
    calc_mi = ((sumxy-(sumx*sumy/n))/(sumxsq-(sumx**2/n)))/calc_sigci
    calc_r_sq = (sumxy-(sumx*sumy/n))**2/((sumxsq-sumx**2/n)*(sumysq-sumy**2/n))
    return calc_sigci, calc_mi, calc_r_sq

def get_curve_df(c_sigci, c_mi, x, y):
    # For Figure
    c_tensile = - int(c_sigci / (8.62 + 0.7 * c_mi))
    # Sigma3 = list(range(c_tensile, int(x.max()+0.15*x.max()+1)))
    increment = (x.max()-c_tensile)/200
    Sigma3 = np.arange(c_tensile, x.max()+increment, increment)
    c_curv = pd.DataFrame({'x':Sigma3})
    c_curv['y'] = c_curv.apply(lambda row: row['x']+c_sigci*sqrt((c_mi*row['x']/c_sigci)+1), axis=1)

    # Vertical Line for Tensile Cutoff
    c_vert = c_curv[c_curv['x']==c_tensile]
    to_append = [c_tensile, 0]
    new_row = len(c_vert)
    c_vert.loc[new_row] = to_append

    # Error R-squared
    params_c = [c_sigci, c_mi]
    residuals = y - objective(x,*params_c)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_sq_curve = 1 - (ss_res/ss_tot)
    r_sq_curve = round(r_sq_curve,4)
    return c_tensile, c_curv, c_vert, r_sq_curve


def lin_reg(x, a, b):
    return a + b*x

def get_linear_df(a, b, p_cov, x, y, min_val, lq_value, uq_value):
    # For Figure
    t_fri = arcsin((b-1)/(b+1))
    t_coh = a * (1-sin(t_fri))/(2*cos(t_fri))
    # min_val = int((max(du['Sigma3'])-min(du['Sigma3']))/40)
    Sigma3 = list(range(min_val, int(x.max()+0.15*x.max()+1)))
    c_line = pd.DataFrame({'Sigma3':Sigma3})
    c_line['Sigma1'] = c_line.apply(lambda row: a+b*row['Sigma3'], axis=1)

    # Error R-squared
    params_l = [a, b]
    residuals = y - objective(x,*params_l)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_sq_l = 1 - (ss_res/ss_tot)
    r_sq_l = round(r_sq_l,4)

    sd_a, sd_b = sqrt(diag(p_cov))
    lq_a = a - sd_a * lq_value
    lq_b = b - sd_b * lq_value
    uq_a = a + sd_a * uq_value
    uq_b = b + sd_b * uq_value
    lq_fri = arcsin((lq_b-1)/(lq_b+1))
    lq_coh = lq_a * (1-sin(lq_fri))/(2*cos(lq_fri))
    uq_fri = arcsin((uq_b-1)/(uq_b+1))
    uq_coh = uq_a * (1-sin(uq_fri))/(2*cos(uq_fri))

    c_line['Low_Std_Sigma1'] = c_line.apply(lambda row: lq_a+lq_b*row['Sigma3'], axis=1)
    c_line['High_Std_Sigma1'] = c_line.apply(lambda row: uq_a+uq_b*row['Sigma3'], axis=1)

    return t_coh, degrees(t_fri), lq_coh, degrees(lq_fri), uq_coh, degrees(uq_fri), r_sq_l, c_line

# Quantreg
def fit_model(data, q, a_intercept):
    mod = smf.quantreg("PeakSigma1 ~ Sigma3", data)
    res = mod.fit(q=q)
    if not np.isnan(a_intercept):
        if res.params["Intercept"] < a_intercept:
            mod = smf.quantreg("PeakSigma1 ~ Sigma3 -1", data)
            res = mod.fit(q=q)
            return [q, a_intercept, res.params["Sigma3"]] + res.conf_int().loc["Sigma3"].tolist()
    return [q, res.params["Intercept"], res.params["Sigma3"]] + res.conf_int().loc["Sigma3"].tolist()

def quantile_models_ucs(data, a_intercept, lq_value, uq_value):
    data.columns = ["Sigma3", "PeakSigma1"]
    quantiles = np.arange(lq_value/100, uq_value/100 + 0.05, (uq_value-lq_value)/100)

    models = [fit_model(data, x, a_intercept) for x in quantiles]
    models = pd.DataFrame(models, columns=["q", "a", "b", "lb", "ub"])
    # print(models)
    ols = smf.ols("PeakSigma1 ~ Sigma3", data).fit()
    ols_ci = ols.conf_int().loc["Sigma3"].tolist()
    intercept = ols.params["Intercept"]
    ols = dict(a=ols.params["Intercept"], b=ols.params["Sigma3"], lb=ols_ci[0], ub=ols_ci[1])
    # print(ols)
    if not np.isnan(a_intercept):
        if intercept < a_intercept:
            ols = smf.ols("PeakSigma1 ~ Sigma3 -1", data).fit()
            ols_ci = ols.conf_int().loc["Sigma3"].tolist()
            ols = dict(a=a_intercept, b=ols.params["Sigma3"], lb=ols_ci[0], ub=ols_ci[1])

    increment = (data.Sigma3.max() - data.Sigma3.min())/20
    xx = np.arange(data.Sigma3.min(), data.Sigma3.max()+increment, increment)

    get_y = lambda a, b: a + xx * b  # a = cohesion, b = np.tan(phi)

    for i in range(models.shape[0]):
        yy = get_y(models.a[i], models.b[i])
        if models.q[i] == lq_value/100:
            dlq = pd.DataFrame({'x': xx, 'y': yy})
            lq_a = models.a[i]
            lq_b = models.b[i]

        elif models.q[i] == uq_value/100:
            duq = pd.DataFrame({'x': xx, 'y': yy})
            uq_a = models.a[i]
            uq_b = models.b[i]

    yy = get_y(ols["a"], ols["b"])
    base_a = ols["a"]
    base_b = ols["b"]

    dbs = pd.DataFrame({'x': xx, 'y': yy})
    return dbs, base_a, base_b, dlq, lq_a, lq_b, duq, uq_a, uq_b
