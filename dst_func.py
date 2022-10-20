import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import curve_fit
# import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from numpy import sin, cos, tan, arcsin, arccos, arctan, radians, degrees, sqrt, diag, log10, log, exp


# ------------------ FUNCTION ----------------------------------
# Direct Shear
def weakfunc(x, a, b):
    # shear stress = cohesion + normal stress * tan(friction_angle)
    # x = Normal Stress, a = cohesion, b = friction angle
    # return a + x * tan(b)
    return a + x * b

def powercurve(x, k, m):
    return k * (x ** m)

def bartonbandis(x, phir, inp_jrc, inp_jcs):
    return x * tan(radians(phir + inp_jrc * log10(inp_jcs / x)))

def fit_model(data, xcol, ycol, q, a_intercept):
    # if fitmethod != fit_selection[1]:
    mod = smf.quantreg(f"{ycol} ~ {xcol}", data)
    res = mod.fit(q=q)
    if not np.isnan(a_intercept):
        if res.params["Intercept"] < a_intercept:
            mod = smf.quantreg(f"{ycol} ~ {xcol} -1", data)
            res = mod.fit(q=q)
            return [q, a_intercept, res.params[xcol]] + res.conf_int().loc[xcol].tolist()
    return [q, res.params["Intercept"], res.params[xcol]] + res.conf_int().loc[xcol].tolist()
    # else:
    #     mod = smf.quantreg("ShearStress ~ NormalStress + tan(radians(inp_jrc + log10(inp_jcs/I(NormalStress)))))", data)
    #     res = mod.fit(q=q)
    #     return [q, a_intercept, res.params["NormalStress"]] + res.conf_int().loc["NormalStress"].tolist()

def quantile_models(data, xcol, ycol, a_intercept, lq_value, uq_value):
    data.columns = [xcol, ycol]
    # quantiles = np.arange(lq_value/100, uq_value/100 + 0.05, (uq_value-lq_value)/100)
    # lq = (50 - ci/2)/100
    # uq = (50 + ci/2)/100
    lq = lq_value/100
    uq = uq_value/100
    # print(lq, uq, ci)
    quantiles = np.arange(lq, uq+0.05, uq - lq)

    models = [fit_model(data, xcol, ycol, x, a_intercept) for x in quantiles]

    models = pd.DataFrame(models, columns=["q", "a", "b", "lb", "ub"])
    # print(models)
    ols = smf.ols(f"{ycol} ~ {xcol}", data).fit()
    ols_ci = ols.conf_int().loc[xcol].tolist()
    intercept = ols.params["Intercept"]
    ols = dict(a=ols.params["Intercept"], b=ols.params[xcol], lb=ols_ci[0], ub=ols_ci[1])
    if not np.isnan(a_intercept):
        if intercept < a_intercept:
            ols = smf.ols(f"{ycol} ~ {xcol} -1", data).fit()
            ols_ci = ols.conf_int().loc[xcol].tolist()
            ols = dict(a=a_intercept, b=ols.params[xcol], lb=ols_ci[0], ub=ols_ci[1])

    increment = (data[xcol].max() - data[xcol].min())/20
    xx = np.arange(data[xcol].min(), data[xcol].max()+increment, increment)

    get_y = lambda a, b: a + xx * b  # a = cohesion, b = np.tan(phi)

    for i in range(models.shape[0]):
        yy = get_y(models.a[i], models.b[i])
        if models.q[i] == lq:
            dlq = pd.DataFrame({'x': xx, 'y': yy})
            # lq_pct = int(models.q[i]*100)
            lq_a = models.a[i]
            lq_b = models.b[i]

        elif models.q[i] == uq:
            duq = pd.DataFrame({'x': xx, 'y': yy})
            # uq_pct = int(models.q[i]*100)
            uq_a = models.a[i]
            uq_b = models.b[i]

    yy = get_y(ols["a"], ols["b"])
    base_a = ols["a"]
    base_b = ols["b"]

    dbs = pd.DataFrame({'x': xx, 'y': yy})
    return dbs, base_a, base_b, dlq, lq_a, lq_b, duq, uq_a, uq_b, lq, uq

def scipy_models(data, a_intercept, ci, fitmethod, fit_selection, inp_jrc, inp_jcs):
    data.columns = ["NormalStress", "ShearStress"]
    x = data['NormalStress']
    y = data['ShearStress']
    ci_n = ci / 100
    lq = 0.5 - ci_n / 2
    uq = 0.5 + ci_n / 2
    # Convert to percentile point of the normal distribution.
    # See: https://en.wikipedia.org/wiki/Standard_score
    pp = (1. + ci_n) / 2.
    nstd = stats.norm.ppf(pp)

    if fitmethod == fit_selection[0]:
        popt, pcov = curve_fit(weakfunc, x, y)
    elif fitmethod == fit_selection[1]:
        popt, pcov = curve_fit(bartonbandis, x, y, inp_jrc, inp_jcs)
    else:
        popt, pcov = curve_fit(weakfunc, x, y)
    print(popt, nstd)
    perr = sqrt(diag(pcov))
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr

    x_line = np.arange(min(x), max(x), 1)
    dbs = pd.DataFrame({'x':x_line})
    dlq = pd.DataFrame({'x':x_line})
    duq = pd.DataFrame({'x':x_line})
    bs_a, bs_b = popt
    lq_a, lq_b = popt_dw
    uq_a, uq_b = popt_up
    print(fitmethod)
    if fitmethod == fit_selection[0]:
        dbs['y'] = dbs.apply(lambda x_l: weakfunc(x_l,*popt))
        dlq['y'] = dlq.apply(lambda x_l: weakfunc(x_l,*popt_dw))
        duq['y'] = duq.apply(lambda x_l: weakfunc(x_l,*popt_up))
    elif fitmethod == fit_selection[1]:
        dbs['y'] = dbs.apply(lambda x_l: bartonbandis(x_l,*popt))
        dlq['y'] = dlq.apply(lambda x_l: bartonbandis(x_l,*popt_dw))
        duq['y'] = duq.apply(lambda x_l: bartonbandis(x_l,*popt_up))
    else:
        dbs['y'] = dbs.apply(lambda x_l: weakfunc(x_l,*popt))
        dlq['y'] = dlq.apply(lambda x_l: weakfunc(x_l,*popt_dw))
        duq['y'] = duq.apply(lambda x_l: weakfunc(x_l,*popt_up))

    return dbs, bs_a, bs_b, dlq, lq_a, lq_b, duq, uq_a, uq_b, lq, uq
