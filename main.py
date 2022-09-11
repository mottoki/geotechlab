import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf

from numpy import sin, cos, tan, arcsin, arccos, arctan, radians, degrees, sqrt, diag, log10, log, exp

from streamlit_option_menu import option_menu

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, Band
import bokeh.models as bmo

from st_aggrid import AgGrid

# ------------------ INPUT --------------------------------
base_colors = {
    'background': '#616161',
    'grid': '#7E7E7E',
    'line': '#808080', #'#FFFAF1',
    'text': '#F9F9F3'
}

colors = px.colors.qualitative.T10

st.set_page_config(layout="wide")

hide_table_row_index = """
    <style>
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """

st.markdown(hide_table_row_index, unsafe_allow_html=True)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ------------------ FUNCTION ----------------------------------
# Direct Shear
def weakfunc(x, a, b):
    # shear stress = cohesion + normal stress * tan(friction_angle)
    # x = Normal Stress, a = cohesion, b = friction angle
    return a + x * np.tan(b)

def powercurve(x, k, m):
    return k * (x ** m)

# -------------------- Sidebar -------------------------------------------
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)

    fig_selection = ['Plotly', 'Bokeh']
    fig_method = st.sidebar.radio("Figure type",
        fig_selection, horizontal=True)

else:
    df = None

if df is not None:
    # -------------------- Tabs -----------------------------------------------
    # tab1, tab2 = st.tabs(["Defect Shear Strength", "Rockmass Shear Strength"])
    options = ['Defect Shear Strength', 'Rockmass Shear Strength']
    selected = option_menu(
        menu_title=None,
        options=options,
        default_index=0, orientation='horizontal')

    # ------------ DST -------------------
    # with tab1:
    if selected ==options[0]:
        # Selections

        df1 = df[df['SampleType']=='DST']
        df1['TestStage'] = pd.to_numeric(df1['TestStage'], downcast='integer', errors='coerce')

        # Additional Filter
        if len(df1)>0:
            color_list = ['Shear Plane Type', 'Rock Type', 'HoleID', 'TestStage']
            p_max_o = int(df1['NormalStress'].max())
            p_max = st.sidebar.number_input('Maximum Normal Stress', value=p_max_o)
            df1 = df1[df1['NormalStress']<=p_max]

            if "Project" in df1.columns:
                project = set(df1['Project'])
                project_selection = st.sidebar.multiselect("Project", (project))
                if project_selection: df1 = df1[df1['Project'].isin(project_selection)]

            if "Prospect" in df1.columns:
                prospect = set(df1['Prospect'])
                prospect_selection = st.sidebar.multiselect("Prospect", (prospect))
                if prospect_selection: df1 = df1[df1['Prospect'].isin(prospect_selection)]
                color_list.append("Prospect")

            if "Formation" in df1.columns:
                formation = set(df1['Formation'])
                formation_selection = st.sidebar.multiselect("Formation", (formation))
                if formation_selection: df1 = df1[df1['Formation'].isin(formation_selection)]

            if "SubFormation" in df1.columns:
                subformation = set(df1['SubFormation'])
                subformation_selection = st.sidebar.multiselect("SubFormation", (subformation))
                if subformation_selection: df1 = df1[df1['SubFormation'].isin(subformation_selection)]

            rock_type = set(df1['Rock Type'])
            rock_selection = st.sidebar.multiselect("Rock Type", (rock_type))
            if rock_selection: df1 = df1[df1['Rock Type'].isin(rock_selection)]

            holeid = set(df1['HoleID'])
            holeid_selection = st.sidebar.multiselect("Hole ID", (holeid))
            if holeid_selection: df1 = df1[df1['HoleID'].isin(holeid_selection)]

            sampletype = set(df1['SampleType'])
            testtype_selection = st.sidebar.multiselect("Test type", (sampletype))
            if testtype_selection: df1 = df1[df1['SampleType'].isin(testtype_selection)]

            teststage = (x for x in set(df1['TestStage']) if np.isnan(x) == False)
            teststage_selection = st.sidebar.multiselect("Test Stage", (teststage))

            shear_type = set(df1['Shear Plane Type'])
            sheartype_selection = st.sidebar.multiselect("Shear Plane Type", (shear_type))

            if teststage_selection: df1 = df1[df1['TestStage'].isin(teststage_selection)]
            if sheartype_selection: df1 = df1[df1['Shear Plane Type'].isin(sheartype_selection)]

            if "Test Year" in df1.columns:
                testyear = set(df1['Test Year'])
                testyear_selection = st.sidebar.multiselect("Test Year", (testyear))
                if testyear_selection: df1 = df1[df1['Test Year'].isin(testyear_selection)]

        x = df1["NormalStress"]
        y = df1["ShearStress"]

        col4, col5 = st.columns(2)

        fit_selection = ('Linear', 'Barton Bandis', 'Power')
        with col4:
            fitmethod = st.radio("Auto Fit Method",
                fit_selection, horizontal=True)
        with col5:
            colormethod_d = st.radio("Color By",
                color_list, horizontal=True)

        if fitmethod == fit_selection[1]:
            col1, col2 = st.columns(2)
            with col1:
                    inp_jrc = st.number_input('Enter jrc', value=2.5)
            with col2:
                    inp_jcs = st.number_input('Enter jcs', value=100)

            col1, col2 = st.columns(2)
            with col1:
                lq_sd = st.number_input('Lower Phir Std Dev', value=1, step=1)
            with col2:
                uq_sd = st.number_input('Upper Phir Std Dev', value=1, step=1)

        else:
            col1, col2 = st.columns(2)
            with col1:
                lq_value = st.number_input('Lower Quantile (%)', value=25, step=5)
            with col2:
                uq_value = st.number_input('Upper Quantile (%)', value=75, step=5)

        col1, col2 = st.columns(2)

        if fitmethod == fit_selection[0]:
            negative_coh_selection = ("On", "Set cohesion")
            with col1:
                negative_coh_method = st.radio("Automatic Cohesion",
                    negative_coh_selection, horizontal=True)
            if negative_coh_method == negative_coh_selection[1]:
                with col2:
                    intercept_coh = st.number_input('Set Cohesion', value=0, step=5)
            else:
                intercept_coh = np.nan
        else:
            negative_k_selection = ("On", "Set k")
            with col1:
                negative_k_method = st.radio("Automatic k",
                    negative_k_selection, horizontal=True)
            if negative_k_method == negative_k_selection[1]:
                with col2:
                    intercept_k = st.number_input('Set k', step=0.,format="%.2f")
            else:
                intercept_k = np.nan

        def fit_model(data, q, a_intercept):
            if fitmethod != fit_selection[1]:
                mod = smf.quantreg("ShearStress ~ NormalStress", data)
                res = mod.fit(q=q)
                if not np.isnan(a_intercept):
                    if res.params["Intercept"] < a_intercept:
                        mod = smf.quantreg("ShearStress ~ NormalStress -1", data)
                        res = mod.fit(q=q)
                        return [q, a_intercept, res.params["NormalStress"]] + res.conf_int().loc["NormalStress"].tolist()
                return [q, res.params["Intercept"], res.params["NormalStress"]] + res.conf_int().loc["NormalStress"].tolist()
            # else:
            #     mod = smf.quantreg("ShearStress ~ NormalStress + I(tan(radians(inp_jrc + log10(inp_jcs/NormalStress))))", data)
            #     res = mod.fit(q=q)
            #     return [q, a_intercept, res.params["NormalStress"]] + res.conf_int().loc["NormalStress"].tolist()

        def quantile_models(data, a_intercept):
            data.columns = ["NormalStress", "ShearStress"]
            quantiles = np.arange(lq_value/100, uq_value/100 + 0.05, (uq_value-lq_value)/100)

            models = [fit_model(data, x, a_intercept) for x in quantiles]
            models = pd.DataFrame(models, columns=["q", "a", "b", "lb", "ub"])
            # print(models)
            ols = smf.ols("ShearStress ~ NormalStress", data).fit()
            ols_ci = ols.conf_int().loc["NormalStress"].tolist()
            intercept = ols.params["Intercept"]
            ols = dict(a=ols.params["Intercept"], b=ols.params["NormalStress"], lb=ols_ci[0], ub=ols_ci[1])
            if not np.isnan(a_intercept):
                if intercept < a_intercept:
                    ols = smf.ols("ShearStress ~ NormalStress -1", data).fit()
                    ols_ci = ols.conf_int().loc["NormalStress"].tolist()
                    ols = dict(a=a_intercept, b=ols.params["NormalStress"], lb=ols_ci[0], ub=ols_ci[1])

            increment = (data.NormalStress.max() - data.NormalStress.min())/20
            xx = np.arange(data.NormalStress.min(), data.NormalStress.max()+increment, increment)

            if fitmethod != fit_selection[1]:
                get_y = lambda a, b: a + xx * b  # a = cohesion, b = np.tan(phi)
            else:
                get_y = lambda a, b: a + xx * tan(radians(b + inp_jrc * log10(inp_jcs/xx))) # a = 0, b = phir

            for i in range(models.shape[0]):
                yy = get_y(models.a[i], models.b[i])
                if models.q[i] == lq_value/100:
                    dlq = pd.DataFrame({'x': xx, 'y': yy})
                    # lq_pct = int(models.q[i]*100)
                    lq_a = models.a[i]
                    lq_b = models.b[i]

                elif models.q[i] == uq_value/100:
                    duq = pd.DataFrame({'x': xx, 'y': yy})
                    # uq_pct = int(models.q[i]*100)
                    uq_a = models.a[i]
                    uq_b = models.b[i]

            yy = get_y(ols["a"], ols["b"])
            base_a = ols["a"]
            base_b = ols["b"]

            dbs = pd.DataFrame({'x': xx, 'y': yy})
            return dbs, base_a, base_b, dlq, lq_a, lq_b, duq, uq_a, uq_b

        if len(x) > 0:
            if fitmethod == fit_selection[0]:
                data = df1[["NormalStress", "ShearStress"]]
                a_intercept = intercept_coh

                dbs, bs_a, bs_b, dlq, lq_a, lq_b, duq, uq_a, uq_b = quantile_models(data, a_intercept)

                bs_a = round(bs_a,2)
                bs_b = round(degrees(arctan(bs_b)),2)

                lq_a = round(lq_a,2)
                lq_b = round(degrees(arctan(lq_b)),2)

                uq_a = round(uq_a,2)
                uq_b = round(degrees(arctan(uq_b)),2)

                val_1 = 'Cohesion'
                val_2 = 'Friction Angle'

                # Manual Fit
                col1, col2, col3 = st.columns(3)
                with col1:
                    manual_on_p = st.radio("Manual Fit?",
                        ('off', 'on'), horizontal=True)
                if manual_on_p == 'on':
                    # col1, col2 = st.columns(2)
                    with col2:
                        d_coh = st.number_input('Manual cohesion', value=bs_a)
                    with col3:
                        d_fric = st.number_input('Manual friction angle', value=bs_b)

                    sigN = list(range(int(min(x)), int(max(x)), 1))
                    dman = pd.DataFrame({'sigN':sigN})
                    dman['sigT'] = dman.apply(lambda row: d_coh+row["sigN"]*tan(radians(d_fric)), axis=1)
                    params_man = [d_coh, radians(d_fric)]
                    residuals = y - weakfunc(x,*params_man)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y-np.mean(y))**2)
                    r_sq_dst_m = 1 - (ss_res/ss_tot)
                    manual_val_1 = d_coh
                    manual_val_2 = d_fric
                    manual_val_3 = round(r_sq_dst_m,4)
                else:
                    manual_val_1 = np.nan

            # Barton Bandis:
            elif fitmethod == fit_selection[1]:

                def bartonbandis(x, phir):
                    return x * tan(radians(phir + inp_jrc * log10(inp_jcs / x)))

                data = df1[["NormalStress", "ShearStress"]]
                a_intercept = 0

                # dbs, bs_a, bs_b, dlq, lq_a, lq_b, duq, uq_a, uq_b = quantile_models(data, a_intercept)
                # print(bs_a, bs_b)
                # Auto Fit
                x0 = np.array([25])
                params, params_covariance = curve_fit(bartonbandis, x, y,p0=x0, bounds=((0),(90)))

                auto_phir = params[0] # automatic phir, jrc, jcs
                # signmin = 10**(log10(inp_jcs)-((70-auto_phir)/inp_jrc))
                # print(auto_phir, signmin)
                # R squated Calculation
                residuals = y - bartonbandis(x,*params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r_sq_dst = 1 - (ss_res/ss_tot)

                x_line = np.arange(1, max(x), 1)
                fit_curve = pd.DataFrame({'x_line':x_line})
                fit_curve['y_line'] = fit_curve.apply(lambda x_l: bartonbandis(x_l,auto_phir))

                # One standard deviation
                sd_phir = sqrt(diag(params_covariance)[0])
                phir_sd_low = auto_phir - sd_phir * lq_sd

                sd_low_curve = pd.DataFrame({'x_line':x_line})
                sd_low_curve['y_line'] = sd_low_curve.apply(lambda x_l: bartonbandis(x_l,phir_sd_low))
                phir_sd_high = auto_phir + sd_phir * uq_sd

                sd_high_curve = pd.DataFrame({'x_line':x_line})
                sd_high_curve['y_line'] = sd_high_curve.apply(lambda x_l: bartonbandis(x_l,phir_sd_high))

                # Clean it up for table
                auto_phir = round(auto_phir,2)
                phir_sd_low = round(phir_sd_low,2)
                phir_sd_high = round(phir_sd_high,2)
                auto_jrc = round(inp_jrc,2)
                auto_jcs = round(inp_jcs,2)
                r_sq_dst = round(r_sq_dst,4)
                sd_phir = round(sd_phir,2)
                sd_jrc = np.nan
                sd_jcs = np.nan
                val_1 = 'Phir'
                val_2 = 'JRC'
                val_3 = 'JCS'

                # Manual Fit
                col1, col2 = st.columns(2)
                with col1:
                    manual_on_p = st.radio("Manual Fit?",
                        ('off', 'on'), horizontal=True)
                if manual_on_p == 'on':
                    with col2:
                        d_phir = st.number_input('Manual phir', value=auto_phir)

                    sigN = list(range(int(min(x)), int(max(x)), 1))
                    dman = pd.DataFrame({'sigN':sigN})
                    dman['sigT'] = dman.apply(lambda row: bartonbandis(row['sigN'], d_phir), axis=1)
                    params_man = [d_phir]
                    residuals = y - bartonbandis(x,*params_man)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y-np.mean(y))**2)
                    r_sq_dst_m = 1 - (ss_res/ss_tot)
                    manual_val_1 = round(d_phir,2)
                    manual_val_2 = inp_jrc
                    manual_val_3 = inp_jcs
                    manual_val_4 = round(r_sq_dst_m,4)
                else:
                    manual_val_1 = np.nan

            # Power Fit
            elif fitmethod == fit_selection[2]:
                df1["Log_NormalStress"] = log(df1["NormalStress"])
                df1["Log_ShearStress"] = log(df1["ShearStress"])
                data = df1[["Log_NormalStress", "Log_ShearStress"]]
                data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
                # data = data[data['Log_NormalStress'].notna()]
                # data = data[data['Log_ShearStress'].notna()]

                a_intercept = log(intercept_k)

                dbs, bs_a, bs_b, dlq, lq_a, lq_b, duq, uq_a, uq_b = quantile_models(data, a_intercept)

                dbs['x'] = exp(dbs['x'])
                dbs['y'] = exp(dbs['y'])
                bs_a = round(exp(bs_a), 4)
                bs_b = round(bs_b, 4)

                dlq['x'] = exp(dlq['x'])
                dlq['y'] = exp(dlq['y'])
                lq_a = round(exp(lq_a), 4)
                lq_b = round(lq_b, 4)

                duq['x'] = exp(duq['x'])
                duq['y'] = exp(duq['y'])
                uq_a = round(exp(uq_a), 4)
                uq_b = round(uq_b, 4)

                val_1 = 'k'
                val_2 = 'm'

                # Manual Fit
                col1, col2, col3 = st.columns(3)
                with col1:
                    manual_on_p = st.radio("Manual Fit?",
                        ('off', 'on'), horizontal=True)
                if manual_on_p == 'on':
                    # col1, col2 = st.columns(2)
                    with col2:
                        k_pow = st.number_input('Manual k', value=bs_a)
                    with col3:
                        m_pow = st.number_input('Manual m', value=bs_b)

                    sigN = list(range(int(min(x)), int(max(x)), 1))
                    dman = pd.DataFrame({'sigN':sigN})
                    dman['sigT'] = dman.apply(lambda row: k_pow*(row["sigN"]**m_pow), axis=1)
                    params_man = [k_pow, m_pow]
                    residuals = y - powercurve(x,*params_man)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y-np.mean(y))**2)
                    r_sq_dst_m = 1 - (ss_res/ss_tot)
                    manual_val_1 = k_pow
                    manual_val_2 = m_pow
                    manual_val_3 = round(r_sq_dst_m)
                else:
                    manual_val_1 = np.nan

            if fig_method == fig_selection[0]:
                figd = px.scatter(
                    df1, x="NormalStress", y="ShearStress",
                    color=colormethod_d, color_discrete_sequence=colors)

                figd.update_traces(marker=dict(size=9))

                if not np.isnan(manual_val_1):
                    figd.add_trace(go.Scatter(x=dman["sigN"], y=dman["sigT"],
                        mode='lines', name='Manual Fit',
                        line=dict(dash='dash', color=colors[8])))

                # Barton Bandis
                if fitmethod == fit_selection[1]:
                    figd.add_trace(go.Scatter(x=fit_curve['x_line'], y=fit_curve['y_line'],
                        mode='lines', name=f'{fitmethod} Curve Fit',
                        line=dict(color='grey')))

                    figd.add_trace(go.Scatter(x=sd_low_curve['x_line'], y=sd_low_curve['y_line'],
                        mode='lines', name=f'-{lq_sd} Phir St.D',
                        line=dict(dash='dash', color=colors[6])))

                    figd.add_trace(go.Scatter(x=sd_high_curve['x_line'], y=sd_high_curve['y_line'],
                        mode='lines', name=f'+{uq_sd} Phir St.D',
                        line=dict(dash='dash', color=colors[5])))

                # Linear and Power
                else:
                    figd.add_trace(go.Scatter(x=dbs['x'], y=dbs['y'],
                        mode='lines', name=f'OLS',
                        line=dict(color='grey')))

                    figd.add_trace(go.Scatter(x=dlq['x'], y=dlq['y'],
                        mode='lines', name=f'{lq_value}% Bound',
                        line=dict(dash='dash', color=colors[6])))

                    figd.add_trace(go.Scatter(x=duq['x'], y=duq['y'],
                        mode='lines', name=f'{uq_value}% Bound',
                        line=dict(dash='dash', color=colors[5])))

                num_dst = len(df1.index)
                figd.update_layout(
                    title_text=f"No. of Data: {num_dst}",
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    height=600,)

                figd.update_xaxes(title_text='Normal Stress', gridcolor='lightgrey',
                    zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
                    tickformat=",.0f")
                figd.update_yaxes(title_text='Shear Stress', gridcolor='lightgrey',
                    zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
                    tickformat=",.0f", range=[0,max(y)*1.05])
                figd.add_shape(
                    type="rect", xref="paper", yref="paper",
                    x0=0, y0=0, x1=1.0, y1=1.0,
                    line=dict(color="black", width=2))
            else:
                # Bokeh graph
                uniq = df1[colormethod_d].unique()
                color_map = bmo.CategoricalColorMapper(factors=uniq, palette=colors)
                source = ColumnDataSource(df1)
                hover = HoverTool(tooltips=[
                    ('Hole ID', '@HoleID'),
                    ("Normal Stress", "@NormalStress"),
                    ("Shear Stress", "@ShearStress"),
                    ])
                p=figure(tools=[hover], x_axis_label='Normal Stress', y_axis_label='Shear Stress')

                p.scatter(x='NormalStress', y='ShearStress', size=9,
                    color={'field': colormethod_d, 'transform': color_map},
                    legend_group=colormethod_d, source=source)

                # Barton Bandis
                if fitmethod == fit_selection[1]:
                    source = ColumnDataSource(fit_curve)
                    p.line(x='x_line', y='y_line', line_width=2, line_color='grey',
                        legend_label=f'{fitmethod} Curve Fit', source=source)

                    source = ColumnDataSource(sd_low_curve)
                    p.line(x='x_line', y='y_line', line_width=2, line_color=colors[6],
                        legend_label=f'-{lq_sd} Phir St.D', source=source)

                    source = ColumnDataSource(sd_high_curve)
                    p.line(x='x_line', y='y_line', line_width=2, line_color=colors[5],
                        legend_label=f'+{uq_sd} Phir St.D', source=source)

                # Linear and Power
                else:
                    source = ColumnDataSource(dbs)
                    p.line(x='x', y='y', line_width=2, line_color='grey',
                        legend_label=f'OLS', source=source)

                    source = ColumnDataSource(dlq)
                    p.line(x='x', y='y', line_width=2, line_color=colors[6],
                        legend_label=f'{lq_value}% Bound', source=source)

                    source = ColumnDataSource(duq)
                    p.line(x='x', y='y', line_width=2, line_color=colors[5],
                        legend_label=f'{uq_value}% Bound', source=source)

                if manual_on_p == 'on':
                    source = ColumnDataSource(dman)
                    p.line(x='sigN', y='sigT', line_width=2, line_color='green',
                        legend_label='Manual Fit', source=source)

                p.legend.location = "top_left"

            # Table
            if fitmethod != fit_selection[1]:
                dst_summary = pd.DataFrame(columns=["Method", val_1, val_2])
                to_append = ["Auto Fit - OLS", bs_a, bs_b]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                to_append = [f"Auto Fit - {lq_value}%", lq_a, lq_b]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                to_append = [f"Auto Fit - {uq_value}%", uq_a, uq_b]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                if not np.isnan(manual_val_1):
                    to_append = ['Manual Fit', manual_val_1, manual_val_2]
                    new_row = len(dst_summary)
                    dst_summary.loc[new_row] = to_append
            else:
                dst_summary = pd.DataFrame(columns=[" ", val_1, val_2, val_3])
                to_append = ["Auto Fit: BB Curve", auto_phir, auto_jrc, auto_jcs]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                to_append = [f"Auto Fit: -{lq_sd} Phir ST.D", phir_sd_low, auto_jrc, auto_jcs]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                to_append = [f"Auto Fit: +{uq_sd} Phir ST.D", phir_sd_high, auto_jrc, auto_jcs]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                if not np.isnan(manual_val_1):
                    to_append = ['Manual Fit', manual_val_1, manual_val_2, manual_val_3]
                    new_row = len(dst_summary)
                    dst_summary.loc[new_row] = to_append

        else:
            figd = go.Figure().add_annotation(
                x=2, y=2,text="No Data to Display",
                font=dict(
                    family="sans serif",size=25,color="crimson"),
                showarrow=False,yshift=10)
            dst_summary = pd.DataFrame()

        if fig_method==fig_selection[0]:
            st.plotly_chart(figd, use_container_width=True)
        else:
            st.bokeh_chart(p, use_container_width=True)

        st.markdown("**Shear Strength**")
        # st.dataframe(dst_summary.style.format(precision=2)) #style.set_precision(2))

        height = 32 + 28*len(dst_summary.index)
        AgGrid(dst_summary, fit_columns_on_grid_load=True, height=height)

        st.subheader("Dataset")
        # st.dataframe(df1[['HoleID', 'Rock Type', 'TestStage', 'NormalStress', 'ShearStress', 'Shear Plane Type']])
        # st.dataframe(df1[1:])
        df1 = df1.iloc[:,1:]
        df1 = df1.dropna(axis=1, how='all')
        AgGrid(df1)
    # ------------ UCS and Rock TXL -------------------
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
        Sigma3 = list(range(c_tensile, int(x.max()+0.15*x.max()+1)))
        c_curv = pd.DataFrame({'Sigma3':Sigma3})
        c_curv['Sigma1'] = c_curv.apply(lambda row: row['Sigma3']+c_sigci*sqrt((c_mi*row['Sigma3']/c_sigci)+1), axis=1)

        # Vertical Line for Tensile Cutoff
        c_vert = c_curv[c_curv['Sigma3']==c_tensile]
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

    # with tab2:
    if selected == options[1]:

        method_selection = ('Hoek Calculation', 'Scipy Curve Fit')
        calc_selection = ('All data', 'HTX + UCS(mean)', 'HTX only')
        color_list = ['SampleType', 'Rock Type', 'HoleID', 'Failure Mode']

        du = df[df['SampleType'].isin(['UCS', 'HTX', 'BZT'])]
        du = du[du['Sigma3'].notna()]
        du = du[du['PeakSigma1'].notna()]
        # print(du)

        if len(du['Sigma3'])>0:
            p_max_o = int(du['Sigma3'].max())
            p_max = st.sidebar.number_input('Maximum Sigma3', value=p_max_o)
            du = du[du['Sigma3']<=p_max]

            if "Project" in du.columns:
                project = set(du['Project'])
                project_selection = st.sidebar.multiselect("Project", (project))
                if project_selection: du = du[du['Project'].isin(project_selection)]

            if "Prospect" in du.columns:
                prospect = set(du['Prospect'])
                prospect_selection = st.sidebar.multiselect("Prospect", (prospect))
                if prospect_selection: du = du[du['Prospect'].isin(prospect_selection)]
                color_list.append("Prospect")

            if "Formation" in du.columns:
                formation = set(du['Formation'])
                formation_selection = st.sidebar.multiselect("Formation", (formation))
                if formation_selection: du = du[du['Formation'].isin(formation_selection)]

            if "SubFormation" in du.columns:
                subformation = set(du['SubFormation'])
                subformation_selection = st.sidebar.multiselect("SubFormation", (subformation))
                if subformation_selection: du = du[du['SubFormation'].isin(subformation_selection)]

            rock_type = set(du['Rock Type'])
            rock_selection = st.sidebar.multiselect("Rock Type", (rock_type))
            if rock_selection: du = du[du['Rock Type'].isin(rock_selection)]

            holeid = set(du['HoleID'])
            holeid_selection = st.sidebar.multiselect("Hole ID", (holeid))
            if holeid_selection: du = du[du['HoleID'].isin(holeid_selection)]

            sampletype = set(du['SampleType'])
            testtype_selection = st.sidebar.multiselect("Test type", (sampletype))
            if testtype_selection: du = du[du['SampleType'].isin(testtype_selection)]

            failure_mode = set(du['Failure Mode'])
            failuremode_selection = st.sidebar.multiselect("Failure Mode", (failure_mode))
            if failuremode_selection: du = du[du['Failure Mode'].isin(failuremode_selection)]

        # Auto Fit - Method and dataset
        col4, col5, col6 = st.columns(3)
        with col4:
            calc_method = st.radio("Auto Fit method",
                method_selection, horizontal=True)
        with col5:
            calc_data = st.radio("Select dataset",
                calc_selection, horizontal=True)
        with col6:
            # Figure
            colormethod_u = st.radio("Color By",
                color_list,
                horizontal=True)

        col1, col2 = st.columns(2)
        with col1:
            lq_value = st.number_input('Lower Standard Deviation', value=1, step=1)
        with col2:
            uq_value = st.number_input('Upper Standard Deviation', value=1, step=1)

        if len(du['Sigma3']) > 0:
            def lin_reg(x, a, b):
                return a + b*x

            def get_linear_df(a, b, p_cov, x, y):
                # For Figure
                t_fri = arcsin((b-1)/(b+1))
                t_coh = a * (1-sin(t_fri))/(2*cos(t_fri))
                min_val = int((max(du['Sigma3'])-min(du['Sigma3']))/40)
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
            # # Quantreg
            # def fit_model(data, q, a_intercept):
            #     mod = smf.quantreg("PeakSigma1 ~ Sigma3", data)
            #     res = mod.fit(q=q)
            #     if not np.isnan(a_intercept):
            #         if res.params["Intercept"] < a_intercept:
            #             mod = smf.quantreg("PeakSigma1 ~ Sigma3 -1", data)
            #             res = mod.fit(q=q)
            #             return [q, a_intercept, res.params["Sigma3"]] + res.conf_int().loc["Sigma3"].tolist()
            #     return [q, res.params["Intercept"], res.params["Sigma3"]] + res.conf_int().loc["Sigma3"].tolist()

            # def quantile_models(data, a_intercept):
            #     data.columns = ["Sigma3", "PeakSigma1"]
            #     quantiles = np.arange(lq_value/100, uq_value/100 + 0.05, (uq_value-lq_value)/100)

            #     models = [fit_model(data, x, a_intercept) for x in quantiles]
            #     models = pd.DataFrame(models, columns=["q", "a", "b", "lb", "ub"])
            #     # print(models)
            #     ols = smf.ols("PeakSigma1 ~ Sigma3", data).fit()
            #     ols_ci = ols.conf_int().loc["Sigma3"].tolist()
            #     intercept = ols.params["Intercept"]
            #     ols = dict(a=ols.params["Intercept"], b=ols.params["Sigma3"], lb=ols_ci[0], ub=ols_ci[1])
            #     # print(ols)
            #     if not np.isnan(a_intercept):
            #         if intercept < a_intercept:
            #             ols = smf.ols("PeakSigma1 ~ Sigma3 -1", data).fit()
            #             ols_ci = ols.conf_int().loc["Sigma3"].tolist()
            #             ols = dict(a=a_intercept, b=ols.params["Sigma3"], lb=ols_ci[0], ub=ols_ci[1])

            #     increment = (data.Sigma3.max() - data.Sigma3.min())/20
            #     xx = np.arange(data.Sigma3.min(), data.Sigma3.max()+increment, increment)

            #     get_y = lambda a, b: a + xx * b  # a = cohesion, b = np.tan(phi)

            #     for i in range(models.shape[0]):
            #         yy = get_y(models.a[i], models.b[i])
            #         if models.q[i] == lq_value/100:
            #             dlq = pd.DataFrame({'x': xx, 'y': yy})
            #             lq_a = models.a[i]
            #             lq_b = models.b[i]

            #         elif models.q[i] == uq_value/100:
            #             duq = pd.DataFrame({'x': xx, 'y': yy})
            #             uq_a = models.a[i]
            #             uq_b = models.b[i]

            #     yy = get_y(ols["a"], ols["b"])
            #     base_a = ols["a"]
            #     base_b = ols["b"]

            #     dbs = pd.DataFrame({'x': xx, 'y': yy})
            #     return dbs, base_a, base_b, dlq, lq_a, lq_b, duq, uq_a, uq_b

            # Calculate mean UCS and BZT test results
            if not du[du['Sigma3']==0].empty:
                mean_ucs = int(du[du['Sigma3']==0]['PeakSigma1'].mean())
            else:
                mean_ucs = 0
            if not du[du['PeakSigma1']==0].empty:
                mean_bzt = int(du[du['PeakSigma1']==0]['Sigma3'].mean())
            else:
                mean_bzt = 0

            # Append Mean UCS
            db = du[du['PeakSigma1']!=0]
            dt = db[db['Sigma3']!=0]
            du_auto = dt[['Sigma3', 'PeakSigma1']]

            to_append = [0, mean_ucs]
            ducs_mean = pd.DataFrame(to_append).transpose()
            ducs_mean.columns=['Sigma3', 'PeakSigma1']
            new_row = len(du_auto)
            du_auto.loc[new_row] = to_append

            # Append Mean Brazilian
            to_append = [mean_bzt, 0]
            dbzt_mean = pd.DataFrame(to_append).transpose()
            dbzt_mean.columns=['Sigma3', 'PeakSigma1']

            # x and y dataset for auto fit
            if calc_data==calc_selection[0]:
                sig3 = db["Sigma3"]
                sig1 = db["PeakSigma1"]
                data = db[["Sigma3", "PeakSigma1"]]
            elif calc_data==calc_selection[1]:
                sig3 = du_auto["Sigma3"]
                sig1 = du_auto["PeakSigma1"]
                data = du_auto[["Sigma3", "PeakSigma1"]]
            else:
                sig3 = dt["Sigma3"]
                sig1 = dt["PeakSigma1"]
                data = dt[["Sigma3", "PeakSigma1"]]

            # Auto Fit Main part
            if calc_method == method_selection[0]:
                # if calc_data==calc_selection[0]:
                # Calculation - ALL
                y = (sig1 - sig3)**2
                xy = sig3 * y
                xsq = sig3**2
                ysq = sig1**2

                sumx = sig3.sum()
                sumy = y.sum()
                sumxy = xy.sum()
                sumxsq = xsq.sum()
                sumysq = ysq.sum()
                n = len(sig3.index)

                auto_sigci, auto_mi, auto_r_sq = calc_hoek(sumx, sumy, sumxy, sumxsq, sumysq, n)

                c_tens, c_curv, c_vert, r_sq_c = get_curve_df(auto_sigci, auto_mi, sig3, sig1)

                # Linear
                # params, params_covariance = curve_fit(lin_reg, sig3, sig1)
                # a, b = params
                # t_coh, t_fri, r_sq_l, c_line = get_linear_df(a, b, params_covariance, sig3, sig1)

            else:
                params, params_covariance = curve_fit(objective, sig3, sig1)
                auto_sigci, auto_mi = params # a = sigci, b = mi
                auto_sigci = round(auto_sigci,2)
                auto_mi = round(auto_mi,2)

                c_tens, c_curv, c_vert, r_sq_c = get_curve_df(auto_sigci, auto_mi, sig3, sig1)

            # Linear
            params, params_covariance = curve_fit(lin_reg, sig3, sig1)
            a, b = params
            # print(a,b)
            bs_a, bs_b, lq_a, lq_b, uq_a, uq_b, r_sq_l, c_line = get_linear_df(a, b, params_covariance, sig3, sig1)
            bs_a = round(bs_a, 2)
            bs_b = round(bs_b, 2)
            lq_a = round(lq_a, 2)
            lq_b = round(lq_b, 2)
            uq_a = round(uq_a, 2)
            uq_b = round(uq_b, 2)
            # print(t_coh,t_fri)

            # a_intercept = np.nan
            # # print(data.head())
            # dbs, bs_a, bs_b, dlq, lq_a, lq_b, duq, uq_a, uq_b = quantile_models(data, a_intercept)

            # bs_b = arcsin((bs_b-1)/(bs_b+1))
            # bs_a = bs_a * (1-sin(bs_b))/(2*cos(bs_b))
            # bs_b = round(degrees(bs_b), 2)
            # bs_a = round(bs_a, 2)

            # lq_b = arcsin((lq_b-1)/(lq_b+1))
            # lq_a = lq_a * (1-sin(lq_b))/(2*cos(lq_b))
            # lq_b = round(degrees(lq_b), 2)
            # lq_a = round(lq_a, 2)

            # uq_b = arcsin((uq_b-1)/(uq_b+1))
            # uq_a = uq_a * (1-sin(uq_b))/(2*cos(uq_b))
            # uq_b = round(degrees(uq_b), 2)
            # uq_a = round(uq_a, 2)

            # print(dbs)
            # Calculated Figure
            figu = px.scatter(
                du, x="Sigma3", y="PeakSigma1",
                color=colormethod_u, color_discrete_sequence=colors)
            figu.update_traces(marker=dict(size=9))

            # figu.add_trace(
            #     go.Scatter(x=dbs['x'], y=dbs['y'],
            #         mode='lines', name=f'Linear Regression',
            #         line=dict(color=colors[-1])))

            # figu.add_trace(
            #     go.Scatter(x=dlq['x'], y=dlq['y'],
            #         mode='lines', name=f'{int(lq_value)}% Bound',
            #         line=dict(dash='dash', color=colors[6])))

            # figu.add_trace(
            #     go.Scatter(x=duq['x'], y=duq['y'],
            #         mode='lines', name=f'{int(uq_value)}% Bound',
            #         line=dict(dash='dash', color=colors[5])))

            figu.add_trace(
                go.Scatter(x=c_line['Sigma3'], y=c_line['Sigma1'],
                    mode='lines', name=f'Linear Regression',
                    line=dict(color=colors[-1])))

            x1 = [x for x in c_line['Sigma3']]
            y_upper = [y for y in c_line['High_Std_Sigma1']]
            y_lower = [y for y in c_line['Low_Std_Sigma1']]
            figu.add_trace(
                go.Scatter(
                    x=x1+x1[::-1],
                    y=y_upper+y_lower[::-1],
                    fill='toself', fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", name='Linear: ±1 Std. Dev'))

            figu.add_trace(
                go.Scatter(x=c_curv['Sigma3'], y=c_curv['Sigma1'],
                    mode='lines', name=f'Auto Fit',
                    line=dict(color='black')))

            figu.add_trace(
                go.Scatter(x=c_vert['Sigma3'], y=c_vert['Sigma1'],
                    mode='lines', name='Calc ALL Vertical', showlegend = False,
                    line=dict(color='black')))

            # Table - Auto Fit
            ucs_summary = pd.DataFrame(columns=["Method", "Sigci", "mi", "Tensile Cutoff"])
            to_append = [f"Auto Fit: {calc_method} - {calc_data}", auto_sigci, auto_mi, -c_tens]
            new_row = len(ucs_summary)
            ucs_summary.loc[new_row] = to_append

            # Table - Linear Regression
            linear_summary = pd.DataFrame(columns=["Method", "Cohesion", "Phi"])
            to_append = [f"Linear Regression: Base Case", bs_a, bs_b]
            new_row = len(linear_summary)
            linear_summary.loc[new_row] = to_append
            to_append = [f"Linear Regression: -{int(lq_value)} Std Dev", lq_a, lq_b]
            new_row = len(linear_summary)
            linear_summary.loc[new_row] = to_append
            to_append = [f"Linear Regression: +{int(uq_value)} Std Dev", uq_a, uq_b]
            new_row = len(linear_summary)
            linear_summary.loc[new_row] = to_append

            # Manual Fit Curve
            col1, col2, col3 = st.columns(3)
            with col1:
                manual_on = st.radio("Manual Fit?",
                    ('Off', 'On'), horizontal=True)
            if manual_on == 'On':
                # col1, col2 = st.columns(2)
                with col2:
                    sigci = st.number_input('Manual sigci', value=mean_ucs)
                with col3:
                    mi = st.number_input('Manual mi', value=auto_mi)

                m_tens, m_curv, m_vert, r_sq_m = get_curve_df(sigci, mi, sig3, sig1)

                # Manual Curve - Figure
                figu.add_trace(
                    go.Scatter(x=m_curv['Sigma3'], y=m_curv['Sigma1'],
                        mode='lines', name='Manual Fit Curve',
                        line=dict(dash='dash', color=base_colors['line'])))

                figu.add_trace(
                    go.Scatter(x=m_vert['Sigma3'], y=m_vert['Sigma1'],
                        mode='lines', name='Manual Vertical', showlegend = False,
                        line=dict(dash='dash', color=base_colors['line'])))

                to_append = ["Manual", sigci, mi, -m_tens]
                new_row = len(ucs_summary)
                ucs_summary.loc[new_row] = to_append

            # Boxplot
            ducs = du[du['Sigma3']==0]
            ducs = ducs[['Sigma3', 'PeakSigma1']]

            figu.add_trace(
                go.Box(x=ducs['Sigma3'], y=ducs['PeakSigma1'],
                    width=int((max(du['Sigma3'])-min(du['Sigma3']))/20),
                    name='UCS box plot', quartilemethod="linear",
                    marker_color = 'indianred'))

            num_ucs = len(du.index)
            figu.update_layout(
                    title_text=f"No. of Data: {num_ucs}",
                    plot_bgcolor='#FFFFFF',
                    paper_bgcolor='#FFFFFF',
                    height=600,)

            figu.update_xaxes(title_text='Sigma 3', gridcolor='lightgrey',
                zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
                tickformat=",.0f", ticks="outside", ticklen=5)
            figu.update_yaxes(title_text='Peak Sigma 1', gridcolor='lightgrey',
                zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
                tickformat=",.0f", ticks="outside", ticklen=5,
                range=[0,max(du['PeakSigma1'])*1.05])
            figu.add_shape(
                type="rect", xref="paper", yref="paper",
                x0=0, y0=0, x1=1.0, y1=1.0,
                line=dict(color="black", width=2))

            # Bokeh graph
            uniq = du[colormethod_u].unique()
            sel_colors = colors[:len(uniq)]
            dict_col = {typ:cl for typ,cl in zip(uniq, sel_colors)}
            du['color'] = du[colormethod_u].map(dict_col)

            color_map = bmo.CategoricalColorMapper(factors=du[colormethod_u].unique(), palette=colors)
            source = ColumnDataSource(du)
            hover = HoverTool(tooltips=[
                ('HoleID', '@HoleID'),
                ("Sigma3", "@Sigma3"),
                ("PeakSigma1", "@PeakSigma1"),
                ])
            p=figure(tools=[hover], x_axis_label='Sigma 3', y_axis_label='Peak Sigma 1')

            if len(ducs) > 0:
                q1 = ducs['PeakSigma1'].quantile(q=0.25)
                q2 = ducs['PeakSigma1'].quantile(q=0.5)
                q3 = ducs['PeakSigma1'].quantile(q=0.75)
                width = int((max(du['Sigma3'])-min(du['Sigma3']))/20)

                p.vbar([0], width, q2, q3, fill_color='indianred', line_color="black", legend_label="boxplot",)
                p.vbar([0], width, q1, q2, fill_color='indianred', line_color="black")

            p.scatter(x='Sigma3', y='PeakSigma1', size=9,
                color={'field': colormethod_u, 'transform': color_map},
                legend_group=colormethod_u, source=source)

            source = ColumnDataSource(c_line)
            p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='lightgrey',
                legend_label='Linear Regression', source=source)

            source = ColumnDataSource(c_curv)
            p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='black',
                legend_label='Auto Fit', source=source)

            source = ColumnDataSource(c_vert)
            p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='black', source=source)

            if manual_on == 'On':
                source = ColumnDataSource(m_curv)
                p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='green',
                    legend_label='Manual Fit', source=source)

                source = ColumnDataSource(m_vert)
                p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='green', source=source)

            source = ColumnDataSource(c_line)
            band = Band(base='Sigma3', lower='Low_Std_Sigma1', upper='High_Std_Sigma1', source=source,
                level='underlay', fill_alpha=0.5, line_width=1, line_color='black')
            p.add_layout(band)

            # p.circle(xx, yy, fill_color="blue", size=9, legend_label="data",)
            p.legend.location = "top_left"

        else:
            figu = go.Figure().add_annotation(
                x=2, y=2,text="No Data to Display",
                font=dict(family="sans serif",size=25,color="crimson"),
                showarrow=False,yshift=10)
            ucs_summary = pd.DataFrame()


        if fig_method==fig_selection[0]:
            st.plotly_chart(figu, use_container_width=True)
        else:
            st.bokeh_chart(p, use_container_width=True)

        # st.subheader("Summary")
        st.markdown("**Sigci and mi**")
        # st.table(ucs_summary)
        height = 31 + 29*len(ucs_summary.index)
        AgGrid(ucs_summary, fit_columns_on_grid_load=True, height=height)

        st.markdown("**Cohesion and Friction Angle**")
        # st.table(linear_summary)
        height = 31 + 29*len(linear_summary.index)
        AgGrid(linear_summary, fit_columns_on_grid_load=True, height=height)

        st.subheader("Dataset")
        # st.table(du[['HoleID', 'Rock Type', 'Sigma3', 'PeakSigma1', 'Failure Mode']])

        du = du.iloc[:,1:]
        du = du.dropna(axis=1, how='all')
        AgGrid(du)


else:
    st.header('Select a File to start')

