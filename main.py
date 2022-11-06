import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
from scipy.optimize import curve_fit
# import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from numpy import sin, cos, tan, arcsin, arccos, arctan, radians, degrees, sqrt, diag, log10, log, exp

from streamlit_option_menu import option_menu

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, Band, CustomJS
import bokeh.models as bmo
from streamlit_bokeh_events import streamlit_bokeh_events

import altair as alt

from st_aggrid import AgGrid

from filter_func import filter_df
from dst_func import weakfunc, powercurve, bartonbandis, fit_model, quantile_models, scipy_models
from ucs_func import objective, calc_hoek, get_curve_df, lin_reg, get_linear_df, quantile_models_ucs
from graph_func import scatter_altair, scatter_altair_u, scatter_plotly

# ------------------ INPUT --------------------------------
base_colors = {
    'background': '#616161',
    'grid': '#7E7E7E',
    'line': '#808080', #'#FFFAF1',
    'text': '#F9F9F3'
}

colors = px.colors.qualitative.T10

st.set_page_config(page_title='Geotech Lab', page_icon=None, layout="wide")

# hide_table_row_index = """
#     <style>
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     </style>
#     """

# st.markdown(hide_table_row_index, unsafe_allow_html=True)


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# ----------------- Key columns -------------------------------
# DST
key_id = ['holeid']
key_project = ['project']
key_from = ['sampfrom']
key_to = ['sampto']
key_form = ['strat']
key_subform = ['strand']
key_rocktype = ['lithology']
key_str = ['rk_str_class_samp']
key_weath = ['rk_weath_class']
key_testtype = ['gs_test_type']
key_bdensity = ['dry_bulk_density']
key_teststage = ['test_stage']
key_normalstress = ['gs_cor_normal_strs']
key_shearstress = ['gs_cor_pk_shr_strs']
key_effsigma3 = ['eff_sigma_3']
key_effsigma1 = ['eff_sigma_1']
key_failangle = ['fail_angle']
key_failnature = ['fail_nature']
key_sigma3 = ['gs_sigma_3']
key_pkstress = ['gs_pk_strs']
key_ucs = ['gs_ucs']
key_lab = ['laboratory']
key_sampletype = ['sampletype']
key_date = ['date_tested']

# -------------------- Sidebar -------------------------------------------
uploaded_files = st.sidebar.file_uploader("Choose a file", accept_multiple_files=True)
df = pd.DataFrame()
dfn = pd.DataFrame()
all_cols = []
if not uploaded_files:
    df = None
else:
    for uploaded_file in uploaded_files:
        # Can be used wherever a "file-like" object is accepted:
        df1 = pd.read_csv(uploaded_file)
        for col in df1.columns:
            if any(substring in col.lower() for substring in key_id):
                colid = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_project):
                colpj = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_from):
                colfrom = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_to):
                colto = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_form):
                colform = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_subform):
                colsubform = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_rocktype):
                colrocktype = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_str):
                colstr = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_weath):
                colweath = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring == col.lower() for substring in key_testtype):
                coltesttype = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_bdensity):
                colbdensity = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_teststage):
                colteststage = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_normalstress):
                colnormstr = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_shearstress):
                colshearstr = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_effsigma3):
                coleffsig3 = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_effsigma1):
                coleffsig1 = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_failnature):
                colfailnat = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_failangle):
                colfailang = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_sigma3):
                colsig3 = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_pkstress):
                colpkstr = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_ucs):
                colucs = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_lab):
                collab = col
                all_cols.append(col)
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_date):
                coltestdate = col
                dfn[col] = df1[col]
            if any(substring in col.lower() for substring in key_sampletype):
                colsamptype = col
                dfn[col] = df1[col]

        dfn = dfn[dfn[colid].notna()] # ID should not be empty
        all_cols = list(set(all_cols))
        dictsamptype = {"gsdirshear":'DST',
            "gstriaxial":'TXL',
            "gstriaxialhoek":'HTX',
            "gsucs":'UCS'}
        dfn[colsamptype] = dfn[colsamptype].str.lower()
        dfn = dfn.replace({colsamptype: dictsamptype})
        df = pd.concat([df,dfn], axis=0, ignore_index=True)


if df is not None:
    # -------------------- Tabs -----------------------------------------------
    options = ['Defect Shear Strength', 'Rockmass Shear Strength', 'Soil Shear Strength']
    selected = option_menu(
        menu_title=None, options=options,
        default_index=0, orientation='horizontal')
    # ------------ DST -------------------
    # with tab1:
    if selected ==options[0]:
        # Selections
        df1 = df[df[colsamptype]=='DST']
        if len(df1)>0:
            df1[colteststage] = pd.to_numeric(df1[colteststage], downcast='integer', errors='coerce')

            # Additional Filter
            color_list = [coltesttype, colrocktype, colid, colteststage]
            p_max_o = int(df1[colnormstr].max())
            p_max = st.sidebar.number_input('Maximum Normal Stress', value=p_max_o)
            df1 = df1[df1[colnormstr]<=p_max]

            # Filter the dataset
            df1, color_list = filter_df(df1, color_list, all_cols, 'DST')
            x = df1[colnormstr]
            y = df1[colshearstr]
            col1, col2, col3 = st.columns(3)

            fit_selection = ('Linear', 'Power')
            with col1:
                fitmethod = st.radio("Auto Fit Method", fit_selection, horizontal=True)
            with col2:
                lq_value = st.number_input('Lower Quantile (%)', value=25, step=5)
            with col3:
                uq_value = st.number_input('Upper Quantile (%)', value=75, step=5)

            col1, col2, col3 = st.columns(3)

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

            # Main DST
            if fitmethod == fit_selection[0]:
                data = df1[[colnormstr, colshearstr]]
                a_intercept = intercept_coh

                dbs, bs_a, bs_b, dlq, lq_a, lq_b, duq, uq_a, uq_b, lq, uq = quantile_models(data, colnormstr, colshearstr, a_intercept, lq_value, uq_value)

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
                    manual_on_p = st.radio("Manual Fit?", ('off', 'on'), horizontal=True)
                if manual_on_p == 'on':
                    with col2:
                        d_coh = st.number_input('Manual cohesion', value=bs_a)
                    with col3:
                        d_fric = st.number_input('Manual friction angle', value=bs_b)

                    sigN = list(range(int(min(x)), int(max(x)), 1))
                    dman = pd.DataFrame({'x':sigN})
                    dman['y'] = dman.apply(lambda row: d_coh+row["x"]*tan(radians(d_fric)), axis=1)

                    manual_val_1 = d_coh
                    manual_val_2 = d_fric
                else:
                    manual_val_1 = np.nan

            # Power Fit
            else:
                df1["Log_NormalStress"] = log(df1[colnormstr])
                df1["Log_ShearStress"] = log(df1[colshearstr])
                data = df1[["Log_NormalStress", "Log_ShearStress"]]
                data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

                a_intercept = log(intercept_k)

                dbs, bs_a, bs_b, dlq, lq_a, lq_b, duq, uq_a, uq_b, lq, uq = quantile_models(data, "Log_NormalStress", "Log_ShearStress", a_intercept, lq_value, uq_value)

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
                    manual_on_p = st.radio("Manual Fit?", ('off', 'on'), key='man_p', horizontal=True)
                if manual_on_p == 'on':
                    with col2:
                        k_pow = st.number_input('Manual k', value=bs_a)
                    with col3:
                        m_pow = st.number_input('Manual m', value=bs_b)

                    sigN = list(range(int(min(x)), int(max(x)), 1))
                    dman = pd.DataFrame({'x':sigN})
                    dman['y'] = dman.apply(lambda row: k_pow*(row["x"]**m_pow), axis=1)

                    manual_val_1 = k_pow
                    manual_val_2 = m_pow
                else:
                    manual_val_1 = np.nan

            # Color list
            colormethod_d = st.radio("Color By", color_list, horizontal=True)

            # Graph
            title = 'DST'
            rotation = 360
            dlq['line'] = f'{lq_value} % bound'
            dbs['line'] = f'OLS Fit Line'
            duq['line'] = f'{uq_value} % bound'
            if not np.isnan(manual_val_1):
                dman['line'] = f'Manual Fit Line'
            else:
                dman = pd.DataFrame()
            figalt = scatter_altair(title, rotation, colnormstr, colshearstr, colormethod_d, df1, duq, dlq, dbs, dman, manual_on_p)

            # Summary Table
            dst_summary = pd.DataFrame(columns=["Method", val_1, val_2])
            to_append = ["Auto Fit - OLS", bs_a, bs_b]
            new_row = len(dst_summary)
            dst_summary.loc[new_row] = to_append
            to_append = [f"{int(lq*100)} th Quantile", lq_a, lq_b]
            new_row = len(dst_summary)
            dst_summary.loc[new_row] = to_append
            to_append = [f"{int(uq*100)} th Quantile", uq_a, uq_b]
            new_row = len(dst_summary)
            dst_summary.loc[new_row] = to_append
            if not np.isnan(manual_val_1):
                to_append = ['Manual Fit', manual_val_1, manual_val_2]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
        else:
            figalt = alt.layer()
            dst_summary = pd.DataFrame()

        st.altair_chart(figalt, use_container_width=True)

        st.markdown("**Shear Strength**")
        height = 32 + 28*len(dst_summary.index)
        AgGrid(dst_summary, fit_columns_on_grid_load=True, height=height)

        st.subheader("Dataset")
        df1 = df1.iloc[:,1:]
        df1 = df1.dropna(axis=1, how='all')
        AgGrid(df1)

    # ------------ UCS and Rock TXL -------------------
    if selected == options[1]:

        method_selection = ('Scipy Curve Fit', 'Hoek Equation')
        calc_selection = ('All data', 'HTX + UCS(mean)', 'HTX only')
        color_list = [colsamptype, colrocktype, colid, colfailnat]

        du = df[df[colsamptype].isin(['UCS', 'HTX', 'BZT'])]
        du[colfailang] = pd.to_numeric(du[colfailang], errors='coerce')
        du.loc[du[colsamptype]=='UCS', colsig3] = 0
        du.loc[du[colsamptype]=='UCS', colpkstr] = du[colucs]
        du = du[du[colsig3].notna()]
        du = du[du[colpkstr].notna()]

        if len(du[colsig3])>0:
            p_max_o = int(du[colsig3].max())
            p_max = st.sidebar.number_input('Maximum Sigma3', value=p_max_o)
            du = du[du[colsig3]<=p_max]

            # Filter the dataset
            du, color_list = filter_df(du, color_list, all_cols, "HTX")

        # Auto Fit - Method and dataset
        col1, col2, col3, col4 = st.columns([2,2,1,1])
        with col1:
            calc_method = st.radio("Auto Fit method", method_selection, horizontal=True)
        with col2:
            calc_data = st.radio("Select dataset", calc_selection, horizontal=True)
        with col3:
            lq_value = st.number_input('Lower Quantile', value=25, step=5)
        with col4:
            uq_value = st.number_input('Upper Quantile', value=75, step=5)

        if len(du[colsig3]) > 0:
            # Calculate mean UCS and BZT test results
            if not du[du[colsig3]==0].empty:
                mean_ucs = int(du[du[colsig3]==0][colpkstr].mean())
            else:
                mean_ucs = 0
            if not du[du[colpkstr]==0].empty:
                mean_bzt = int(du[du[colpkstr]==0][colsig3].mean())
            else:
                mean_bzt = 0

            # Append Mean UCS
            db = du[du[colpkstr]!=0]
            dt = db[db[colsig3]!=0]
            du_auto = dt[[colsig3, colpkstr]]
            to_append = [0, mean_ucs]
            ducs_mean = pd.DataFrame(to_append).transpose()
            ducs_mean.columns=[colsig3, colpkstr]
            new_row = len(du_auto)
            du_auto.loc[new_row] = to_append

            # Append Mean Brazilian
            to_append = [mean_bzt, 0]
            dbzt_mean = pd.DataFrame(to_append).transpose()
            dbzt_mean.columns=[colsig3, colpkstr]

            # x and y dataset for auto fit
            if calc_data==calc_selection[0]:
                sig3 = db[colsig3]
                sig1 = db[colpkstr]
                data = db[[colsig3, colpkstr]]
            elif calc_data==calc_selection[1]:
                sig3 = du_auto[colsig3]
                sig1 = du_auto[colpkstr]
                data = du_auto[[colsig3, colpkstr]]
            else:
                sig3 = dt[colsig3]
                sig1 = dt[colpkstr]
                data = dt[[colsig3, colpkstr]]

            # Auto Fit Main part
            if calc_method == method_selection[1]:
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

            else:
                params, params_covariance = curve_fit(objective, sig3, sig1)
                auto_sigci, auto_mi = params # a = sigci, b = mi
                auto_sigci = round(auto_sigci,2)
                auto_mi = round(auto_mi,2)

                c_tens, c_curv, c_vert, r_sq_c = get_curve_df(auto_sigci, auto_mi, sig3, sig1)

            a_intercept = np.nan

            dbs, bs_a, bs_b, dlq, lq_a, lq_b, duq, uq_a, uq_b = quantile_models_ucs(data, a_intercept, lq_value, uq_value)

            bs_b = arcsin((bs_b-1)/(bs_b+1))
            bs_a = bs_a * (1-sin(bs_b))/(2*cos(bs_b))
            bs_b = round(degrees(bs_b), 2)
            bs_a = round(bs_a, 2)
            lq_b = arcsin((lq_b-1)/(lq_b+1))
            lq_a = lq_a * (1-sin(lq_b))/(2*cos(lq_b))
            lq_b = round(degrees(lq_b), 2)
            lq_a = round(lq_a, 2)
            uq_b = arcsin((uq_b-1)/(uq_b+1))
            uq_a = uq_a * (1-sin(uq_b))/(2*cos(uq_b))
            uq_b = round(degrees(uq_b), 2)
            uq_a = round(uq_a, 2)

            col1, col2, col3 = st.columns(3)
            with col1:
                manual_on = st.radio("Manual Fit?", ('off', 'on'), key='man_u', horizontal=True)
            if manual_on == 'on':
                with col2:
                    sigci = st.number_input('Manual sigci', value=mean_ucs)
                with col3:
                    mi = st.number_input('Manual mi', value=auto_mi)

            # Figure
            colormethod_u = st.radio("Color By", color_list, horizontal=True)

            # Altair
            title = 'UCS'
            rotation = 360
            dlq['line'] = f'{lq_value} % bound'
            dbs['line'] = f'OLS Fit Line'
            duq['line'] = f'{uq_value} % bound'
            c_curv['line'] = f'Auto Fit Curve'
            ducs = du[du[colsig3]==0]
            ducs = ducs[[colsig3, colpkstr]]
            if manual_on == 'on':
                m_tens, m_curv, m_vert, r_sq_m = get_curve_df(sigci, mi, sig3, sig1)
                m_curv['line'] = f'Manual Fit Curve'
            else:
                m_curv = pd.DataFrame()

            figaltu = scatter_altair_u(title, rotation, colsig3, colpkstr, colormethod_u, du, duq, dlq, dbs, ducs, c_curv, m_curv, manual_on)

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
            to_append = [f"Quantile Regression: {int(lq_value)} %", lq_a, lq_b]
            new_row = len(linear_summary)
            linear_summary.loc[new_row] = to_append
            to_append = [f"Quantile Regression: {int(uq_value)} %", uq_a, uq_b]
            new_row = len(linear_summary)
            linear_summary.loc[new_row] = to_append

            # Manual Fit Curve
            if manual_on == 'on':
                to_append = ["Manual", sigci, mi, -m_tens]
                new_row = len(ucs_summary)
                ucs_summary.loc[new_row] = to_append

        else:
            ucs_summary = pd.DataFrame()

        st.altair_chart(figaltu, use_container_width=True)

        st.markdown("**Sigci and mi**")
        height = 31 + 29*len(ucs_summary.index)
        AgGrid(ucs_summary, fit_columns_on_grid_load=True, height=height)

        st.markdown("**Cohesion and Friction Angle**")
        height = 31 + 29*len(linear_summary.index)
        AgGrid(linear_summary, fit_columns_on_grid_load=True, height=height)

        st.subheader("Dataset")
        du = du.iloc[:,0:]
        du = du.dropna(axis=1, how='all')
        AgGrid(du)

    # ---------------- Tab3: Soil Strength ---------------------------
    if selected == options[2]:

        @st.experimental_singleton
        def load_data(df):
            return df[df[colsamptype].isin(['TXL'])]

        color_list = [colsamptype, colrocktype, colid]

        # Initialise session state
        if "txl_query" not in st.session_state:
            st.session_state["txl_query"] = set()

        # ds = df[df[colsamptype].isin(['TXL'])]
        ds = load_data(df)

        if len(ds)>0:
            ds = ds[ds[coleffsig3].notna()]
            ds = ds[ds[coleffsig1].notna()]
            p_max_o = ds[coleffsig3].max()
            p_max = st.sidebar.number_input('Maximum Sigma3', value=p_max_o)
            ds = ds[ds[coleffsig3]<=p_max]

            # Filter the dataset
            ds, color_list = filter_df(ds, color_list, all_cols, "TXL")

            # Auto Fit - Method and dataset
            col1, col2, col3 = st.columns(3)
            method_selection = ['Linear Regression']
            with col1:
                fit_method = st.radio("Auto Fit method", method_selection, horizontal=True)
            with col2:
                lq_value = st.number_input('Lower Quantile (%)', key='soil_lq', value=25, step=5)
            with col3:
                uq_value = st.number_input('Upper Quantile (%)', key='soil_uq', value=75, step=5)

            # Set intercept
            negative_int_selection = ("On", "Set Intercept")
            with col1:
                negative_int_method = st.radio("Automatic Intercept",
                    negative_int_selection, horizontal=True)
            if negative_int_method == negative_int_selection[1]:
                with col2:
                    intercept_int = st.number_input('Set Intercept', value=0, step=5)
            else:
                intercept_int = np.nan

            # PQ plot
            ds['P'] = ds.apply(lambda row: (row[coleffsig1]+row[coleffsig3])/2, axis=1)
            ds['Q'] = ds.apply(lambda row: (row[coleffsig1]-row[coleffsig3])/2, axis=1)
            ds['esig3-esig1'] = (ds[coleffsig3].astype(int).astype(str)+"-"+ds[coleffsig1].astype(int).astype(str))
            ds['selected'] = True
            # print(st.session_state['txl_query'])
            if st.session_state["txl_query"]:
                ds.loc[~ds["esig3-esig1"].isin(st.session_state["txl_query"]), "selected"] = False
            # print(ds[["P","Q", "P-Q", "selected"]].head())
            data = ds[ds['selected']==True][["P", "Q"]]
            a_intercept = intercept_int

            dbs, bs_a, bs_b, dlq, lq_a, lq_b, duq, uq_a, uq_b, lq, uq = quantile_models(data, "P", "Q", a_intercept, lq_value, uq_value)

            bs_a = round(bs_a/(sqrt(1-bs_b**2)),2)
            bs_b = round(degrees(arcsin(bs_b)),2)
            lq_a = round(lq_a/(sqrt(1-lq_b**2)),2)
            lq_b = round(degrees(arcsin(lq_b)),2)
            uq_a = round(uq_a/(sqrt(1-uq_b**2)),2)
            uq_b = round(degrees(arcsin(uq_b)),2)

            val_1 = 'Cohesion'
            val_2 = 'Friction Angle'

            # Manual Fit
            col1, col2, col3 = st.columns(3)
            with col1:
                manual_on_s = st.radio("Manual Fit?", ('off', 'on'), key='manual_soil', horizontal=True)
            if manual_on_s == 'on':
                with col2:
                    m_coh = st.number_input('Manual cohesion', value=bs_a)
                with col3:
                    m_fric = st.number_input('Manual friction angle', value=bs_b)

                x = ds["P"]
                y = ds["Q"]
                myP = list(range(int(min(x)), int(max(x)), 1))
                dman = pd.DataFrame({'x':myP})
                m_b = sin(radians(m_fric))
                m_a = m_coh * (sqrt(1-m_b**2))
                dman['y'] = dman.apply(lambda row: m_a+row["x"]*m_b, axis=1)

                manual_val_1 = m_coh
                manual_val_2 = m_fric
            else:
                manual_val_1 = np.nan

            colormethod_s = st.radio("Color By", color_list, horizontal=True)
            # Graph
            title = 'TXL'
            rotation = 360
            dlq['line'] = f'{lq_value} % bound'
            dbs['line'] = f'OLS Fit Line'
            duq['line'] = f'{uq_value} % bound'
            if not np.isnan(manual_val_1):
                dman['line'] = f'Manual Fit Line'
            else:
                dman = pd.DataFrame()
            figalts = scatter_altair(title, rotation, 'P', 'Q', colormethod_s, ds, duq, dlq, dbs, dman, manual_on_s)

            # Plotly
            col1, col2 = st.columns([1,1])
            with col1:
                fige = scatter_plotly(title, rotation, coleffsig3, coleffsig1, 'selected', ds, None, None, None, None, None)
                selected_points = plotly_events(fige, select_event=True,  override_height=600)
            with col2:
                figpl = scatter_plotly(title, rotation, 'P', 'Q', 'selected', ds, duq, dlq, dbs, dman, manual_on_s)
                st.plotly_chart(figpl, use_container_width=True)
                # selected_points = plotly_events(figpl, select_event=True,  override_height=600)
            dsel = pd.DataFrame(selected_points)
            current_query = {}
            current_query["txl_query"] = {f"{int(el['x'])}-{int(el['y'])}" for el in selected_points}
            # Update session state
            rerun = False
            if current_query["txl_query"] - st.session_state["txl_query"]:
                st.session_state['txl_query'] = current_query["txl_query"]
                rerun = True
            # print(current_query)
            if rerun:
                st.experimental_rerun()
            # st.write(dsel)

            # Summary Table
            txl_summary = pd.DataFrame(columns=["Method", val_1, val_2])
            to_append = ["Auto Fit - OLS", bs_a, bs_b]
            new_row = len(txl_summary)
            txl_summary.loc[new_row] = to_append
            to_append = [f"{int(lq*100)} th Quantile", lq_a, lq_b]
            new_row = len(txl_summary)
            txl_summary.loc[new_row] = to_append
            to_append = [f"{int(uq*100)} th Quantile", uq_a, uq_b]
            new_row = len(txl_summary)
            txl_summary.loc[new_row] = to_append
            if not np.isnan(manual_val_1):
                to_append = ['Manual Fit', manual_val_1, manual_val_2]
                new_row = len(txl_summary)
                txl_summary.loc[new_row] = to_append
        else:
            figalts = alt.layer()
            txl_summary = pd.DataFrame()

        # st.altair_chart(figalts, use_container_width=True)

        st.markdown("**Shear Strength**")

        height = 32 + 28*len(txl_summary.index)
        AgGrid(txl_summary, fit_columns_on_grid_load=True, height=height)

        st.subheader("Dataset")
        # Table of data
        ds = ds.iloc[:,0:]
        ds = ds.dropna(axis=1, how='all')
        AgGrid(ds)
else:
    st.header('Select a File to start')

