import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import curve_fit

from numpy import sin, cos, tan, arcsin, arccos, arctan, radians, degrees, sqrt


# ------------------ INPUT --------------------------------
base_colors = {
    'background': '#616161',
    'grid': '#7E7E7E',
    'line': '#808080', #'#FFFAF1',
    'text': '#F9F9F3'
}

# ------------------ FUNCTION ----------------------------------
# Direct Shear
def weakfunc(x, a, b):
    # shear stress = cohesion + normal stress * tan(friction_angle)
    # x = Normal Stress, a = cohesion, b = friction angle
    return a + x * np.tan(b)

def powercurve(x, k, m):
    return k * (x ** m)

# Rock UCS and TXL
def objective(x, a, b):
    # row['Sigma_3'] + sigci*math.sqrt((mi * row['Sigma_3'] / sigci) + 1)
    # x = Sigma3, a = sigci, b = mi
    return x + a*np.sqrt((b * x / a) + 1)

# -------------------- Sidebar -------------------------------------------
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_file)

    rock_type = set(df['Lithology'])
    holeid = set(df['HoleID'])
    teststage = (x for x in set(df['TestStage']) if np.isnan(x) == False)

    # Selections
    rock_selection = st.sidebar.multiselect(
        "Rock Type",
        (rock_type)
    )

    holeid_selection = st.sidebar.multiselect(
        "Hole ID",
        (holeid)
    )

    ex_or_in = st.sidebar.radio("Exclude or Include",
            ('Exclude', 'Include'), horizontal=True),

    teststage_selection = st.sidebar.multiselect(
        "Test Stage",
        (teststage)
    )

    # st.write(ex_or_in[0])
    # Filter
    if rock_selection: df = df[df['Lithology'].isin(rock_selection)]
    if holeid_selection:
        if ex_or_in[0] == 'Exclude':
            df = df[~df['HoleID'].isin(holeid_selection)]
        else:
            df = df[df['HoleID'].isin(holeid_selection)]
    if teststage_selection: df = df[df['TestStage'].isin(teststage_selection)]

else:
    df = None

if df is not None:
    # -------------------- Tabs -----------------------------------------------
    tab1, tab2 = st.tabs(["Defect Shear Strength", "Rockmass Shear Strength"])
    colors = px.colors.qualitative.T10

    # ------------ DST -------------------
    with tab1:
        st.header("DST")

        df1 = df[df['TestType']=='SSDS']

        figd = px.scatter(
            df1, x="NormalStress", y="ShearStress",
            color="HoleID", color_discrete_sequence=colors)

        x = df1["NormalStress"]
        y = df1["ShearStress"]

        fitmethod = st.radio("Fit Method",
            ('Linear', 'Power'), horizontal=True)

        if len(x) > 0:
            if fitmethod == 'Linear':

                # Auto Fit
                params, params_covariance = curve_fit(weakfunc, x, y)
                auto_c, auto_f = params # automatic cohesion and friction angle

                residuals = y - weakfunc(x,*params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r_sq_dst = 1 - (ss_res/ss_tot)

                x_line = np.arange(0, max(x), 1)
                fit_curve = pd.DataFrame({'x_line':x_line})
                fit_curve['y_line'] = fit_curve.apply(lambda x_l: weakfunc(x_l,auto_c,auto_f))

                auto_c = round(auto_c,2)
                auto_f = round(degrees(auto_f),2)
                # print(auto_c, auto_f)
                r_sq_dst = round(r_sq_dst,4)
                val_1 = 'Cohesion'
                val_2 = 'Friction Angle'

                # Manual Fit
                col1, col2 = st.columns(2)
                with col1:
                    d_coh = st.number_input('Manual cohesion', value=auto_c)
                with col2:
                    d_fric = st.number_input('Manual friction angle', value=25)
                sigN = list(range(0, int(max(x)), 1))
                dman = pd.DataFrame({'sigN':sigN})
                dman['sigT'] = dman.apply(lambda row: d_coh+row["sigN"]*tan(radians(d_fric)), axis=1)
                params_man = [d_coh, radians(d_fric)]
                residuals = y - weakfunc(x,*params_man)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r_sq_dst_m = 1 - (ss_res/ss_tot)
                manual_val_1 = d_coh
                manual_val_2 = d_fric
                manual_val_3 = r_sq_dst_m

            elif fitmethod == 'Power':

                # Auto Fit
                params, params_covariance = curve_fit(powercurve, x, y)
                auto_c, auto_f = params # automatic cohesion and friction angle

                residuals = y - powercurve(x,*params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r_sq_dst = 1 - (ss_res/ss_tot)

                x_line = np.arange(0, max(x), 1)
                fit_curve = pd.DataFrame({'x_line':x_line})
                fit_curve['y_line'] = fit_curve.apply(lambda x_l:powercurve(x_l,auto_c,auto_f))

                auto_c = round(auto_c,4)
                auto_f = round(auto_f,4)
                r_sq_dst = round(r_sq_dst,4)
                val_1 = 'k'
                val_2 = 'm'

                # Manual Fit
                col1, col2 = st.columns(2)
                with col1:
                    k_pow = st.number_input('Manual k', value=auto_c)
                with col2:
                    m_pow = st.number_input('Manual m', value=0.998)
                sigN = list(range(0, int(max(x)), 1))
                dman = pd.DataFrame({'sigN':sigN})
                dman['sigT'] = dman.apply(lambda row: k_pow*(row["sigN"]**m_pow), axis=1)
                params_man = [k_pow, m_pow]
                residuals = y - powercurve(x,*params_man)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r_sq_dst_m = 1 - (ss_res/ss_tot)
                manual_val_1 = k_pow
                manual_val_2 = m_pow
                manual_val_3 = r_sq_dst_m

            figd.update_traces(marker=dict(size=9))

            figd.add_trace(go.Scatter(x=dman["sigN"], y=dman["sigT"],
                mode='lines', name='Manual Fit',
                line=dict(dash='dash', color=base_colors['line'])))

            figd.add_trace(go.Scatter(x=fit_curve['x_line'], y=fit_curve['y_line'],
                mode='lines', name=f'Curve Fit - {fitmethod}',
                line=dict(color=colors[6])))

            num_dst = len(df1.index)
            figd.update_layout(
                title_text=f"No. of Data: {num_dst}",
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',)
            figd.update_xaxes(title_text='Normal Stress', gridcolor='lightgrey',
                zeroline=True, zerolinewidth=2, zerolinecolor='black')
            figd.update_yaxes(title_text='Shear Stress', gridcolor='lightgrey',
                zeroline=True, zerolinewidth=2, zerolinecolor='black')

            # Table
            dst_summary = pd.DataFrame(columns=["Method", val_1, val_2, "R squared"])
            to_append = [fitmethod, auto_c, auto_f, r_sq_dst]
            new_row = len(dst_summary)
            dst_summary.loc[new_row] = to_append
            to_append = ['Manual Fit', manual_val_1, manual_val_2, manual_val_3]
            new_row = len(dst_summary)
            dst_summary.loc[new_row] = to_append

        st.plotly_chart(figd, use_container_width=True)

        st.table(dst_summary)

        st.header("Dataset")
        st.table(df1[['HoleID', 'Lithology', 'TestStage', 'NormalStress', 'ShearStress', 'Peak or Residual', 'Shear Plane Type']])

    # ------------ UCS and Rock TXL -------------------
    with tab2:
        st.header("UCS and Rock TXL")
        du = df[df['TestType'].isin(['Uniax', 'Triax', 'Brazilian'])]
        du = du[du['Sigma3'].notna()]
        du = du[du['PeakSigma1'].notna()]

        # Auto Fit Curve
        if not du[du['Sigma3']==0].empty:
            mean_ucs = int(du[du['Sigma3']==0]['PeakSigma1'].mean())
        else:
            mean_ucs = 0
        if not du[du['PeakSigma1']==0].empty:
            mean_bzt = int(du[du['PeakSigma1']==0]['Sigma3'].mean())
        else:
            mean_bzt = 0

        du_auto = du[du['Sigma3']!=0]
        du_auto = du_auto[du_auto['PeakSigma1']!=0]
        du_auto = du_auto[['Sigma3', 'PeakSigma1']]

        du_auto = du[du['Sigma3']!=0]
        du_auto = du_auto[du_auto['PeakSigma1']!=0]
        du_auto = du_auto[['Sigma3', 'PeakSigma1']]

        to_append = [0, mean_ucs]
        ducs_mean = pd.DataFrame(to_append).transpose()
        ducs_mean.columns=['Sigma3', 'PeakSigma1']
        new_row = len(du_auto)
        du_auto.loc[new_row] = to_append
        # Append Mean Brazilian
        to_append = [mean_bzt, 0]
        dbzt_mean = pd.DataFrame(to_append).transpose()
        dbzt_mean.columns=['Sigma3', 'PeakSigma1']

        # Auto Sigci and Mi
        x = du_auto["Sigma3"]
        y = du_auto["PeakSigma1"]
        params, params_covariance = curve_fit(objective, x, y)
        auto_sigci, auto_mi = params # a = sigci, b = mi
        auto_sigci = round(auto_sigci,2)
        auto_mi = round(auto_mi,2)
        residuals = y - objective(x,*params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        r_sq_curve = 1 - (ss_res/ss_tot)
        r_sq_curve = round(r_sq_curve,4)
        auto_tensile = - int(auto_sigci / (8.62 + 0.7 * auto_mi))
        x_line = np.arange(auto_tensile, max(x), 1)
        fit_curve = pd.DataFrame({'x_line':x_line})
        fit_curve['y_line'] = fit_curve.apply(lambda x_l: objective(x_l,auto_sigci,auto_mi))

        # Vertical Line for Tensile Cutoff
        auto_vertical = fit_curve[fit_curve['x_line']==auto_tensile]
        to_append = [auto_tensile, 0]
        new_row = len(auto_vertical)
        auto_vertical.loc[new_row] = to_append

        # Manual Fit Curve
        col1, col2 = st.columns(2)
        with col1:
            sigci = st.number_input('Manual sigci', value=mean_ucs)
        with col2:
            mi = st.number_input('Manual mi', value=auto_mi)
        manual_tensile = - int(sigci / (8.62 + 0.7 * mi))
        params_man = [sigci, mi]
        residuals = y - objective(x,*params_man)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y-np.mean(y))**2)
        r_sq_curve_man = 1 - (ss_res/ss_tot)
        r_sq_curve_man = round(r_sq_curve_man,4)

        Sigma3 = list(range(manual_tensile, int(du['Sigma3'].max())))
        hcurv = pd.DataFrame({'Sigma3':Sigma3})
        hcurv['Sigma1'] = hcurv.apply(lambda row: row['Sigma3']+sigci*sqrt((mi*row['Sigma3']/sigci)+1), axis=1)

        manual_vertical = hcurv[hcurv['Sigma3']==manual_tensile]
        to_append = [manual_tensile, 0]
        new_row = len(manual_vertical)
        manual_vertical.loc[new_row] = to_append

        # Figure
        figu = go.Figure()
        figu.add_trace(
            go.Scatter(x=ducs_mean['Sigma3'], y=ducs_mean['PeakSigma1'],
                name='Mean UCS', mode='markers',
                marker=dict(size=15, color=colors[-3])))

        figu.add_trace(
            go.Scatter(x=dbzt_mean['Sigma3'], y=dbzt_mean['PeakSigma1'],
                name='Mean Brazilian', mode='markers',
                marker=dict(size=15, color=colors[-4])))

        figu.add_trace(
            go.Scatter(x=fit_curve['x_line'], y=fit_curve['y_line'],
                mode='lines', name='Scipy Fit Curve',
                line=dict(color=colors[0])))

        figu.add_trace(
            go.Scatter(x=auto_vertical['x_line'], y=auto_vertical['y_line'],
                mode='lines', name='Auto Vertical', showlegend = False,
                line=dict(color=colors[0])))

        figu.add_trace(
            go.Scatter(x=hcurv['Sigma3'], y=hcurv['Sigma1'],
                mode='lines', name='Manual Fit Curve',
                line=dict(dash='dash', color=base_colors['line'])))

        figu.add_trace(
            go.Scatter(x=manual_vertical['Sigma3'], y=manual_vertical['Sigma1'],
                mode='lines', name='Manual Vertical', showlegend = False,
                line=dict(dash='dash', color=base_colors['line'])))

        ducs = du[du['Sigma3']==0]
        ducs = ducs[['Sigma3', 'PeakSigma1']]

        figu.add_trace(
            go.Box(x=ducs['Sigma3'], y=ducs['PeakSigma1'], width=int(max(du['Sigma3'])/20),
                name='UCS box plot', quartilemethod="linear",
                marker_color = 'indianred'))

        testtypes = du['TestType'].unique()
        icolor = 1
        for testtype in testtypes:
            figu.add_trace(
                go.Scatter(
                    x=du[du['TestType']==testtype]['Sigma3'],
                    y=du[du['TestType']==testtype]['PeakSigma1'],
                    mode='markers', name=testtype,
                    marker=dict(size=9, color=colors[icolor])
                    # mode='markers', marker=dict(color=du['Code'])
                    )
                )
            icolor += 1

        num_ucs = len(du.index)
        figu.update_layout(
                title_text=f"No. of Data: {num_ucs}",
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',)
        figu.update_xaxes(title_text='Sigma 3', gridcolor='lightgrey',
            zeroline=True, zerolinewidth=2, zerolinecolor='black')
        figu.update_yaxes(title_text='Peak Sigma 1', gridcolor='lightgrey',
            zeroline=True, zerolinewidth=2, zerolinecolor='black')

        st.plotly_chart(figu, use_container_width=True)

        # Table
        ucs_summary = pd.DataFrame(columns=["Method", "Sigci", "Mi", "Tensile Cutoff", "R squared"])
        to_append = ["Auto Fit", auto_sigci, auto_mi, auto_tensile, r_sq_curve]
        new_row = len(ucs_summary)
        ucs_summary.loc[new_row] = to_append
        to_append = ["Manual", sigci, mi, manual_tensile, r_sq_curve_man]
        new_row = len(ucs_summary)
        ucs_summary.loc[new_row] = to_append

        st.table(ucs_summary)

        st.table(du[['HoleID', 'Lithology', 'Sigma3', 'PeakSigma1', 'Failure Mode']])

else:
    st.header('Select a File to start')

