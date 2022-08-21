import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import curve_fit

from numpy import sin, cos, tan, arcsin, arccos, arctan, radians, degrees, sqrt, diag


# ------------------ INPUT --------------------------------
base_colors = {
    'background': '#616161',
    'grid': '#7E7E7E',
    'line': '#808080', #'#FFFAF1',
    'text': '#F9F9F3'
}

st.set_page_config(layout="wide")

hide_table_row_index = """
    <style>
    thead tr th:first-child {display:none}
    tbody th {display:none}
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

    rock_type = set(df['Rock Type'])
    holeid = set(df['HoleID'])
    sampletype = set(df['SampleType'])
    teststage = (x for x in set(df['TestStage']) if np.isnan(x) == False)
    shear_type = set(df['Shear Plane Type'])

    # Selections
    rock_selection = st.sidebar.multiselect(
        "Rock Type", (rock_type))

    holeid_selection = st.sidebar.multiselect(
        "Hole ID", (holeid))

    ex_or_in = st.sidebar.radio("Exclude or Include",
        ('Exclude', 'Include'), horizontal=True),

    testtype_selection = st.sidebar.multiselect(
        "Test type", (sampletype))

    ex_or_in_testtype = st.sidebar.radio("Exclude or Include",
        ('exclude', 'include'), horizontal=True),

    # Filter
    if rock_selection: df = df[df['Rock Type'].isin(rock_selection)]
    if holeid_selection:
        if ex_or_in[0] == 'exclude':
            df = df[~df['HoleID'].isin(holeid_selection)]
        else:
            df = df[df['HoleID'].isin(holeid_selection)]

    if testtype_selection:
        if ex_or_in_testtype[0] == 'Exclude':
            df = df[~df['SampleType'].isin(testtype_selection)]
        else:
            df = df[df['SampleType'].isin(testtype_selection)]

else:
    df = None

if df is not None:
    # -------------------- Tabs -----------------------------------------------
    tab1, tab2 = st.tabs(["Defect Shear Strength", "Rockmass Shear Strength"])
    colors = px.colors.qualitative.T10

    # ------------ DST -------------------
    with tab1:
        # Selections
        st.header("DST")

        df1 = df[df['TestType']=='SSDS']

        # Additional Filter
        if len(df1)>0:
            teststage_selection = st.sidebar.multiselect(
                "Test Stage", (teststage))

            sheartype_selection = st.sidebar.multiselect(
                "Shear Plane Type", (shear_type))

            if teststage_selection: df1 = df1[df1['TestStage'].isin(teststage_selection)]
            if sheartype_selection: df1 = df1[df1['Shear Plane Type'].isin(sheartype_selection)]

        x = df1["NormalStress"]
        y = df1["ShearStress"]

        fitmethod = st.radio("Fit Method",
            ('Linear', 'Power'), horizontal=True)

        if len(x) > 0:
            if fitmethod == 'Linear':

                # Auto Fit
                params, params_covariance = curve_fit(weakfunc, x, y)
                auto_c, auto_f = params # automatic cohesion and friction angle

                # R squated Calculation
                residuals = y - weakfunc(x,*params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r_sq_dst = 1 - (ss_res/ss_tot)

                x_line = np.arange(0, max(x), 1)
                fit_curve = pd.DataFrame({'x_line':x_line})
                fit_curve['y_line'] = fit_curve.apply(lambda x_l: weakfunc(x_l,auto_c,auto_f))

                # One standard deviation
                sd_c, sd_f = sqrt(diag(params_covariance))
                c_sd_low = auto_c - sd_c
                f_sd_low = auto_f - sd_f
                sd_low_curve = pd.DataFrame({'x_line':x_line})
                sd_low_curve['y_line'] = sd_low_curve.apply(lambda x_l: weakfunc(x_l,c_sd_low,f_sd_low))
                c_sd_high = auto_c + sd_c
                f_sd_high = auto_f + sd_f
                sd_high_curve = pd.DataFrame({'x_line':x_line})
                sd_high_curve['y_line'] = sd_high_curve.apply(lambda x_l: weakfunc(x_l,c_sd_high,f_sd_high))

                # Clean it up for table
                auto_c = round(auto_c,2)
                auto_f = round(degrees(auto_f),2)
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

                # R squated Calculation
                residuals = y - powercurve(x,*params)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y-np.mean(y))**2)
                r_sq_dst = 1 - (ss_res/ss_tot)

                x_line = np.arange(0, max(x), 1)
                fit_curve = pd.DataFrame({'x_line':x_line})
                fit_curve['y_line'] = fit_curve.apply(lambda x_l:powercurve(x_l,auto_c,auto_f))

                # One standard deviation
                sd_c, sd_f = sqrt(diag(params_covariance))
                c_sd_low = auto_c - sd_c
                f_sd_low = auto_f - sd_f
                # print(auto_c, sd_c, auto_f, sd_f)
                # print(c_sd_low, f_sd_low)
                sd_low_curve = pd.DataFrame({'x_line':x_line})
                sd_low_curve['y_line'] = sd_low_curve.apply(lambda x_l:powercurve(x_l,c_sd_low,f_sd_low))
                c_sd_high = auto_c + sd_c
                f_sd_high = auto_f + sd_f
                sd_high_curve = pd.DataFrame({'x_line':x_line})
                sd_high_curve['y_line'] = sd_high_curve.apply(lambda x_l:powercurve(x_l,c_sd_high,f_sd_high))

                # Clean up
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

            colormethod = st.radio("Color By",
                ('Rock Type', 'HoleID', 'TestStage', 'Shear Plane Type'),
                horizontal=True)
            figd = px.scatter(
                df1, x="NormalStress", y="ShearStress",
                color=colormethod, color_discrete_sequence=colors)

            figd.update_traces(marker=dict(size=9))

            figd.add_trace(go.Scatter(x=dman["sigN"], y=dman["sigT"],
                mode='lines', name='Manual Fit',
                line=dict(dash='dash', color=colors[8])))

            figd.add_trace(go.Scatter(x=fit_curve['x_line'], y=fit_curve['y_line'],
                mode='lines', name=f'Curve Fit - {fitmethod}',
                line=dict(color='grey')))

            figd.add_trace(go.Scatter(x=sd_low_curve['x_line'], y=sd_low_curve['y_line'],
                mode='lines', name=f'-1 STD',
                line=dict(dash='dash', color=colors[7])))

            figd.add_trace(go.Scatter(x=sd_high_curve['x_line'], y=sd_high_curve['y_line'],
                mode='lines', name=f'+1 STD',
                line=dict(dash='dash', color=colors[7])))

            num_dst = len(df1.index)
            figd.update_layout(
                title_text=f"No. of Data: {num_dst}",
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',)
            figd.update_xaxes(title_text='Normal Stress', gridcolor='lightgrey',
                zeroline=True, zerolinewidth=2, zerolinecolor='black')
            figd.update_yaxes(title_text='Shear Stress', gridcolor='lightgrey',
                zeroline=True, zerolinewidth=2, zerolinecolor='black')
            figd.add_shape(
                type="rect", xref="paper", yref="paper",
                x0=0, y0=0, x1=1.0, y1=1.0,
                line=dict(color="black", width=1))

            # Table
            dst_summary = pd.DataFrame(columns=["Method", val_1, val_2, "R squared"])
            to_append = [fitmethod, auto_c, auto_f, r_sq_dst]
            new_row = len(dst_summary)
            dst_summary.loc[new_row] = to_append
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


        st.plotly_chart(figd, use_container_width=True)

        st.table(dst_summary)

        st.header("Dataset")
        st.table(df1[['HoleID', 'Rock Type', 'TestStage', 'NormalStress', 'ShearStress', 'Peak or Residual', 'Shear Plane Type']])

    # ------------ UCS and Rock TXL -------------------
    def calc_hoek(sumx, sumy, sumxy, sumxsq, sumysq, n):
        calc_sigci = sqrt((sumy/n)-((sumxy-(sumx*sumy/n))/(sumxsq-(sumx**2/n)))*sumx/n)
        calc_mi = ((sumxy-(sumx*sumy/n))/(sumxsq-(sumx**2/n)))/calc_sigci
        calc_r_sq = (sumxy-(sumx*sumy/n))**2/((sumxsq-sumx**2/n)*(sumysq-sumy**2/n))
        return calc_sigci, calc_mi, calc_r_sq

    def get_curve_df(c_sigci, c_mi, x, y):
        # For Figure
        c_tensile = - int(c_sigci / (8.62 + 0.7 * c_mi))
        Sigma3 = list(range(c_tensile, int(x.max())))
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

    with tab2:
        method_selection = ('Hoek Calculation', 'Scipy Curve Fit')
        calc_selection = ('All data', 'HTX + UCS(mean)', 'HTX only')

        st.header("UCS and Rock TXL")
        du = df[df['TestType'].isin(['Uniax', 'Triax', 'Brazilian'])]
        du = du[du['Sigma3'].notna()]
        du = du[du['PeakSigma1'].notna()]

        # Auto Fit - Method and dataset
        col3, col4, col5 = st.columns(3)
        with col3:
            calc_method = st.radio("Auto Fit method",
                method_selection, horizontal=True)
        with col4:
            calc_data = st.radio("Select dataset",
                calc_selection, horizontal=True)
        with col5:
            # Figure
            colormethod_u = st.radio("Color By",
                ('SampleType', 'Rock Type', 'HoleID'),
                horizontal=True)

        # figu = go.Figure()
        figu = px.scatter(
                du, x="Sigma3", y="PeakSigma1",
                color=colormethod_u, color_discrete_sequence=colors)
        figu.update_traces(marker=dict(size=9))

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
        dt = du[du['Sigma3']!=0]
        dt = dt[dt['PeakSigma1']!=0]
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
            sig3 = du["Sigma3"]
            sig1 = du["PeakSigma1"]
        elif calc_data==calc_selection[1]:
            sig3 = du_auto["Sigma3"]
            sig1 = du_auto["PeakSigma1"]
        else:
            sig3 = dt["Sigma3"]
            sig1 = dt["PeakSigma1"]

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

        else:
            params, params_covariance = curve_fit(objective, sig3, sig1)
            auto_sigci, auto_mi = params # a = sigci, b = mi
            auto_sigci = round(auto_sigci,2)
            auto_mi = round(auto_mi,2)

            c_tens, c_curv, c_vert, r_sq_c = get_curve_df(auto_sigci, auto_mi, sig3, sig1)

            # residuals = y - objective(x,*params)
            # ss_res = np.sum(residuals**2)
            # ss_tot = np.sum((y-np.mean(y))**2)
            # r_sq_curve = 1 - (ss_res/ss_tot)
            # r_sq_curve = round(r_sq_curve,4)
            # auto_tensile = - int(auto_sigci / (8.62 + 0.7 * auto_mi))
            # x_line = np.arange(auto_tensile, max(x), 1)
            # fit_curve = pd.DataFrame({'x_line':x_line})
            # fit_curve['y_line'] = fit_curve.apply(lambda x_l: objective(x_l,auto_sigci,auto_mi))

            # # Vertical Line for Tensile Cutoff
            # auto_vertical = fit_curve[fit_curve['x_line']==auto_tensile]
            # to_append = [auto_tensile, 0]
            # new_row = len(auto_vertical)
            # auto_vertical.loc[new_row] = to_append

        # Calculated Figure
        # if calc_data==calc_selection[0]:
        figu.add_trace(
            go.Scatter(x=c_curv['Sigma3'], y=c_curv['Sigma1'],
                mode='lines', name=f'Calculated - {calc_selection[0]}',
                line=dict(color=colors[2])))

        figu.add_trace(
            go.Scatter(x=c_vert['Sigma3'], y=c_vert['Sigma1'],
                mode='lines', name='Calc ALL Vertical', showlegend = False,
                line=dict(color=colors[2])))

        # Table - Auto Fit
        ucs_summary = pd.DataFrame(columns=["Method", "Sigci", "Mi", "Tensile Cutoff", "R squared"])
        to_append = [f"Auto Fit: {calc_method} - {calc_data}", auto_sigci, auto_mi, -c_tens, r_sq_c]
        new_row = len(ucs_summary)
        ucs_summary.loc[new_row] = to_append

        # Manual Fit Curve
        manual_on = st.radio("Manual Fit On or Off?",
            ('Off', 'On'), horizontal=True)
        if manual_on == 'On':
            col1, col2 = st.columns(2)
            with col1:
                sigci = st.number_input('Manual sigci', value=mean_ucs)
            with col2:
                mi = st.number_input('Manual mi', value=auto_mi)

            m_tens, m_curv, m_vert, r_sq_m = get_curve_df(sigci, mi, sig3, sig1)

            # manual_tensile = - int(sigci / (8.62 + 0.7 * mi))
            # params_man = [sigci, mi]
            # residuals = y - objective(x,*params_man)
            # ss_res = np.sum(residuals**2)
            # ss_tot = np.sum((y-np.mean(y))**2)
            # r_sq_curve_man = 1 - (ss_res/ss_tot)
            # r_sq_curve_man = round(r_sq_curve_man,4)

            # Sigma3 = list(range(manual_tensile, int(du['Sigma3'].max())))
            # hcurv = pd.DataFrame({'Sigma3':Sigma3})
            # hcurv['Sigma1'] = hcurv.apply(lambda row: row['Sigma3']+sigci*sqrt((mi*row['Sigma3']/sigci)+1), axis=1)

            # manual_vertical = hcurv[hcurv['Sigma3']==manual_tensile]
            # to_append = [manual_tensile, 0]
            # new_row = len(manual_vertical)
            # manual_vertical.loc[new_row] = to_append

            # Manual Curve - Figure
            figu.add_trace(
                go.Scatter(x=m_curv['Sigma3'], y=m_curv['Sigma1'],
                    mode='lines', name='Manual Fit Curve',
                    line=dict(dash='dash', color=base_colors['line'])))

            figu.add_trace(
                go.Scatter(x=m_vert['Sigma3'], y=m_vert['Sigma1'],
                    mode='lines', name='Manual Vertical', showlegend = False,
                    line=dict(dash='dash', color=base_colors['line'])))

            to_append = ["Manual", sigci, mi, -m_tens, r_sq_m]
            new_row = len(ucs_summary)
            ucs_summary.loc[new_row] = to_append

        # figu.add_trace(
        #     go.Scatter(x=ducs_mean['Sigma3'], y=ducs_mean['PeakSigma1'],
        #         name='Mean UCS', mode='markers',
        #         marker=dict(size=15, color=colors[-3])))

        # figu.add_trace(
        #     go.Scatter(x=dbzt_mean['Sigma3'], y=dbzt_mean['PeakSigma1'],
        #         name='Mean Brazilian', mode='markers',
        #         marker=dict(size=15, color=colors[-4])))


        # else:
        #     figu.add_trace(
        #         go.Scatter(x=calc_curv_txl['Sigma3'], y=calc_curv_txl['Sigma1'],
        #             mode='lines', name=f'Calculated - {calc_selection[1]}',
        #             line=dict(color=colors[0])))

        #     figu.add_trace(
        #         go.Scatter(x=cal_vert_txl['Sigma3'], y=cal_vert_txl['Sigma1'],
        #             mode='lines', name='Calc TXL Vertical', showlegend = False,
        #             line=dict(color=colors[0])))


        # Boxplot
        ducs = du[du['Sigma3']==0]
        ducs = ducs[['Sigma3', 'PeakSigma1']]

        figu.add_trace(
            go.Box(x=ducs['Sigma3'], y=ducs['PeakSigma1'], width=int(max(du['Sigma3'])/20),
                name='UCS box plot', quartilemethod="linear",
                marker_color = 'indianred'))

        num_ucs = len(du.index)
        figu.update_layout(
                title_text=f"No. of Data: {num_ucs}",
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',)
        figu.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1.0, y1=1.0,
            line=dict(color="black", width=1))
        figu.update_xaxes(title_text='Sigma 3', gridcolor='lightgrey',
            zeroline=True, zerolinewidth=2, zerolinecolor='black')
        figu.update_yaxes(title_text='Peak Sigma 1', gridcolor='lightgrey',
            zeroline=True, zerolinewidth=2, zerolinecolor='black')

        st.plotly_chart(figu, use_container_width=True)

        st.header("Summary")
        st.table(ucs_summary)

        st.header("Dataset")
        st.table(du[['HoleID', 'Rock Type', 'Sigma3', 'PeakSigma1', 'Failure Mode']])

else:
    st.header('Select a File to start')

