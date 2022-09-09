import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import curve_fit

from numpy import sin, cos, tan, arcsin, arccos, arctan, radians, degrees, sqrt, diag, log10

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
        # st.header("DST")

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

        # color_list = set(color_list)

        col4, col5 = st.columns(2)

        fit_selection = ('Linear', 'Barton Bandis', 'Power')
        with col4:
            fitmethod = st.radio("Auto Fit Method",
                fit_selection, horizontal=True)
        with col5:
            colormethod_d = st.radio("Color By",
                color_list, horizontal=True)

        if len(x) > 0:
            if fitmethod == fit_selection[0]:

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
                # print(auto_f, sd_f, degrees(auto_f), degrees(sd_f))
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
                sd_c = round(sd_c,2)
                sd_f = round(degrees(sd_f),2)
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
                        d_coh = st.number_input('Manual cohesion', value=auto_c)
                    with col3:
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
                else:
                    manual_val_1 = np.nan

            # Barton Bandis:
            elif fitmethod == fit_selection[1]:
                col1, col2 = st.columns(2)
                with col1:
                        inp_jrc = st.number_input('Enter jrc', value=2.5)
                with col2:
                        inp_jcs = st.number_input('Enter jcs', value=100)

                def bartonbandis(x, phir):
                    return x * tan(radians(phir + inp_jrc * log10(inp_jcs / x)))
                # Auto Fit
                params, params_covariance = curve_fit(bartonbandis, x, y, bounds=((0),(90)))

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
                phir_sd_low = auto_phir - sd_phir
                # jrc_sd_low = auto_jrc - sd_jrc
                # jcs_sd_low = auto_jcs - sd_jcs
                sd_low_curve = pd.DataFrame({'x_line':x_line})
                sd_low_curve['y_line'] = sd_low_curve.apply(lambda x_l: bartonbandis(x_l,phir_sd_low))
                phir_sd_high = auto_phir + sd_phir
                # jrc_sd_high = auto_jrc + sd_jrc
                # jcs_sd_high = auto_jcs + sd_jcs
                sd_high_curve = pd.DataFrame({'x_line':x_line})
                sd_high_curve['y_line'] = sd_high_curve.apply(lambda x_l: bartonbandis(x_l,phir_sd_high))

                # Clean it up for table
                auto_phir = round(auto_phir,2)
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
                    # col1, col2 = st.columns(2)
                    with col2:
                        d_phir = st.number_input('Manual phir', value=auto_phir)
                    # with col3:
                    #     d_jrc = st.number_input('Manual jrc', value=inp_jrc)
                    # with col4:
                    #     d_jcs = st.number_input('Manual jcs', value=inp_jcs)

                    # signmin = 10**(log10(inp_jcs)-((70-d_phir)/inp_jrc))
                    # if signmin < 1: signmin = 1
                    sigN = list(range(1, int(max(x)), 1))
                    dman = pd.DataFrame({'sigN':sigN})
                    dman['sigT'] = dman.apply(lambda row: bartonbandis(row['sigN'], d_phir), axis=1)
                    params_man = [d_phir]
                    residuals = y - bartonbandis(x,*params_man)
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y-np.mean(y))**2)
                    r_sq_dst_m = 1 - (ss_res/ss_tot)
                    manual_val_1 = d_phir
                    manual_val_2 = inp_jrc
                    manual_val_3 = inp_jcs
                    manual_val_4 = r_sq_dst_m
                else:
                    manual_val_1 = np.nan

            # Power Fit
            elif fitmethod == fit_selection[2]:

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
                c_sd_low = auto_c
                f_sd_low = auto_f - sd_f

                sd_low_curve = pd.DataFrame({'x_line':x_line})
                sd_low_curve['y_line'] = sd_low_curve.apply(lambda x_l:powercurve(x_l,c_sd_low,f_sd_low))
                c_sd_high = auto_c
                f_sd_high = auto_f + sd_f
                sd_high_curve = pd.DataFrame({'x_line':x_line})
                sd_high_curve['y_line'] = sd_high_curve.apply(lambda x_l:powercurve(x_l,c_sd_high,f_sd_high))

                # Clean up
                # print(auto_c, auto_f)
                auto_c = round(auto_c,4)
                auto_f = round(auto_f,4)
                r_sq_dst = round(r_sq_dst,4)
                sd_c = round(sd_c,4)
                sd_f = round(sd_f,4)
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
                        k_pow = st.number_input('Manual k', value=auto_c)
                    with col3:
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

                source = ColumnDataSource(fit_curve)
                p.line(x='x_line', y='y_line', line_width=2, line_color='grey',
                    legend_label=f'Curve Fit - {fitmethod}', source=source)

                source = ColumnDataSource(sd_low_curve)
                p.line(x='x_line', y='y_line', line_width=2, line_color=colors[7],
                    legend_label=f'-1 STD', source=source)

                source = ColumnDataSource(sd_high_curve)
                p.line(x='x_line', y='y_line', line_width=2, line_color=colors[7],
                    legend_label=f'+1 STD', source=source)

                if manual_on_p == 'on':
                    source = ColumnDataSource(dman)
                    p.line(x='sigN', y='sigT', line_width=2, line_color='green',
                        legend_label='Manual Fit', source=source)

                p.legend.location = "top_left"

            # Table
            if fitmethod != fit_selection[1]:
                dst_summary = pd.DataFrame(columns=[" ", val_1, val_2, "R squared"])
                to_append = ["Auto Fit - Mean", auto_c, auto_f, r_sq_dst]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                to_append = ["Auto Fit - STD", sd_c, sd_f, np.nan]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                if not np.isnan(manual_val_1):
                    to_append = ['Manual Fit', manual_val_1, manual_val_2, manual_val_3]
                    new_row = len(dst_summary)
                    dst_summary.loc[new_row] = to_append
            else:
                dst_summary = pd.DataFrame(columns=[" ", val_1, val_2, val_3, "R squared"])
                to_append = ["Auto Fit - Mean", int(auto_phir), int(auto_jrc), int(auto_jcs), r_sq_dst]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                to_append = ["Auto Fit - STD", sd_phir, sd_jrc, sd_jcs, np.nan]
                new_row = len(dst_summary)
                dst_summary.loc[new_row] = to_append
                if not np.isnan(manual_val_1):
                    to_append = ['Manual Fit', manual_val_1, manual_val_2, manual_val_3, manual_val_4]
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
        st.dataframe(dst_summary.style.format(precision=2)) #style.set_precision(2))
        # AgGrid(dst_summary, fit_columns_on_grid_load=True, height=110)

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
        Sigma3 = list(range(c_tensile, int(x.max()+0.15*x.max())))
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

    def lin_reg(x, a, b):
        return a + b*x

    def get_linear_df(a, b, p_cov, x, y):
        # For Figure
        t_fri = arcsin((b-1)/(b+1))
        t_coh = a * (1-sin(t_fri))/(2*cos(t_fri))
        min_val = int((max(du['Sigma3'])-min(du['Sigma3']))/40)
        Sigma3 = list(range(min_val, int(x.max()+0.15*x.max())))
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
        c_line['Low_Std_Sigma1'] = c_line.apply(lambda row: (a-sd_a)+(b-sd_b)*row['Sigma3'], axis=1)
        c_line['High_Std_Sigma1'] = c_line.apply(lambda row: (a+sd_a)+(b+sd_b)*row['Sigma3'], axis=1)

        return t_coh, degrees(t_fri), r_sq_l, c_line

    # with tab2:
    if selected == options[1]:
        method_selection = ('Hoek Calculation', 'Scipy Curve Fit')
        calc_selection = ('All data', 'HTX + UCS(mean)', 'HTX only')

        du = df[df['TestType'].isin(['Uniax', 'Triax', 'Brazilian'])]
        du = du[du['Sigma3'].notna()]
        du = du[du['PeakSigma1'].notna()]

        if len(du['Sigma3'])>0:
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
                ('SampleType', 'Rock Type', 'HoleID', 'Failure Mode'),
                horizontal=True)

        if len(du['Sigma3']) > 0:

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
            t_coh, t_fri, r_sq_l, c_line = get_linear_df(a, b, params_covariance, sig3, sig1)

            # Calculated Figure
            figu = px.scatter(
                du, x="Sigma3", y="PeakSigma1",
                color=colormethod_u, color_discrete_sequence=colors)
            figu.update_traces(marker=dict(size=9))

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
                    hoverinfo="skip", name='Linear: ±1 STD'))

            figu.add_trace(
                go.Scatter(x=c_curv['Sigma3'], y=c_curv['Sigma1'],
                    mode='lines', name=f'Auto Fit',
                    line=dict(color='black')))

            figu.add_trace(
                go.Scatter(x=c_vert['Sigma3'], y=c_vert['Sigma1'],
                    mode='lines', name='Calc ALL Vertical', showlegend = False,
                    line=dict(color='black')))

            # Table - Auto Fit
            ucs_summary = pd.DataFrame(columns=["Method", "Sigci", "mi", "Tensile Cutoff", "R squared"])
            to_append = [f"Auto Fit: {calc_method} - {calc_data}", int(auto_sigci), int(auto_mi), -c_tens, r_sq_c]
            new_row = len(ucs_summary)
            ucs_summary.loc[new_row] = to_append

            # Table - Linear Regression
            linear_summary = pd.DataFrame(columns=["Method", "Cohesion", "Phi", "STD"])
            to_append = [f"Linear Regression", int(t_coh), int(t_fri), r_sq_l]
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

                to_append = ["Manual", sigci, mi, -m_tens, r_sq_m]
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
                font=dict(
                    family="sans serif",size=25,color="crimson"),
                showarrow=False,yshift=10)
            ucs_summary = pd.DataFrame()


        if fig_method==fig_selection[0]:
            st.plotly_chart(figu, use_container_width=True)
        else:
            st.bokeh_chart(p, use_container_width=True)

        # st.subheader("Summary")
        st.markdown("**Sigci and mi**")
        st.table(ucs_summary)

        st.markdown("**Cohesion and Friction Angle**")
        st.table(linear_summary)

        st.subheader("Dataset")
        # st.table(du[['HoleID', 'Rock Type', 'Sigma3', 'PeakSigma1', 'Failure Mode']])

        du = du.iloc[:,1:]
        du = du.dropna(axis=1, how='all')
        AgGrid(du)


else:
    st.header('Select a File to start')

