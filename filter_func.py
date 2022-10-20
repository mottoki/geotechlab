import streamlit as st
import numpy as np

def filter_df(df1, color_list, all_cols, samptype):
    for col in all_cols:
        if col in df1.columns:
            elements = set(df1[df1[col].notna()][col])
            element_selection = st.sidebar.multiselect(col, (elements), key=f'filt_{samptype}_{col}')
            if element_selection: df1 = df1[df1[col].isin(element_selection)]
    # if "Project" in df1.columns:
    #     project = set(df1['Project'])
    #     project_selection = st.sidebar.multiselect("Project", (project))
    #     if project_selection: df1 = df1[df1['Project'].isin(project_selection)]

    # if "Prospect" in df1.columns:
    #     prospect = set(df1['Prospect'])
    #     prospect_selection = st.sidebar.multiselect("Prospect", (prospect))
    #     if prospect_selection: df1 = df1[df1['Prospect'].isin(prospect_selection)]
    #     color_list.append("Prospect")

    # if "Formation" in df1.columns:
    #     formation = set(df1['Formation'])
    #     formation_selection = st.sidebar.multiselect("Formation", (formation))
    #     if formation_selection: df1 = df1[df1['Formation'].isin(formation_selection)]

    # if "SubFormation" in df1.columns:
    #     subformation = set(df1['SubFormation'])
    #     subformation_selection = st.sidebar.multiselect("SubFormation", (subformation))
    #     if subformation_selection: df1 = df1[df1['SubFormation'].isin(subformation_selection)]

    # if "Rock Type" in df1.columns:
    #     rock_type = set(df1['Rock Type'])
    #     rock_selection = st.sidebar.multiselect("Rock Type", (rock_type))
    #     if rock_selection: df1 = df1[df1['Rock Type'].isin(rock_selection)]

    # if "HoleID" in df1.columns:
    #     holeid = set(df1['HoleID'])
    #     holeid_selection = st.sidebar.multiselect("Hole ID", (holeid))
    #     if holeid_selection: df1 = df1[df1['HoleID'].isin(holeid_selection)]

    # if "SampleType" in df1.columns:
    #     sampletype = set(df1['SampleType'])
    #     testtype_selection = st.sidebar.multiselect("Test type", (sampletype))
    #     if testtype_selection: df1 = df1[df1['SampleType'].isin(testtype_selection)]

    # if "TestStage" in df1.columns:
    #     teststage = (x for x in set(df1['TestStage']) if np.isnan(x) == False)
    #     teststage_selection = st.sidebar.multiselect("Test Stage", (teststage))
    #     if teststage_selection: df1 = df1[df1['TestStage'].isin(teststage_selection)]

    # if "Shear Plane Type" in df1.columns:
    #     shear_type = set(df1['Shear Plane Type'])
    #     sheartype_selection = st.sidebar.multiselect("Shear Plane Type", (shear_type))
    #     if sheartype_selection: df1 = df1[df1['Shear Plane Type'].isin(sheartype_selection)]

    # if "Test Year" in df1.columns:
    #     testyear = set(df1['Test Year'])
    #     testyear_selection = st.sidebar.multiselect("Test Year", (testyear))
    #     if testyear_selection: df1 = df1[df1['Test Year'].isin(testyear_selection)]

    # if "Failure Mode" in df1.columns:
    #     failure_mode = set(df1['Failure Mode'])
    #     failuremode_selection = st.sidebar.multiselect("Failure Mode", (failure_mode))
    #     if failuremode_selection: du = du[du['Failure Mode'].isin(failuremode_selection)]


    return df1, color_list
