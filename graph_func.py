import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from bokeh.plotting import figure, show
import altair as alt
import numpy as np

def scatter_altair(title, rotation, xcol, ycol, colorcol, df, dfub, dflb, dfb, dman, man_on):
    # selection = alt.selection_multi(fields=[colorcol], bind='legend')
    selection = alt.selection_interval()
    pnt=alt.Chart(df, title=title).mark_circle(size=80).encode(
        x=alt.X(xcol, sort=None, title=xcol), # scale=alt.Scale(type='log', base=10, domain=[0, 2])),
        y=alt.Y(ycol, axis=alt.Axis(grid=True), title=ycol),
        # color=alt.condition(selection, colorcol, alt.value('lightgrey')),
        color=alt.Color(colorcol, sort=None, legend=alt.Legend(title=colorcol)),
        opacity=alt.condition(selection, alt.value(0.8), alt.value(0.2)),
        tooltip=[colorcol, xcol, ycol]).add_selection(selection) #.interactive()
    # print(selection)

    lb_line = alt.Chart(dflb).mark_line(color='black', opacity=0.5, size=4).encode(
        x=alt.X('x', sort=None),
        y=alt.Y('y'), #color=alt.Color('line', sort=None),
        strokeDash=alt.StrokeDash('line', sort=None),
        tooltip=['line'])

    bs_line = alt.Chart(dfb).mark_line(color='black', opacity=0.8, size=3).encode(
        x=alt.X('x', sort=None),
        y=alt.Y('y'), #color=alt.Color('line', sort=None),
        strokeDash=alt.StrokeDash('line', sort=None, legend=alt.Legend(title='Line of Fit')),
        tooltip=['line'])

    ub_line = alt.Chart(dfub).mark_line(color='black', opacity=0.5, size=4).encode(
        x=alt.X('x', sort=None),
        y=alt.Y('y'), #color=alt.Color('line', sort=None, #scale=alt.Scale(scheme='dark2),
            # legend=alt.Legend(title=colorcol)),
        strokeDash=alt.StrokeDash('line', sort=None),
        tooltip=['line'])

    if man_on=='on':
        man_line = alt.Chart(dman).mark_line(color='red', opacity=0.8, size=3).encode(
            x=alt.X('x', sort=None),
            y=alt.Y('y'), color=alt.Color('line', sort=None),
            tooltip=['line'])

    if man_on=='on':
        fig = alt.layer(pnt, bs_line, ub_line, lb_line, man_line)
    else:
        fig = alt.layer(pnt, bs_line, ub_line, lb_line)

    fig = fig.properties(height=600).configure_axis(
        labelFontSize=14,titleFontSize=15).configure_legend(
        titleFontSize=14,labelFontSize=14)
    return fig

def scatter_altair_u(title, rotation, xcol, ycol, colorcol, df, dfub, dflb, dfb, ducs, dauto, dman, man_on):
    selection = alt.selection_multi(fields=[colorcol], bind='legend')
    pnt=alt.Chart(df, title=title).mark_circle(size=80).encode(
        x=alt.X(xcol, sort=None, title=xcol), # scale=alt.Scale(type='log', base=10, domain=[0, 2])),
        y=alt.Y(ycol, axis=alt.Axis(grid=True), title=ycol),
        color=alt.Color(colorcol, sort=None, legend=alt.Legend(title=colorcol)),
        opacity=alt.condition(selection, alt.value(0.9), alt.value(0.1)),
        tooltip=[colorcol, xcol, ycol]).add_selection(selection) #.interactive()

    lb_line = alt.Chart(dflb).mark_line(color='black', opacity=0.5, size=4).encode(
        x=alt.X('x', sort=None),
        y=alt.Y('y'), #color=alt.Color('line', sort=None),
        strokeDash=alt.StrokeDash('line', sort=None),
        tooltip=['line'])

    bs_line = alt.Chart(dfb).mark_line(color='black', opacity=0.8, size=3).encode(
        x=alt.X('x', sort=None),
        y=alt.Y('y'), #color=alt.Color('line', sort=None),
        strokeDash=alt.StrokeDash('line', sort=None, legend=alt.Legend(title='Line of Fit')),
        tooltip=['line'])

    ub_line = alt.Chart(dfub).mark_line(color='black', opacity=0.5, size=4).encode(
        x=alt.X('x', sort=None),
        y=alt.Y('y'), #color=alt.Color('line', sort=None, #scale=alt.Scale(scheme='dark2),
            # legend=alt.Legend(title=colorcol)),
        strokeDash=alt.StrokeDash('line', sort=None),
        tooltip=['line'])

    boxplot = alt.Chart(ducs).mark_boxplot(color='grey', ticks=True, size=30, opacity=0.7).encode(
        x=alt.X(xcol),
        y=alt.Y(ycol),)
        #color=alt.Color('line', sort=None))

    auto_curve = alt.Chart(dauto).mark_line(opacity=0.8, size=3).encode(
        x=alt.X('x', sort=None),
        y=alt.Y('y'), color=alt.Color('line', sort=None),
        tooltip=['line'])

    if man_on=='on':
        man_line = alt.Chart(dman).mark_line(color='red', opacity=0.8, size=3).encode(
            x=alt.X('x', sort=None),
            y=alt.Y('y'), color=alt.Color('line', sort=None),
            tooltip=['line'])

    if man_on=='on':
        fig = alt.layer(boxplot, pnt, bs_line, ub_line, lb_line, auto_curve, man_line)
    else:
        fig = alt.layer(boxplot, pnt, bs_line, ub_line, lb_line, auto_curve)

    fig = fig.properties(height=600).configure_axis(
        labelFontSize=14,titleFontSize=15).configure_legend(
        titleFontSize=14,labelFontSize=14)
    return fig

# if fig_method == fig_selection[0]:
#     figd = px.scatter(
#         df1, x="NormalStress", y="ShearStress",
#         color=colormethod_d, color_discrete_sequence=colors)

#     figd.update_traces(marker=dict(size=9))

#     if not np.isnan(manual_val_1):
#         figd.add_trace(go.Scatter(x=dman["sigN"], y=dman["sigT"],
#             mode='lines', name='Manual Fit',
#             line=dict(dash='dash', color=colors[8])))

#     # Linear and Power
#     else:
#         figd.add_trace(go.Scatter(x=dbs['x'], y=dbs['y'],
#             mode='lines', name=f'OLS',
#             line=dict(color='grey')))

#         figd.add_trace(go.Scatter(x=dlq['x'], y=dlq['y'],
#             mode='lines', name=f'{int(lq*100)}th Quantile',
#             line=dict(dash='dash', color=colors[6])))

#         figd.add_trace(go.Scatter(x=duq['x'], y=duq['y'],
#             mode='lines', name=f'{int(uq*100)}th Quantile',
#             line=dict(dash='dash', color=colors[5])))

#     num_dst = len(df1.index)
#     figd.update_layout(
#         title_text=f"No. of Data: {num_dst}",
#         plot_bgcolor='#FFFFFF',
#         paper_bgcolor='#FFFFFF',
#         height=600,)

#     figd.update_xaxes(title_text='Normal Stress', gridcolor='lightgrey',
#         zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
#         tickformat=",.0f")
#     figd.update_yaxes(title_text='Shear Stress', gridcolor='lightgrey',
#         zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
#         tickformat=",.0f", range=[0,max(y)*1.05])
#     figd.add_shape(
#         type="rect", xref="paper", yref="paper",
#         x0=0, y0=0, x1=1.0, y1=1.0,
#         line=dict(color="black", width=2))
# else:
#     # Bokeh graph
#     uniq = df1[colormethod_d].unique()
#     color_map = bmo.CategoricalColorMapper(factors=uniq, palette=colors)
#     source = ColumnDataSource(df1)
#     hover = HoverTool(tooltips=[
#         ('Hole ID', '@HoleID'),
#         ("Normal Stress", "@NormalStress"),
#         ("Shear Stress", "@ShearStress"),
#         ])
#     p=figure(tools=[hover], x_axis_label='Normal Stress', y_axis_label='Shear Stress')

#     p.scatter(x='NormalStress', y='ShearStress', size=9,
#         color={'field': colormethod_d, 'transform': color_map},
#         legend_group=colormethod_d, source=source)

#     # Barton Bandis
#     if fitmethod == fit_selection[1]:
#         source = ColumnDataSource(fit_curve)
#         p.line(x='x_line', y='y_line', line_width=2, line_color='grey',
#             legend_label=f'{fitmethod} Curve Fit', source=source)

#         source = ColumnDataSource(sd_low_curve)
#         p.line(x='x_line', y='y_line', line_width=2, line_color=colors[6],
#             legend_label=f'-{lq_sd} Phir St.D', source=source)

#         source = ColumnDataSource(sd_high_curve)
#         p.line(x='x_line', y='y_line', line_width=2, line_color=colors[5],
#             legend_label=f'+{uq_sd} Phir St.D', source=source)

#     # Linear and Power
#     else:
#         source = ColumnDataSource(dbs)
#         p.line(x='x', y='y', line_width=2, line_color='grey',
#             legend_label=f'OLS', source=source)

#         source = ColumnDataSource(dlq)
#         p.line(x='x', y='y', line_width=2, line_color=colors[6],
#             legend_label=f'{lq_value}% Bound', source=source)

#         source = ColumnDataSource(duq)
#         p.line(x='x', y='y', line_width=2, line_color=colors[5],
#             legend_label=f'{uq_value}% Bound', source=source)

#     if manual_on_p == 'on':
#         source = ColumnDataSource(dman)
#         p.line(x='sigN', y='sigT', line_width=2, line_color='green',
#             legend_label='Manual Fit', source=source)

#     p.legend.location = "top_left"

# Plotly
def scatter_plotly(title, rotation, xcol, ycol, colorcol, df, dfub, dflb, dfb, dman, man_on):
    colors = px.colors.qualitative.T10
    colors_selected = ["rgba(99,110,250,0.8)", "rgba(99,110,250,0.2)"]
    num_set = len(colorcol)
    # fig = px.scatter(
    #     df, x=xcol, y=ycol,
    #     color=colorcol, color_discrete_sequence=colors)
    # fig.update_traces(marker=dict(size=9))

    fig = go.Figure()
    if dfb is not None:
        fig.add_trace(go.Scatter(x=dfb['x'], y=dfb['y'],
            mode='lines', name=f'Linear Reg.',
            line=dict(color='black')))

    # Find the different groups
    groups = df[colorcol].unique()
    groups = sorted(groups, reverse=True)

    # Create as many traces as different groups there are and save them in data list
    i=0
    for group in groups:
        df_group = df[df[colorcol] == group]
        trace = go.Scatter(x=df_group[xcol],
            y=df_group[ycol],
            mode='markers',
            name=str(group),
            marker=dict(color=colors_selected[i],
                size=9),)
            # marker=dict(color=colors[i], size=9),)
        fig.add_trace(trace)
        i += 1
    # fig.add_traces(
    #     list(px.line(dfb, x='x',y='y').select_traces()))
    fig.data = [fig.data[i] for i in reversed(range(len(fig.data)))]
# figu.add_trace(
#     go.Scatter(x=dlq['x'], y=dlq['y'],
#         mode='lines', name=f'{int(lq_value)}% Bound',
#         line=dict(dash='dash', color=colors[6])))

# figu.add_trace(
#     go.Scatter(x=duq['x'], y=duq['y'],
#         mode='lines', name=f'{int(uq_value)}% Bound',
#         line=dict(dash='dash', color=colors[5])))

# # figu.add_trace(
# #     go.Scatter(x=c_line['Sigma3'], y=c_line['Sigma1'],
# #         mode='lines', name=f'Linear Regression',
# #         line=dict(color=colors[-1])))

# # x1 = [x for x in c_line['Sigma3']]
# # y_upper = [y for y in c_line['High_Std_Sigma1']]
# # y_lower = [y for y in c_line['Low_Std_Sigma1']]
# # figu.add_trace(
# #     go.Scatter(
# #         x=x1+x1[::-1],
# #         y=y_upper+y_lower[::-1],
# #         fill='toself', fillcolor='rgba(0,100,80,0.2)',
# #         line=dict(color='rgba(255,255,255,0)'),
# #         hoverinfo="skip", name='Linear: ±1 Std. Dev'))

# figu.add_trace(
#     go.Scatter(x=c_curv['Sigma3'], y=c_curv['Sigma1'],
#         mode='lines', name=f'Auto Fit',
#         line=dict(color='black')))

# figu.add_trace(
#     go.Scatter(x=c_vert['Sigma3'], y=c_vert['Sigma1'],
#         mode='lines', name='Calc ALL Vertical', showlegend = False,
#         line=dict(color='black')))
# # Boxplot
# ducs = du[du['Sigma3']==0]
# ducs = ducs[['Sigma3', 'PeakSigma1']]

# figu.add_trace(
#     go.Box(x=ducs['Sigma3'], y=ducs['PeakSigma1'],
#         width=int((max(du['Sigma3'])-min(du['Sigma3']))/20),
#         name='UCS box plot', quartilemethod="linear",
#         marker_color = 'indianred'))

    num_data = len(df.index)
    fig.update_layout(
        title_text=f"No. of Data: {num_data}",
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        height=600,
        dragmode='lasso')

    fig.update_xaxes(title_text=xcol, gridcolor='lightgrey',
        zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
        tickformat=",.0f", ticks="outside", ticklen=5,
        range=[0,max(df[xcol])*1.05])
    fig.update_yaxes(title_text=ycol, gridcolor='lightgrey',
        zeroline=True, zerolinewidth=3, zerolinecolor='lightgrey',
        tickformat=",.0f", ticks="outside", ticklen=5,
        range=[0,max(df[ycol])*1.05])
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1.0, y1=1.0,
        line=dict(color="black", width=2))
    return fig

# # Manual Fit Curve
# if manual_on == 'on':
#     # m_tens, m_curv, m_vert, r_sq_m = get_curve_df(sigci, mi, sig3, sig1)

#     # Manual Curve - Figure
#     figu.add_trace(
#         go.Scatter(x=m_curv['Sigma3'], y=m_curv['Sigma1'],
#             mode='lines', name='Manual Fit Curve',
#             line=dict(dash='dash', color=base_colors['line'])))

#     figu.add_trace(
#         go.Scatter(x=m_vert['Sigma3'], y=m_vert['Sigma1'],
#             mode='lines', name='Manual Vertical', showlegend = False,
#             line=dict(dash='dash', color=base_colors['line'])))

#     to_append = ["Manual", sigci, mi, -m_tens]
#     new_row = len(ucs_summary)
#     ucs_summary.loc[new_row] = to_append

            # # Bokeh graph
            # uniq = du[colormethod_u].unique()
            # sel_colors = colors[:len(uniq)]
            # dict_col = {typ:cl for typ,cl in zip(uniq, sel_colors)}
            # du['color'] = du[colormethod_u].map(dict_col)

            # color_map = bmo.CategoricalColorMapper(factors=du[colormethod_u].unique(), palette=colors)
            # source = ColumnDataSource(du)
            # hover = HoverTool(tooltips=[
            #     ('HoleID', '@HoleID'),
            #     ("Sigma3", "@Sigma3"),
            #     ("PeakSigma1", "@PeakSigma1"),
            #     ])
            # p=figure(tools=[hover], x_axis_label='Sigma 3', y_axis_label='Peak Sigma 1')

            # if len(ducs) > 0:
            #     q1 = ducs['PeakSigma1'].quantile(q=0.25)
            #     q2 = ducs['PeakSigma1'].quantile(q=0.5)
            #     q3 = ducs['PeakSigma1'].quantile(q=0.75)
            #     width = int((max(du['Sigma3'])-min(du['Sigma3']))/20)

            #     p.vbar([0], width, q2, q3, fill_color='indianred', line_color="black", legend_label="boxplot",)
            #     p.vbar([0], width, q1, q2, fill_color='indianred', line_color="black")

            # p.scatter(x='Sigma3', y='PeakSigma1', size=9,
            #     color={'field': colormethod_u, 'transform': color_map},
            #     legend_group=colormethod_u, source=source)

            # source = ColumnDataSource(c_line)
            # p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='lightgrey',
            #     legend_label='Linear Regression', source=source)

            # source = ColumnDataSource(c_curv)
            # p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='black',
            #     legend_label='Auto Fit', source=source)

            # source = ColumnDataSource(c_vert)
            # p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='black', source=source)

            # if manual_on == 'On':
            #     source = ColumnDataSource(m_curv)
            #     p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='green',
            #         legend_label='Manual Fit', source=source)

            #     source = ColumnDataSource(m_vert)
            #     p.line(x='Sigma3', y='Sigma1', line_width=2, line_color='green', source=source)

            # source = ColumnDataSource(c_line)
            # band = Band(base='Sigma3', lower='Low_Std_Sigma1', upper='High_Std_Sigma1', source=source,
            #     level='underlay', fill_alpha=0.5, line_width=1, line_color='black')
            # p.add_layout(band)

            # # p.circle(xx, yy, fill_color="blue", size=9, legend_label="data",)
            # p.legend.location = "top_left"
