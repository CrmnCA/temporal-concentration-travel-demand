
import matplotlib
import matplotlib.pyplot as plt
import contextily as cx
import geopandas as gpd
import pandas as pd
import numpy as np
from mycolorpy import colorlist as mcp
from matplotlib import colors as mcolors
from math import pi, ceil, floor
import matplotlib.lines as mlines
import seaborn as sns




def plot_daily_throuput_stn(t, time_map, df_stn_ts, df_agg_ts):
    
    '''
    Return a plot of the daily throughput

    :param t: array of time slots
    :param time_map: dictionary to map time slots to actual times of the day
    :param df_stn_ts: dataset with throughput data for all the stations
    :param df_agg_ts: dataset with throughput data for the average of all the stations
    retun plot of network with the daily evolution of throughput for all the stations
    '''

    fig, ax = plt.subplots(figsize=(18,12))
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    ax.tick_params(axis = 'both', which = 'both', top = 'off', bottom='off', left='off', right='off', labelbottom=True, labelleft=True, width = 0, length =
                   0, color = 'k', labelsize = 26, pad=6)

    ax.set_xlim([-1, 95])
    ax.set_ylim([-3, 107])
    
    ax.set_xticks([i*10 for i in range(9)], [time_map[i*10+1] for i in range(9)], rotation=90)
    
    for i in range(9):
        ax.plot([i*10, i*10],[-7,107], color='gray', lw=0.7, zorder=0)
        
    ax.plot([-4,95], [0,0], color='gray', lw=0.7, zorder=0)
    
#     ax.axis('off')

    palette=mcp.gen_color(cmap='viridis', n=len(df_stn_ts))

#     # Plot daily throughput of each station with smoothing (interpolation, then rolling average)
#     t_smooth = np.linspace(1,86,86*10-9)
  
#     for i in range(len(df_stn_ts)):
#         y = np.array(df_stn_ts.iloc[i,1:87]/np.max(df_stn_ts.iloc[i,1:87])*100).astype('float64')
#         y_smooth = np.interp(t_smooth, t, y)
#         y_smooth = pd.Series(y_smooth)
#         y_smooth = y_smooth.rolling(5).mean()
#         plt.plot(t_smooth, y_smooth, color=palette[i], alpha=0.5, lw=0.8)

#     y = np.array(df_agg_ts.iloc[:,1]/np.max(df_agg_ts.iloc[:,1])*100).astype('float64')
#     y_smooth = np.interp(t_smooth, t, y)
#     y_smooth = pd.Series(y_smooth)
#     y_smooth = y_smooth.rolling(5).mean()
#     plt.plot(t_smooth, y_smooth, color='red', alpha=1, lw=4, zorder=len(df_stn_ts))

    df_stn_normalised = pd.DataFrame({column: [0 for i in range(len(df_stn_ts))] for column in df_stn_ts.columns[1:-1]})
    for i in range(len(df_stn_ts)):
        y = np.array(df_stn_ts.iloc[i,1:-1]/np.max(df_stn_ts.iloc[i,1:-1])*100).astype('float64')
        df_stn_normalised.iloc[i,:] = y


    df_stn_plot = pd.DataFrame({column: [0 for i in range(7)] for column in df_stn_normalised.columns})
    for j in range(len(df_stn_normalised.columns)):
        df_stn_plot.iloc[0, j] = np.median(df_stn_normalised.iloc[:, j])
    for j in range(len(df_stn_normalised.columns)):
        df_stn_plot.iloc[1, j] = np.quantile(df_stn_normalised.iloc[:, j], 0.025)
    for j in range(len(df_stn_normalised.columns)):
        df_stn_plot.iloc[2, j] = np.quantile(df_stn_normalised.iloc[:, j], 0.25)
    for j in range(len(df_stn_normalised.columns)):
        df_stn_plot.iloc[3, j] = np.quantile(df_stn_normalised.iloc[:, j], 0.75)
    for j in range(len(df_stn_normalised.columns)):
        df_stn_plot.iloc[4, j] = np.quantile(df_stn_normalised.iloc[:, j], 0.975)    
    for j in range(len(df_stn_normalised.columns)):
        df_stn_plot.iloc[5, j] = np.min(df_stn_normalised.iloc[:, j])
    for j in range(len(df_stn_normalised.columns)):
        df_stn_plot.iloc[6, j] = np.max(df_stn_normalised.iloc[:, j])

        
    palette=mcp.gen_color(cmap='viridis', n=4)
    
    plt.plot(np.linspace(1,86,86), df_stn_plot.iloc[0, :], palette[0], lw=4.5, zorder=3)
    t_smooth = np.linspace(1,86,86*10-9)
    y_q25 = np.array(df_stn_plot.iloc[2,:]).astype('float64')
    y_q25smooth = np.interp(t_smooth, np.linspace(1,86,86), y_q25)
    y_q25smooth = pd.Series(y_q25smooth)
    y_q75 = np.array(df_stn_plot.iloc[3,:]).astype('float64')
    y_q75smooth = np.interp(t_smooth, np.linspace(1,86,86), y_q75)
    y_q75smooth = pd.Series(y_q75smooth)
    ax.fill_between(t_smooth, y_q25smooth, y_q75smooth, color = palette[0], alpha=0.35)
    y_q025 = np.array(df_stn_plot.iloc[1,:]).astype('float64')
    y_q025smooth = np.interp(t_smooth, np.linspace(1,86,86), y_q025)
    y_q025smooth = pd.Series(y_q025smooth)
    y_q975 = np.array(df_stn_plot.iloc[4,:]).astype('float64')
    y_q975smooth = np.interp(t_smooth, np.linspace(1,86,86), y_q975)
    y_q975smooth = pd.Series(y_q975smooth)
    ax.fill_between(t_smooth, y_q025smooth, y_q975smooth, color = palette[0], alpha=0.2)
    plt.scatter(np.linspace(1,86,86), df_stn_plot.iloc[5, :], color=palette[0])
    plt.scatter(np.linspace(1,86,86), df_stn_plot.iloc[6, :], color=palette[1])

    ax.text(0.485, -0.20, r'Time of the day', ha='center', rotation=0, size = 28, transform=ax.transAxes)
    ax.text(-0.095, 0.5, 'Station throughput\n(relative to maximum value)', va='center', ha='center',
            rotation=90, size = 28, transform=ax.transAxes)

    # Add color bar
#     plt.viridis()
#     cbaxes = fig.add_axes([0.80, 0.56, 0.02, 0.28])
#     plt.scatter([0,0.25,0.5,0.75,1], [0,0.25,0.5,0.75,1], s=0, c=[0,0.25,0.5,0.75,1], cmap='viridis',
#                 alpha=0.9)
#     cbar = plt.colorbar(cax=cbaxes, ticks=[0,0.25,0.5,0.75,1], orientation='vertical')
#     cbaxes.tick_params(axis = 'both', which = 'both', width = 0, length = 0, color = 'k', labelsize = 25,
#                        pad=10)
#     cbaxes.set_yticklabels(['20','12500','25000','37500','50000'], rotation=0)
#     ax.text(0.85, 0.61, 'Daily throughput', transform=ax.transAxes, ha='center', rotation=90, color='k',
#             size = 25, zorder=11)


    h1 = mlines.Line2D([], [],
                              color=palette[0],
                              linestyle='-', linewidth=4.5,
                              marker = 's', markersize=0,
                              label='Median', alpha=1)
    h2 = mlines.Line2D([], [],
                            color=palette[0],
                            linestyle='-', linewidth=0,
                            marker='s', markersize=17,
                            label='75% CI', alpha=0.4)
    h3 = mlines.Line2D([], [],
                            color=palette[0],
                            linestyle='-', linewidth=0,
                            marker='s', markersize=17,
                            label='95% CI', alpha=0.2)
    h4 = mlines.Line2D([], [],
                            color=palette[0],
                            linestyle='-', linewidth=0,
                            marker='o', markersize=17,
                            label='Minimum', alpha=1)
    h5 = mlines.Line2D([], [],
                            color=palette[1],
                            linestyle='-', linewidth=0,
                            marker='o', markersize=17,
                            label='Maximum', alpha=1)

    ax.legend(handles=[h1,h2,h3,h4,h5], fontsize=20, title_fontsize=16, bbox_to_anchor=(0.85, 0.9))
    
    
    return fig





# Create function for plotting network
def plot_ntw(stn_gdf, line_gdf, plot_crs, measure):

    '''
    Return a plot of the network

    :param stn_gdf: gdf of stations
    :param line_gdf: gdf of lines
    :param plot_crs: crs for plotting
    :param measure: measure being plotted; 'gini'
    retun plot of network with value of gini index for each station
    '''

    fig, ax = plt.subplots(figsize=(18,15))

    # Plot lines
    line_gdf.plot(ax=ax, linewidth=0.7, color='k')

    # Plot gini
    title_label = 'Gini Index'
    leg_label = 'G'
    vmin = min(stn_gdf['gini'])
    vmax = max(stn_gdf['gini'])
    s_throughput = 4*stn_gdf['Daily throughput']**0.5
    
    # Plot stations
    stn_gdf.plot(column=measure, ax=ax,
                 cmap='viridis_r', vmin=vmin, vmax=vmax,
                 markersize=s_throughput, edgecolor='darkgrey', linewidth=0.5, alpha=0.85)
    
    # Plot cross for stations with non-significant value of Gini
    stn_gdf_nonsignificant = stn_gdf[stn_gdf['gini_pvalue'] > 0.05].reset_index(drop = True)
    stn_gdf_nonsignificant.plot(marker = 'x', markersize=190, color='red', linewidth=3, alpha=1, zorder=3, ax=ax)
    
    # Set axes limits
    xlim = ax.get_xlim()
    ax.set_xlim([xlim[0] - 0.02*(xlim[1]-xlim[0]), xlim[1] + 0.15*(xlim[1]-xlim[0])])
    
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0] - 0.09*(ylim[1]-ylim[0]), ylim[1] + 0.09*(ylim[1]-ylim[0])])
    
    # Add colorbar
    cax = fig.add_axes([0.805, 0.38, .017, .28])
    sm = plt.cm.ScalarMappable(cmap='viridis_r', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fake up the array of the scalar mappable
    sm._A = []
    fig.colorbar(sm, cax=cax)
    cax.tick_params(axis = 'both', which = 'both', labelbottom=False, labelright=True, width = 0, length = 0, color = 'k', labelsize = 22, pad=10)
    
    
    # Add base map
    cx.add_basemap(ax, crs=plot_crs, source=cx.providers.CartoDB.PositronNoLabels, alpha=0.7)
    
    # Add North arrow
    im = plt.imread('./north-arrow.png')
    newax = fig.add_axes([0.155,0.28,0.05,0.052], zorder=1)
    newax.tick_params(axis = 'both', which = 'both', labelbottom=False, labelleft=False, width = 0, length = 0)
    newax.set_facecolor('None')
    plt.setp(newax.spines.values(), linewidth=0)
    newax.imshow(im)


    # Add scale-bar
    scale_rect = matplotlib.patches.Rectangle((0.04,0.19), width=0.07, height=0.01,
                                              edgecolor='k', facecolor='k', transform=ax.transAxes)
    ax.add_patch(scale_rect)
    ax.text(0.075, 0.215, s='5 km', fontsize=22, horizontalalignment='center', transform=ax.transAxes)

    ax.tick_params(axis = 'both', which = 'both', labelbottom=False, labelleft=False, width = 0, length = 0, color = 'k', labelsize = 32, pad=10)
    
    ax.text(0.96, 0.84, 'Gini index', transform=ax.transAxes, ha='right', color='k', size = 23, zorder=11)

    ax.scatter([0.65],[0.135], color='red', marker='x', linewidth=3.5, s=300, transform=ax.transAxes)
    ax.text(0.96, 0.12, 'Gini index p-value > 0.05', transform=ax.transAxes, ha='right', color='k', size = 23, zorder=11)
 
    return fig
    
    
    
    
    
    
def plot_rmse(ft_rm_gini):
    
    fig, ax = plt.subplots(figsize=(18,12))
    
    ax.set_ylim([0.0765, 0.087])
    
    ax.tick_params(axis = 'both', which = 'both', labelbottom=True, labelleft=True, width = 0, length =
                   0, color = 'k', labelsize = 32, pad=10)
    
    ax.plot(ft_rm_gini.index.values, np.sqrt(ft_rm_gini['mse_mean']), marker='o', markersize=15, color='midnightblue')
    
    ax.text(0.485, -0.12, 'Number of features removed', ha='center', rotation=0, size = 32, transform=ax.transAxes)
    ax.text(-0.135, 0.5, 'Root of MSE with removal of features', va='center', ha='center',
            rotation=90, size = 32, transform=ax.transAxes)
    
    return fig




def plot_ss(n_cl_list, ss_list):
    
    fig, ax = plt.subplots(figsize=(18,12))
    
#     ax.set_ylim([0.0765, 0.087])
    
    ax.tick_params(axis = 'both', which = 'both', labelbottom=True, labelleft=True, width = 0, length =
                   0, color = 'k', labelsize = 32, pad=10)
    
    ax.plot(n_cl_list, ss_list, marker='o', markersize=15, color='midnightblue')
    
    ax.text(0.485, -0.12, 'Number of clusters in k-means algorithm', ha='center', rotation=0, size = 32, transform=ax.transAxes)
    ax.text(-0.10, 0.5, 'Silhouette score', va='center', ha='center',
            rotation=90, size = 32, transform=ax.transAxes)
    
    return fig




# Create function to plot radar chart of cluster centroid
def plot_radar_centroid(cl_centroid, average_centroid, colors):

    '''
    Plot radar chart of cluster centroid

    :param cl_centroid: df of cluster centroids
    :return the radar chart of cluster centroid
    '''

    # Number of variables
    var = ['Population\n density', 'Jobs\n density', 'Retail', 'Institutional\n communal\n accomodation',
       'Betweenness\n centrality']
    n_var = len(var)

    # Angle of axis
    angles = [n / float(n_var) * 2 * pi for n in range(n_var)]
    angles += angles[:1]

    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(15,15))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw ylabels
    y_min = floor(min(cl_centroid.min()))
    y_max = ceil(max(cl_centroid.max()))
    plt.ylim(y_min, y_max)

    # Plot average
    values = list(average_centroid)
    values += values[:1]
    ax.plot(angles, values, linewidth=4, color='dimgrey',
            linestyle=':', label="Average")
    
    # Plot cluster centroid
    
    for cl in range(4):
        values = cl_centroid.iloc[cl].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values,  
                linewidth=5, marker="o", markersize=15, color = colors[cl], label=f"Cluster {cl+1}")
        
    # Draw one axe per variable + add labels
    ax.tick_params(labelsize=24)
    ax.set_rticks([1, 2, 3])
    ax.set_rlabel_position(250)
    ax.set_xticks(angles[:-1], var)
    

    labels = []
    for label, angle in zip(ax.get_xticklabels(), angles):
        x,y = label.get_position()
        lab = ax.text(x,y, ' ', transform=label.get_transform(),
                  ha=label.get_ha(), va=label.get_va())
        lab.set_rotation(90-angle/(2*pi)*360)
        labels.append(lab)
    ax.set_xticklabels([])


    # Styling
    plt.legend(bbox_to_anchor=(1.2, 0.6), fontsize=35)

    return fig







# Create function to plot clusters average journey profile
def plot_avg_demand_profile(df_throughput, df_entry, df_exit, cl_lab, colors, time_map):

    '''
    Plot the average demand profile of stations in cluster

    :param df_throughput: df of throughput
    :param df_entry: df of entry
    :param df_exit: df of entry
    :param cl_lab: df of stations and cluster labels
    :return the plot of average journey profile of throughout, entries and exits
    '''

    fig, ax = plt.subplots(3, 1, figsize=(12, 30),
                           sharex='col', sharey='row')
    
    fig.subplots_adjust(hspace=0.5)
    
    entex_label = {0: 'throughput',
                   1: 'entry',
                   2: 'exit'}

    for cl in range(4):
        # Get stations in cluster
        stn = cl_lab.loc[cl_lab['Cluster'] == cl+1, 'StationName']

        for entex_ax in range(3):

            # Get data for plotting
            plot_data = locals()['df_' + entex_label[entex_ax]]
            plot_data = plot_data.loc[stn, :]

#             # Plot station data
#             for j in range(len(plot_data)):
#                 ax[entex_ax].plot([i for i in range(len(plot_data.columns))], plot_data.iloc[j,:]/max(plot_data.iloc[j,:])*100, c="gray", alpha=0.2);

            ax[entex_ax].tick_params(axis = 'both', which = 'both', labelbottom=True, labelleft=True, width = 0, length =
                   0, color = 'k', labelsize = 32, pad=10)
    
            ax[entex_ax].plot([i for i in range(len(plot_data.columns))], plot_data.mean()/max(plot_data.mean())*100,
                              c=colors[cl], linewidth=4)

            # Styling
            ax[entex_ax].tick_params(axis='x', rotation=315)
            ax[entex_ax].set_xticks([i*10 for i in range(9)], [time_map[i*10+1] for i in range(9)], rotation='45')
    
            ax[entex_ax].set_xlabel('Time', size=30, labelpad=10)

            ax[entex_ax].set_ylabel('Proportion', size=30, labelpad=10)


    return fig









# Create function to plot location of stations in cluster
def plot_cluster_loc(inner_gdf, caz_gdf,
                line_gdf, station_gdf, colors,
                cl_lab):

    '''
    Plot the locations of stations in cluster

    :param inner_gdf: gdf of Inner London
    :param caz_gdf: gdf of CAZ
    :param line_gdf: gdf of underground lines
    :param station_gdf: gdf of stations
    :param col_pal: color palette
    :param cl_lab: cluster labels
    :param cl: cluster to plot
    :return the plot of location of stations in cluster
    '''

    fig, ax = plt.subplots(figsize=(18, 15))
    
    
    # Plot outlines of London, Inner London and CAZ
    inner_gdf.plot(ax=ax, facecolor='saddlebrown', edgecolor='saddlebrown',
                   lw=.5, ls='-', alpha=.7, zorder=1)
#     caz_gdf.plot(ax=ax, facecolor='maroon', edgecolor='maroon',
#                  lw=.5, ls='-', alpha=.7, zorder=1)
    
    inner_leg = mlines.Line2D([], [],
                              color='saddlebrown',
                              linestyle='-', linewidth=0,
                              marker = 's', markersize=17,
                              label='Inner London', alpha=0.6)
#     caz_leg = mlines.Line2D([], [],
#                             color='maroon',
#                             linestyle='-', linewidth=.5,
#                             label='CAZ', alpha=0.6)

    cl_1 = mlines.Line2D([], [],
                            color=colors[0],
                            linestyle='-', linewidth=0,
                            marker='o', markersize=17,
                            label='Cluster 1', alpha=0.85)
    cl_2 = mlines.Line2D([], [],
                            color=colors[1],
                            linestyle='-', linewidth=0,
                            marker='o', markersize=17,
                            label='Cluster 2', alpha=0.85)
    cl_3 = mlines.Line2D([], [],
                            color=colors[2],
                            linestyle='-', linewidth=0,
                            marker='o', markersize=17,
                            label='Cluster 3', alpha=0.85)
    cl_4 = mlines.Line2D([], [],
                            color=colors[3],
                            linestyle='-', linewidth=0,
                            marker='o', markersize=17,
                            label='Cluster 4', alpha=0.85)

    ax.legend(handles=[inner_leg, cl_1, cl_2, cl_3, cl_4], fontsize=20, title_fontsize=16)

    for cl in range(4):
        
        # Get stations in cluster
        stn = cl_lab.loc[cl_lab['Cluster'] == cl+1, 'StationName']
        # Plot underground lines
        line_gdf.plot(ax=ax,  linewidth=0.7, color='k')

        # Plot stations
        plot_data = station_gdf[station_gdf['StationName'].isin(stn)]
        s_throughput = 4*plot_data['Daily throughput']**0.5
        plot_data.plot(ax=ax, markersize=s_throughput, 
                       facecolor=colors[cl], edgecolor='darkgrey', linewidth=0.5, alpha=0.85)
#         plot_data.plot(ax=ax, markersize=s_throughput, 
#                        facecolor='None', edgecolor=colors[cl],)

#         data_null = stations_gdf[~station_gdf['StationName'].isin(stn)]
#         data_null.plot(ax=ax, markersize=50, 
#                        facecolor='white', edgecolor='black')

    
    # Set axes limits
    xlim = ax.get_xlim()
    ax.set_xlim([xlim[0] - 0.02*(xlim[1]-xlim[0]), xlim[1] + 0.15*(xlim[1]-xlim[0])])
    
    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0] - 0.09*(ylim[1]-ylim[0]), ylim[1] + 0.09*(ylim[1]-ylim[0])])
    

    # Add north arrow, https://stackoverflow.com/a/58110049/604456
    im = plt.imread('./north-arrow.png')
    newax = fig.add_axes([0.155,0.28,0.05,0.052], zorder=1)
    newax.tick_params(axis = 'both', which = 'both', labelbottom=False, labelleft=False, width = 0, length = 0)
    newax.set_facecolor('None')
    plt.setp(newax.spines.values(), linewidth=0)
    newax.imshow(im)

    # Add scale-bar
    scale_rect = matplotlib.patches.Rectangle((0.04,0.19), width=0.07, height=0.01,
                                              edgecolor='k', facecolor='k', transform=ax.transAxes)
    ax.add_patch(scale_rect)
    ax.text(0.075, 0.215, s='5 km', fontsize=22, horizontalalignment='center', transform=ax.transAxes)

    # Add base map
    cx.add_basemap(ax, crs=station_gdf.crs, source=cx.providers.CartoDB.PositronNoLabels, alpha=0.7)

    ax.set_axis_off()

    return fig




def plot_gini_clusters(df, cluster_df, colors):

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.tick_params(axis = 'both', which = 'both', labelbottom=True, labelleft=True, width = 0, length =
                   0, color = 'k', labelsize = 32, pad=10)
    
    ax.set_ylim([0.35,0.95])

    # Get values of measure and merge cluster label
    plot_data = pd.merge(df, cluster_df, on='StationName')
    
      
    # Plot data

    sns.boxplot(data=plot_data, x='Cluster', y='gini',
                        width=0.25, linewidth=2, boxprops={"facecolor": 'None'}, ax=ax)
    
    s = [4*np.sum(plot_data.iloc[i, 1:87])**0.5 for i in range(len(plot_data))]
    plot_data['size'] = s
    
    for cl in range(4):
        cluster_data = plot_data[plot_data['Cluster']==cl+1].reset_index(drop=True)
        
        width = 0.3
        max_dev = max([abs(x-np.median(cluster_data['gini'])) for x in cluster_data['gini']])
        random_pos = [np.random.uniform(cl-width/2+width/2*abs(x-np.median(cluster_data['gini']))/max_dev, 
                      cl+width/2-width/2*abs(x-np.median(cluster_data['gini']))/max_dev, (1,1))[0][0] 
                      for x in cluster_data['gini']]
        
        ax.scatter(random_pos, cluster_data['gini'],
                   facecolor = colors[cl]+(0.5,), edgecolor= colors[cl]+(1,), s = cluster_data['size']
                   )
        
    ax.set_xlabel('Cluster', fontsize=32, labelpad=12)
    # ax.set_xticklabels(cl_name.values(), size=16)
    ax.set_ylabel("Gini index", fontsize=32, labelpad=12)
    

    return fig
