import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from rfpimp import *
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

# Draw buffer around stations

def draw_buffer(stations_gdf):
    stations_buffer = stations_gdf.copy()
    stations_buffer['buffer'] = stations_gdf.geometry.buffer(1000)
    stations_buffer.set_geometry(col='buffer', drop=True, inplace=True)

    buffer_area = stations_buffer.geometry.area.values[0]/1000000
    
    return [stations_buffer, buffer_area]




# Attach landuse data

def attach_landuse_data(la_gdf, stations_buffer, stations_gdf):
    
    # Land use data
    # https://www.gov.uk/government/statistical-data-sets/live-tables-on-land-use
    # Accessed: 21 July 2022
    land_use = pd.read_excel('data/features/land_use_2018.xlsx',
                             sheet_name='Processed')

    # Merge land use data to gdf
    la_land_gdf = la_gdf.merge(land_use, left_on='lad18nm', right_on='Local Authority Name')

    # Find intersection between buffer and land use gdf
    intersect_data = gpd.overlay(stations_buffer, la_land_gdf)

    # Get land use categories
    land_use_cat = land_use.columns[2:]

    stn_land_list = []

    # Find percentage of land use surrounding each stations
    for stn in stations_gdf['StationName'].tolist():

        # Get station's land use data
        data_temp = intersect_data[intersect_data['StationName'] == stn]

        # Compute area of intersections
        data_temp['area'] = data_temp.geometry.area

        # Convert area into ratio
        data_temp['area'] = data_temp['area']/sum(data_temp['area'])

        # Collect land use area and convert to percentage
        land_use_list = [sum(data_temp['area']*data_temp[i]) for i in land_use_cat]
        tot_area = sum(land_use_list)
        land_use_pct = [j/tot_area for j in land_use_list]
        stn_land_list.append(land_use_pct)

    # Create dataframe to hold value of explanatory variables
    exp_var = pd.DataFrame(stn_land_list,
                           columns=land_use_cat,
                           index=stations_gdf['StationName'].tolist())

    # Keep only relevant land use
    exp_var = exp_var[['Community Buildings', 'Leisure and recreational buildings',
                       'Industry', 'Offices', 'Retail', 'Storage and Warehousing',
                       'Institutional and Communal Accommodation', 'Residential']]

    # Rename land use categories
    exp_var.rename({'Community Buildings': 'community_bldg',
                    'Leisure and recreational buildings': 'leisure_recreational_bldg',
                    'Industry': 'industry',
                    'Offices': 'offices',
                    'Retail': 'retail', 
                    'Storage and Warehousing': 'storage_and_warehousing',
                    'Institutional and Communal Accommodation': 'instit_communal_accom',
                    'Residential': 'residential'},
                   axis=1, inplace=True)
    
    return exp_var 



# Attach location data

def attach_location_data(la_gdf, stations_gdf, exp_var):

    inner_london = ['Camden', 'Hackney', 'Hammersmith and Fulham',
                    'Islington', 'Kensington and Chelsea', 'Lambeth', 
                    'Lewisham', 'Southwark', 'Tower Hamlets', 'Wandsworth',
                    'Westminster', 'City of London', 'Haringey', 'Newham']

    inner_gdf = la_gdf[la_gdf['lad18nm'].isin(inner_london)]
    inner_gdf = gpd.GeoDataFrame(gpd.GeoSeries(data=inner_gdf.unary_union, crs='epsg:27700')).\
        rename(columns={0: 'inner'})
    inner_gdf = inner_gdf.set_geometry('inner')

    # Stations in Inner London
    inner_stn = gpd.sjoin(stations_gdf, inner_gdf,
                          op='intersects', how='inner')

    # Merge to explanatory variables
    exp_var['inner'] = 0
    exp_var.loc[exp_var.index.isin(inner_stn['StationName']), 'inner'] = 1

    # Central Activity Zone
    # https://data.london.gov.uk/dataset/central_activities_zone
    # Accessed: 21 July 2022

    caz_gdf = gpd.read_file('data/features/central_activities_zone.gpkg')
    caz_gdf = gpd.GeoDataFrame(gpd.GeoSeries(data=caz_gdf.unary_union, crs='epsg:27700')).\
        rename(columns={0: 'caz'})
    caz_gdf = caz_gdf.set_geometry('caz')

    # Stations in CAZ
    caz_stn = gpd.sjoin(stations_gdf, caz_gdf,
                        op='intersects', how='inner')

    # Merge to explanatory variables
    exp_var['caz'] = 0
    exp_var.loc[exp_var.index.isin(caz_stn['StationName']), 'caz'] = 1
    
    return exp_var 




# Attach population and job density data

def attach_pop_job_data(lsoa_gdf, stations_buffer, stations_gdf, exp_var):
               
    # https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareapopulationdensity
    # Accessed: 21 July 2022
    pop_density = pd.read_excel('data/features/lsoa_pop_density_mid2020.xlsx',
                                sheet_name='Mid-2020 Population Density', skiprows=4)

    # Merge population density to gdf
    lsoa_pop_gdf = lsoa_gdf.merge(pop_density[['LSOA Code', 'LSOA Name', 'People per Sq Km']],
                                  left_on='LSOA11CD', right_on='LSOA Code')

    # Find intersection between buffer and land use gdf
    intersect_data = gpd.overlay(stations_buffer, lsoa_pop_gdf)

    stn_pop_list = []

    # Find population density surrounding each stations
    for stn in stations_gdf['StationName'].tolist():

        # Get station's land use data
        data_temp = intersect_data[intersect_data['StationName'] == stn]

        # Compute area of intersections
        data_temp['area'] = data_temp.geometry.area

        # Convert area into ratio
        data_temp['area'] = data_temp['area']/sum(data_temp['area'])

        # Compute average population density
        stn_pop_list.append(sum(data_temp['area']/sum(data_temp['area'])*\
                                data_temp['People per Sq Km']))

    # Merge to explanatory variables
    exp_var['pop_density'] = stn_pop_list
    
    # https://www.nomisweb.co.uk/query/construct/summary.asp?mode=construct&dataset=189&version=0
    # Accessed: 21 July 2022
    job_density = pd.read_csv('data/features/lsoa_jobs_count_2020.csv')

    # Total number of jobs in each LSOA
    job_density['total']=job_density.iloc[:, 2:].sum(axis=1)
    job_cnt = job_density.groupby('mnemonic')['total'].sum()

    # Merge jobs density to gdf
    lsoa_jobs_gdf = lsoa_gdf.merge(job_cnt,
                                   left_on='LSOA11CD', right_on='mnemonic')

    # Convert to jobs per unit area
    lsoa_jobs_gdf['jobs_density'] = lsoa_jobs_gdf['total']/\
        (lsoa_jobs_gdf.geometry.area/1000000)

    # Find intersection between buffer and jobs density gdf
    intersect_data = gpd.overlay(stations_buffer, lsoa_jobs_gdf)

    stn_job_list = []

    # Find jobs density surrounding each stations
    for stn in stations_gdf['StationName'].tolist():

        # Get station's land use data
        data_temp = intersect_data[intersect_data['StationName'] == stn]

        # Compute area of intersections
        data_temp['area'] = data_temp.geometry.area

        # Convert area into ratio
        data_temp['area'] = data_temp['area']/sum(data_temp['area'])

        # Compute average population density
        stn_job_list.append(sum(data_temp['area']/sum(data_temp['area'])*\
                                data_temp['jobs_density']))

    # Merge to explanatory variables
    exp_var['jobs_density'] = stn_job_list
    
    return exp_var




# Attach public transport access points data

def attach_naptan_data(bng_crs, stations_gdf, stations_buffer, buffer_area, exp_var):

    # https://www.data.gov.uk/dataset/ff93ffc1-6656-47d8-9155-85ea0b8f2251/national-public-transport-access-nodes-naptan
    # Accessed: 21 July 2021
    naptan = pd.read_csv('data/features/naptan.csv')

    # Create gdf
    naptan_gdf =\
        gpd.GeoDataFrame(naptan,
                         geometry=[Point(x, y) for x, y
                                   in zip(naptan.Longitude, naptan.Latitude)],
                         crs='EPSG:4326')
    naptan_gdf.to_crs(bng_crs, inplace=True)

    stn_naptan_list = []

    # Find number of busstops surrounding each stations
    for stn in stations_gdf['StationName'].tolist():

        # Get station's buffer
        data_temp = stations_buffer[stations_buffer['StationName'] == stn]

        # Find stations within buffer
        data_temp = gpd.sjoin(naptan_gdf, data_temp,
                              op='intersects', how='inner')

        # Count number of stations
        stn_naptan_list.append(len(data_temp)/buffer_area)

    # Merge to explanatory variables
    exp_var['naptan_density'] = stn_naptan_list
    
    return exp_var




# Attach network structure data

def attach_network_data(exp_var):

    G = nx.read_graphml("outputs/stn_data/geometry/network_graph")

    # Degree centrality
    deg_cent = pd.DataFrame.from_dict(nx.degree_centrality(G), orient='index').\
        rename({0: 'deg_centrality'}, axis=1)

    exp_var = exp_var.merge(deg_cent, left_index=True, right_index=True)

    # Betweenness centrality
    btwn_cent = pd.DataFrame.from_dict(nx.betweenness_centrality(G), orient='index').\
        rename({0: 'btwn_centrality'},axis=1)

    exp_var = exp_var.merge(btwn_cent, left_index=True, right_index=True)
    
    return exp_var




# Function to use RF to select variables

def rf_select(df_exp_var, df_dep_var, rf_model, n_run):

    '''
    Use Random Forest to select variables by removing features with 
    least importance sequentially

    :param df_exp_var: df of explanatory variables
    :param df_dep_var: df of dependent variable
    :param rf_model: random forest model
    :param n_run: number of runs for each variable
    :return ft_impt: df of importance and rank for each run
    :return perf: df of R2 score and mse for each run
    :return ft_rm: df of mean R2 and variables removed
    '''

    # Copy df
    dep_var = df_dep_var.copy()
    exp_var_rm = df_exp_var.copy()

    # Create lists to hold values
    n_features_rm_list = []
    run_list = []
    feature_list = []
    imp_list = []
    rank_list = []
    r2_list = []
    mse_list = []
    feature_rm_list = []
    r2_mean = []
    mse_mean = []

    # Number of features
    n_feature = len(exp_var_rm.columns)

    for n_remove in range(n_feature-1):
        print(f"{n_remove} variables removed")
        
        # Note that all the results reported in the dissertation have been uploaded on the output folders.
        # This 'continue' command is included here to bypass the RF modelling,
        # so that subsequent codes can reproduce the results reported in the dissertation.
        # It can be removed if the interest is in re-using the code for other purposes.
        # continue

        # Create list to hold values for current number of features
        n_features_rm_curr = []
        run_curr = []
        feature_curr = []
        imp_curr = []
        rank_curr = []
        r2_curr = []
        mse_curr = []

        # Iterate training
        for run in range(n_run):

            # Split training and testing dataset
            train_x, test_x, train_y, test_y =\
                train_test_split(exp_var_rm, dep_var)

            # Fit random forest
            rf_model.fit(train_x, train_y);

            # Get importance of features and R2 value
            imp = dropcol_importances(rf_model, train_x, train_y)
            n_features_rm_curr = n_features_rm_curr + [n_remove]*len(imp.index)
            run_curr = run_curr + [run]*len(imp.index)
            feature_curr = feature_curr + list(imp.index)
            imp_curr = imp_curr + list(imp['Importance'])
            rank_curr = rank_curr + list(range(1, len(imp.index)+1))
            r2_curr.append(rf_model.score(X=test_x, y=test_y))
            mse_curr.append(mean_squared_error(test_y, rf_model.predict(test_x)))

        # Append lists to store
        n_features_rm_list = n_features_rm_list + n_features_rm_curr
        run_list = run_list + run_curr
        feature_list = feature_list + feature_curr
        imp_list = imp_list + imp_curr
        rank_list = rank_list + rank_curr
        r2_list = r2_list + r2_curr
        mse_list = mse_list + mse_curr

        # Create dataframe of importance rank
        imp_df = pd.DataFrame(zip(feature_curr, rank_curr),
                              columns=['Feature', 'Rank'])

        # Sort by importance rank of feature
        imp_avg = imp_df.groupby('Feature')\
            .agg(mode=('Rank', lambda x: list(pd.Series.mode(x))[0]),
                 median=('Rank', 'median'),
                 mean=('Rank', 'mean'))
        imp_avg.sort_values(by=['mode', 'median', 'mean'],
                            ascending=True, inplace=True)

        # Get least important feature and drop it
        feature_rm = imp_avg.index[-1]
        feature_rm_list.append(feature_rm)
        exp_var_rm.drop(columns=feature_rm, inplace=True)

        # Compute mean R2 and append
        r2_mean.append(np.mean(r2_curr))
        mse_mean.append(np.mean(mse_curr))

    # Create dataframe and store results
    ft_impt = pd.DataFrame(zip(n_features_rm_list, run_list,
                               feature_list, imp_list, rank_list),
                           columns=['num_features_rm', 'run_num',
                                    'feature', 'importance', 'rank'])

    r2_idx = []
    for i in range(n_feature-1):
        for j in range(n_run):
            r2_idx.append(i)

    perf = pd.DataFrame(zip(r2_idx, r2_list, mse_list),
                        columns=['num_features_rm', 'r2', 'mse'])

    ft_rm = pd.DataFrame(zip(r2_mean, mse_mean, feature_rm_list),
                         columns=['r2_mean', 'mse_mean', 'feature_rm'])

    print('Done!')

    return ft_impt, perf, ft_rm




def initialize_rf(n_runs, n_trees):
    
    # Initialize hyperparameters to tune
    # Ref: https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1301
    # Max features
    list_max_features = ['sqrt', 'log2', 'auto', 1/3]
    # Max sample
    list_max_samples = [0.2, 0.4, 0.6, 0.8, None]
    # Min samples in each leaf
    list_min_samples_leaf = [1, 5, 10]
    grid_hyperparameters = {'max_features': list_max_features,
                            'max_samples': list_max_samples,
                            'min_samples_leaf': list_min_samples_leaf}

    # Initialize RF regressor and GridSearchCV
    rf = RandomForestRegressor()
    cv_rf = model_selection.GridSearchCV(rf, grid_hyperparameters,
                                         scoring='neg_mean_squared_error',
                                         verbose=1, n_jobs=-1)

    
    
    return [cv_rf, n_trees]




def train_and_tune(exp_var, stations_gdf, n_runs, n_trees):

    [cv_rf, n_trees] = initialize_rf(n_runs, n_trees)
    
    # Split training and testing dataset for tuning hyperparameter
    train_x, test_x, train_y, test_y =\
        train_test_split(exp_var, stations_gdf['gini'])


    # Run gridsearchCV and get best hyperparameters
    cv_rf.fit(train_x, train_y)
    cv_gini = cv_rf.best_params_
    cv_gini_df = pd.DataFrame.from_dict(cv_gini, orient='index').\
        rename({0: 'best_param'}, axis=1)

    
    return [cv_gini, cv_gini_df]

