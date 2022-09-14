# import libraries 
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import geopandas as gpd
import json
import seaborn as sns
from IPython.display import HTML

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.base import clone
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
from pyclustering.cluster.kmedians import kmedians
from scipy.spatial.distance import cityblock

def fig_caption(title, fig_num):
    """Print figure caption on jupyter notebook"""
    
    display(HTML(f"""<p style="font-size:12px;font-style:default;"><b>
                     Figure {fig_num}. {title}.</b></p>"""))
     

def table_caption(title, table_num):
    """Print table caption on jupyter notebook"""
    
    display(HTML(f"""<center style="font-size:12px;font-style:default;"><b>
                     Table {table_num}. {title}.</b></center>""")
           )

    
def clean_data(df):
    """ Clean the listings by removing unnecessary columns. Return filtered
    dataframe that includes only the five major property types.
    
    Parameter
    --------
    df : pandas DataFrame
        This contains AirBnB listings from Melbourne in Victoria, Australia.
    """
    
    # convert price to float
    df['price'] = (df.price.astype('str')
                   .apply(lambda x:re.sub(r'[$,]', '', x)).astype('float'))

    # clean column of bathroom
    df['bathrooms_text'].replace('Half-bath', '1 private bath', 
                                 inplace=True)
    df['bathrooms_text'].replace('Private half-bath', '1 private bath', 
                                 inplace=True)
    df['bathrooms_text'].replace('Shared half-bath', '1 shared bath', 
                                 inplace=True)
    df['bathrooms_text'].replace('Private half-bath', '1 private bath', 
                                 inplace=True)
    df['bathrooms_text'].replace('1 private bath', '1 private bath', 
                                 inplace=True)
    df['bathrooms_text'].replace('1 bath', '1 private bath', 
                                 inplace=True)
    df['bathrooms_text'].replace('1 shared bath', '1 shared bath', 
                                 inplace=True)
    df['bathrooms_text'].replace('1 baths', '1 private bath', 
                                 inplace=True)
    df['bathrooms_text'].replace('1 shared baths', '1 shared bath', 
                                 inplace=True)
    df[['number_of_baths', 'bath_type']] = (df['bathrooms_text']
                                            .str
                                            .extract(r'(\d+\.?\d*)\s+(.*)'))
    df['number_of_baths'] = df['number_of_baths'].astype('float')
    df = df.drop('bathrooms_text', axis=1)
    
    # columns to drop
    drop_object_cols = ['listing_url', 'last_scraped', 'name', 
                        'description', 'neighborhood_overview', 
                        'picture_url', 'host_url', 'host_name',
                        'host_since', 'host_location', 'host_about', 
                        'host_response_time', 'host_response_rate', 
                        'host_acceptance_rate', 'host_thumbnail_url', 
                        'host_picture_url', 'host_neighbourhood', 
                        'host_has_profile_pic', 'host_identity_verified',
                        'neighbourhood', 'calendar_last_scraped',
                        'first_review', 'last_review', 'instant_bookable']

    drop_float_cols = ['neighbourhood_group_cleansed', 
                       'host_total_listings_count', 
                       'minimum_nights_avg_ntm', 
                       'maximum_nights_avg_ntm',
                      'latitude', 'longitude']
    
    drop_int_cols = ['scrape_id', 'host_id', 'minimum_minimum_nights',
                 'maximum_minimum_nights', 'minimum_maximum_nights', 
                 'maximum_maximum_nights', 'number_of_reviews_l30d']
    
    drop_cols = drop_object_cols + drop_float_cols + drop_int_cols
    
    # drop unnecessary columns
    df = df.drop(columns=drop_cols)
    
    # make values in property types consistent
    replacement = {'.*apartment':'apartment',
              '.*\shouse': 'house',
              '.*\stownhouse': 'townhouse',
              '.*\scondominium': 'condominium',
              '.*\sguesthouse': 'guesthouse',
               }
    
    for k in replacement.keys():
        df['property_type'] = df['property_type'].replace(k, replacement[k], 
                                                          regex=True)
        
    # filter only the 5 main property types
    cat_property = ['apartment', 'house', 'townhouse', 
                    'condominium', 'guesthouse']
    df_airbnb = df[df['property_type'].isin(cat_property)]
    
    return df_airbnb


def clean_amenities(df):
    
    df['amenities'] = (df['amenities'].str.replace('[','')
                       .str.replace(']','').str.replace('"','')
                       .str.strip().str.lower())
    
    amenities_df = pd.DataFrame(df['amenities'].str.split(',', expand=True))
    
    replacement = {'.*parking.*':'parking',
              'free\s': '',
              'and\s': '',
              '.*hdtv.*': 'hdtv',
              '.*oven.*': 'oven',
               '.*dedicated workspace.*': 'dedicated workspace',
              '.*gas stove.*': 'gas stove',
              '.*body wash.*': 'body wash',
              '.*body soap.*': 'body soap',
              'dryer.*': 'dryer',
              '.*crib.*': 'crib',
              '.*air conditioning.*': 'airconditioning',
              'wifi.*': 'wifi',
              '.*books.*toys.*': r"children's books and toys",
              '.*\stv.*': 'tv',
              '.*conditioner': 'conditioner',
              '.*shampoo': 'shampoo',
              '.*conditioner': 'conditioner',
              '.*sound system.*': 'sound system',
              '.*electric stove': 'electric stove',
              'clothing storage.*': 'clothing storage',
              'game console.*': 'game console',
              '.*refrigerator': 'refrigerator',
              '.*induction stove': 'induction stove',
              'carport.*': 'carport',
              'private\s.*garden.*':'private garden',
              'shared\sindoor.*pool':'shared indoor pool',
              'shared\soutdoor.*pool':'shared outdoor pool',
              'residential\sgarage.*': 'residential garage',
              'private\soutdoor.*pool': 'private outdoor pool',
              '.*\swasher.*': 'washer'}
    
    for k in replacement.keys():
        amenities_df = amenities_df.replace(k, replacement[k], regex=True)
    
    merged_df = df.merge(amenities_df, how='left',
                         left_index=True, right_index=True)
    
    merged_df = merged_df.drop('amenities', axis=1)
    
    return amenities_df, merged_df


def clean_verifications(df):
    
    df['host_verifications'] = (df['host_verifications'].str.replace('[','')
                                .str.replace(']','').str.replace("'","")
                                .str.strip().str.lower())
    
    verifications = (pd.DataFrame(df['host_verifications']
                                  .str.split(',', expand=True)))
    
    
    merged_df = df.merge(verifications, how='left',
                         left_index=True, right_index=True)
    
    merged_df = merged_df.drop('host_verifications', axis=1)
    
    return verifications, merged_df


def missing(df):
    """ Return the stat of missing values per columnn."""
    
    missing = pd.DataFrame(df.isnull().sum())
    missing.columns = ['number of missing values']
    missing['% missing'] = (np.round(100*(missing['number of missing values']
                                          /df.shape[0])))

    return missing.sort_values(by='number of missing values', 
                                  ascending=False)


def drop_missing(df, missing_df):
    """ Return a pandas DataFrame with less than 25% missing values.
    
    Parameters
    ----------
    df : pandas DataFrame
        This is where columns with missing values will be dropped.
    missing_df : pandas DataFrame
        This contains data of number of missing values and % missing.
    """
    
    cols_to_drop = list(missing_df[missing_df['% missing'] >= 25].index)
    
    clean_df = df.drop(cols_to_drop, axis='columns')
    
    return clean_df


def impute(df, missing_df):
    """ Return a pandas DataFrame with imputed missing data. Categorical 
    data are replaced with the mode while floats or integers are replaced 
    with the median value.
    
    Parameters
    ----------
    df : pandas DataFrame
        This is where columns with missing values will be dropped.
    missing_df : pandas DataFrame
        This contains data of number of missing values and % missing.
    """
    
    cols = list(missing_df[missing_df['number of missing values'] > 0].index)
    
    for col in cols:
        if df[col].dtype == 'object':
            df.loc[:, col] = (SimpleImputer(strategy='most_frequent')
                              .fit_transform(df[[col]]))
            
        elif df[col].dtype == 'float64' or df[col].dtype == 'int64':
            df.loc[:, col] = (SimpleImputer(strategy='median')
                              .fit_transform(df[[col]]))
        
    return df


def plot_neighborhood_count(df):
    '''Plots a bar chart of the number of listings of each neighbourhood.'''
    municipality = pd.read_csv('melbourne_municipalities.csv')
    
    df_merge = pd.merge(df, municipality, on='neighbourhood_cleansed')

    count_neighborhood = (pd.DataFrame(df_merge['neighbourhood_cleansed']
                                       .value_counts()))

    count_neighborhood = (count_neighborhood.reset_index()
                          .rename(columns={'index':'neighborhood',
                                           'neighbourhood_cleansed':'count'})
                         .sort_values(by='count', ascending=False))

    top_10 = count_neighborhood.head(10)

    fig = px.bar(top_10.sort_values(by='count', ascending=True), 
                 x='count', y='neighborhood',
                 color=['others']*(len(top_10)-1)+['Melbourne'],
                 color_discrete_sequence=['#BDBDBD','#FF5A5F'])

    fig.update_xaxes(title_text='Count')
    fig.update_yaxes(title_text='Neighborhood')
    fig.update_layout(title={
        'text': ('City of Melbourne has the most number of listings'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        showlegend=False)
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig.show()
    
    
def plot_property_count(df):
    '''Plots a bar chart of the number of listings per property count.'''
    count_property = (pd.DataFrame(df['property_type']
                                       .value_counts()))

    count_property = (count_property.reset_index()
                          .rename(columns={'index':'property type',
                                           'property_type':'count'}))

    fig = px.bar(count_property.sort_values(by='count', ascending=True), 
                 x='count', y='property type',
                 color=['others']*(len(count_property)-1)+['apartment'],
                 color_discrete_sequence=['#BDBDBD','#FF5A5F'])

    fig.update_xaxes(title_text='Count')
    fig.update_yaxes(title_text='Property Type')
    fig.update_layout(title={
        'text': ('Most of the listings in Melbourne are Apartments'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        showlegend=False)
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig.show()
    
def plot_room_count(df):
    '''Plots a bar graph of the number of listings per room type'''
    count_room = (pd.DataFrame(df['room_type']
                                       .value_counts()))
    count_room = (count_room.reset_index()
                          .rename(columns={'index':'room type',
                                           'room_type':'count'}))

    fig = px.bar(count_room.sort_values(by='count', ascending=True), 
                 x='count', y='room type',
                 color=['others']*(len(count_room)-1)+['apartment'],
                 color_discrete_sequence=['#BDBDBD','#FF5A5F'])

    fig.update_xaxes(title_text='Count')
    fig.update_yaxes(title_text='Room Type')
    fig.update_layout(title={
        'text': ('Most listings in Melbourne are Entire Homes/Apartments'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        showlegend=False)
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig.show()

def plot_room_price(df):
    '''Plots a box plot of the average price per room type.'''
    price_room = pd.DataFrame(df[['room_type', 'price']])
    
    entire = (price_room[price_room['room_type']=='Entire home/apt']
              .sort_values(by='price', ascending=True))
    entire = pd.DataFrame(entire.price.value_counts())
    entire = (entire.reset_index()
                          .rename(columns={'index':'price',
                                           'price':'count'}))

    private = (price_room[price_room['room_type']=='Private room']
                 .sort_values(by='price', ascending=True))
    private = pd.DataFrame(private.price.value_counts())
    private = (private.reset_index().rename(columns={'index':'price',
                                                 'price':'count'}))

    shared = (price_room[price_room['room_type']=='Shared room']
                 .sort_values(by='price', ascending=True))
    shared = pd.DataFrame(shared.price.value_counts())
    shared = (shared.reset_index().rename(columns={'index':'price',
                                                           'price':'count'}))

    hotel = (price_room[price_room['room_type']=='Hotel room']
                 .sort_values(by='price', ascending=True))
    hotel = pd.DataFrame(hotel.price.value_counts())
    hotel = (hotel.reset_index().rename(columns={'index':'price',
                                                 'price':'count'}))

    fig = go.Figure()

    fig.add_trace(go.Box(y=entire['count'], 
                               marker=dict(color="#00A699"),
                              name='Entire home/apt'))

    fig.add_trace(go.Box(y=private['count'],
                               marker=dict(color="#00A699"),
                              name='Private room'))

    fig.add_trace(go.Box(y=shared['count'],
                              marker=dict(color="#00A699"),
                              name='Shared room'))

    fig.add_trace(go.Box(y=hotel['count'],
                              marker=dict(color="#00A699"),
                              name='Hotel room'))

    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    
    fig.update_layout(title={
        'text': ('Entire Home/Apartment have the largest price range '
                'among all Room Types'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        showlegend=False)
    
    fig.update_yaxes(title_text='Price')
    fig.update_xaxes(title_text='Room Type')

    fig.show()

def plot_prop_price(df):
    '''Plots a box plot of the average price per property type.'''
    price_prop = pd.DataFrame(df[['property_type', 'price']])
    
    apartment = (price_prop[price_prop['property_type']=='apartment']
                 .sort_values(by='price', ascending=True))
    apartment = pd.DataFrame(apartment.price.value_counts())
    apartment = (apartment.reset_index()
                          .rename(columns={'index':'price',
                                           'price':'count'}))

    condo = (price_prop[price_prop['property_type']=='condominium']
                 .sort_values(by='price', ascending=True))
    condo = pd.DataFrame(condo.price.value_counts())
    condo = (condo.reset_index().rename(columns={'index':'price',
                                                 'price':'count'}))

    guesthouse = (price_prop[price_prop['property_type']=='guesthouse']
                 .sort_values(by='price', ascending=True))
    guesthouse = pd.DataFrame(guesthouse.price.value_counts())
    guesthouse = (guesthouse.reset_index().rename(columns={'index':'price',
                                                           'price':'count'}))

    house = (price_prop[price_prop['property_type']=='house']
                 .sort_values(by='price', ascending=True))
    house = pd.DataFrame(house.price.value_counts())
    house = (house.reset_index().rename(columns={'index':'price',
                                                 'price':'count'}))
    
    townhouse = (price_prop[price_prop['property_type']=='townhouse']
                 .sort_values(by='price', ascending=True))
    townhouse = pd.DataFrame(townhouse.price.value_counts())
    townhouse = (townhouse.reset_index()
                          .rename(columns={'index':'price',
                                           'price':'count'}))

    fig = go.Figure()

    fig.add_trace(go.Box(y=apartment['count'], 
                               marker=dict(color="#00A699"),
                              name='apartment'))

    fig.add_trace(go.Box(y=condo['count'],
                               marker=dict(color="#00A699"),
                              name='condominium'))

    fig.add_trace(go.Box(y=guesthouse['count'],
                              marker=dict(color="#00A699"),
                              name='guesthouse'))

    fig.add_trace(go.Box(y=house['count'],
                              marker=dict(color="#00A699"),
                              name='house'))
    
    fig.add_trace(go.Box(y=townhouse['count'],
                              marker=dict(color="#00A699"),
                              name='townhouse'))

    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig.update_layout(title={
        'text': ('Apartments have the largest price range '
                'among all the Property Types'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        showlegend=False)

    fig.update_yaxes(title_text='Price')
    fig.update_xaxes(title_text='Property Type')
    
    fig.show()

def plot_avg_price_neighbour(df, bracket):
    """
    Plot the top highest or the top lowest average price per neighbourhood
    
    Parameters
    ----------
    df : dataframe
        This contains the listings.
    bracket : string
        Determines whether user wants to get the top highest or top lowest
        average price.
    """
    avg_price_neigh = pd.DataFrame(df
                                   .groupby('neighbourhood_cleansed')['price']
                                   .mean().sort_values(ascending=False))
    
    if bracket=='top':
        top_avg_price = avg_price_neigh.head(10)

        top_avg_price = (top_avg_price.reset_index()
                              .rename(columns={
                                  'neighbourhood_cleansed':'Neighbourhood',
                                  'price':'Avg Price'}))

        fig = px.bar(top_avg_price.sort_values(by='Avg Price', 
                                               ascending=True), 
                     x='Avg Price', y='Neighbourhood',
                     color=['others']*(len(top_avg_price)-1)+['Yarra Ranges'],
                     color_discrete_sequence=['#BDBDBD','#FF5A5F'])
        fig.update_xaxes(title_text='Average Price')
        fig.update_yaxes(title_text='Neighbourhood')
        fig.update_layout(title={
            'text': ('Yarra Ranges has the Highest Average Listing Price '
                     'in Melbourne'),
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            showlegend=False)
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
        })
        fig.show()
    elif bracket=='low':
        low_avg_price = avg_price_neigh.tail(10)

        low_avg_price = (low_avg_price.reset_index()
                              .rename(columns={
                                  'neighbourhood_cleansed':'Neighbourhood',
                                  'price':'Avg Price'}))
        fig = px.bar(low_avg_price.sort_values(by='Avg Price', 
                                               ascending=False), 
                     x='Avg Price', y='Neighbourhood',
                     color=['others']*(len(low_avg_price)-1)+['Brimbank'],
                     color_discrete_sequence=['#BDBDBD','#FF5A5F'])
        fig.update_xaxes(title_text='Average Price')
        fig.update_yaxes(title_text='Neighbourhood')
        fig.update_layout(title={
            'text': ('Brimbank has the Lowest Average Listing Price '
                     'in Melbourne'),
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            showlegend=False)
        fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
        })
        fig.show()
        
def plot_availability(df):
    '''Plot availability of listings in the next 30, 60, 90 and 365 days.'''
    
    availability = (pd.DataFrame(df.groupby('neighbourhood_cleansed')
                                 ['availability_30','availability_60', 
                                  'availability_90', 'availability_365']
                                 .mean()))

    availability = availability.reset_index()
    availability = (pd.melt(availability, 
                            id_vars=['neighbourhood_cleansed'], 
                            value_vars=['availability_30','availability_60', 
                                        'availability_90', 
                                        'availability_365']))

    fig = (px.line(availability, x="variable", 
                   y="value", color='neighbourhood_cleansed',
                   labels={
                       'neighbourhood_cleansed': 'Neighbourhood',
                       'variable': 'Availability',
                       'value': 'Number of Days Available'}, 
                   color_discrete_sequence=(['#BDBDBD']*18+['#FF5A5F']+
                                            ['#BDBDBD']*9+['#FF5A5F']+
                                            ['#BDBDBD'])))
    fig.update_layout(title={
        'text': ('Yarra has the highest booking rate'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    
    fig.show()

def plot_amenities(df):
    '''Plot a bar chart of the most common amennities.'''
    
    amenities_count = (pd.DataFrame(df.stack().value_counts()))
    top_amenities = (amenities_count.head(10).reset_index()
                     .rename(columns={'index':'amenities',
                                      0:'count'}))
    
    fig = px.bar(top_amenities.sort_values(by='count', ascending=True),
             x='count', y='amenities',
             color=['others']*(len(top_amenities)-1)+['parking'],
             color_discrete_sequence=['#BDBDBD','#FF5A5F'])
    
    fig.update_layout(title={
        'text': ('Parking space is the most frequent amenity'
                 ' found in listings in Melbourne'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
                      showlegend=False)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.show()

def listing_per_host(df):
    '''Plot a box plot of how many listings each host has.'''
    
    num_host_listings = (pd.DataFrame(df['host_listings_count']
                                      .value_counts()))
    
    host = (num_host_listings['host_listings_count']
            .sort_values(ascending=True))
   
    host = (host.reset_index()
                          .rename(columns={'index':'host_listings_count',
                                           'host_listings_count':'count'}))
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(y=host['host_listings_count'], 
                               marker=dict(color="#00A699"),
                              name='Number of Listings per host'))
    
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    
    fig.update_layout(title={
        'text': ('A certain AirBnB host has 408 listings'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        showlegend=False)

    fig.update_yaxes(title_text='Number of Listings per Host')
        
    fig.show()
    
def rating_host(df):
    '''Plot a box plot of the ratings each listing has.'''
    
    host_review_rating = (pd.DataFrame(df['review_scores_rating']
                                       .value_counts()))
    
    rating = (host_review_rating['review_scores_rating']
              .sort_values(ascending=True))
    
    rating = (host_review_rating.reset_index()
                          .rename(columns={'index':'review_scores_rating',
                                           'review_scores_rating':'count'}))
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(y=rating['review_scores_rating'], 
                               marker=dict(color="#00A699"),
                              name='Review Score Rating'))
    
    fig.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })
    
    fig.update_layout(title={
        'text': ('Most listings in Melbourne have High '
                'Review Scores'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        showlegend=False)

    fig.update_yaxes(title_text='Review Score')
    
    fig.show()
    
def plot_verifications(df):
    '''Plot a bar graph of the most common verification type.'''
    
    verif_count = (pd.DataFrame(df.stack().value_counts()))
    top_verif = (verif_count.head(10).reset_index()
                     .rename(columns={'index':'host verifications',
                                      0:'count'}))
    
    fig = px.bar(top_verif.sort_values(by='count', ascending=True),
             x='count', y='host verifications',
             color=['others']*(len(top_verif)-1)+['email'],
             color_discrete_sequence=['#BDBDBD','#FF5A5F'])
    
    fig.update_layout(title={
        'text': ('Email is the most frequent means of verification '
                 ' of host of listings in Melbourne'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
                      showlegend=False)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    fig.show()
        
def ohe(df):
    '''Perform one-hot encoding to all categorical variables.'''
    
    clean_df = df.rename(columns={'neighbourhood_cleansed':'neighborhood'})
    cat_types = ['host_is_superhost', 
                 'neighborhood', 
                 'property_type', 
                 'room_type', 
                 'has_availability', 
                 'bath_type']
    one_hot_encoded = (pd.get_dummies(clean_df[cat_types])
                       .drop(['host_is_superhost_f',
                              'has_availability_f'], 
                             axis=1))

    merged_df = clean_df.merge(one_hot_encoded, how='left',
                         left_index=True, right_index=True)

    merged_df = merged_df.drop(cat_types, axis=1)
    
    return merged_df


def ohe_amenities(df, amenities_df):
    '''Perfrom binary bag-of-words representation of the 
    `amenities`'''
    
    amenities_df['amenities'] = (amenities_df[amenities_df.columns[0:]]
                                 .apply(lambda x: ','.join(x.dropna()
                                                           .astype(str)
                                                           .str.strip()),
                                        axis=1))

    amenities = amenities_df['amenities'].str.split(',').to_list()
    mlb = MultiLabelBinarizer()
    amenities_mlb = mlb.fit(amenities)
    amenities_mlb = mlb.transform(amenities)
    columns = mlb.classes_
    amenities_df = pd.DataFrame(amenities_mlb, columns=columns)
    amenities_df = amenities_df.iloc[:,4:]

    amenities_df.columns = ([f'amenities_{col}' 
                             for col in amenities_df.columns])
    
    merged_df = df.merge(amenities_df, how='left',
                         left_index=True, right_index=True)
    
    return merged_df


def ohe_verifications(df, verif_df):  
    '''Perfrom binary bag-of-words representation of the 
    `host_verifications`'''

    verif_df['host_verifications'] = (verif_df[verif_df.columns[0:]]
                             .apply(lambda x: ','.join(x.dropna()
                                                       .astype(str)
                                                       .str.strip()),
                                    axis=1))
    
    verifications = verif_df['host_verifications'].str.split(',').to_list()

    mlb = MultiLabelBinarizer()
    verifications_mlb = mlb.fit(verifications)
    verifications_mlb = mlb.transform(verifications)
    columns = mlb.classes_
    verifications_df = pd.DataFrame(verifications_mlb, columns=columns)

    verifications_df.columns = ([f'host_verifications_{col}' 
                             for col in verifications_df.columns])

    final_df = df.merge(verifications_df, how='left',
                         left_index=True, right_index=True)
    
    final_df = final_df.drop(['host_verifications', 
                              'amenities'], axis=1)

    return final_df

def scaling(df):
    '''Apply MinMax Scaling to the design matrix.'''
    
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.iloc[:,1:])
    scaled_df = pd.DataFrame(scaled)
    
    return scaled_df, df

def truncated_svd(X):
    """Return transformed matrix, explained variance ratio and
    sv components from truncated SVD.
    
    Parameter
    ---------
    X : design matrix
    """
    
    n = X.shape[1]
    svd = TruncatedSVD(n_components=(n-1))
    df_new = svd.fit_transform(X)
    
    exp_var_ratio = svd.explained_variance_ratio_
    sv_comp = svd.components_
    
    return df_new, exp_var_ratio, sv_comp

def sv_comp_df(scaled_df, ohe_all, sv_comp):
    """Return a data frame of the SV components."""
    
    n = scaled_df.shape[1]
    cols = [f'SV{i + 1}' for i in range(n - 1)]
    ohe_all = ohe_all.drop('id', axis=1)
    df_sv = pd.DataFrame(sv_comp.T, columns=cols, index=ohe_all.columns)
    
    return df_sv


def plot_svd(X_new, df_sv, sv_comp, i, j):
    """Plot the transformed dataset and original projection
    on the ith and jth SVs."""
    
    length = np.linalg.norm(df_sv[[f'SV{i}', f'SV{j}']], axis=1)
    index = length.argsort()[-15:]
    p = sv_comp.T
    features = df_sv.index
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw=dict(wspace=0.3))
    ax[0].scatter(X_new[:, i-1], X_new[:, j-1])
    ax[0].set_xlabel(f'SV{i}')
    ax[0].set_ylabel(f'SV{j}')
    ax[0].set_title('Transformed Dataset')

    for feature, vec in zip(features[index], p[index]):
        ax[1].annotate('', xy=(vec[i-1], vec[j-1]),  xycoords='data',
                       xytext=(0, 0), textcoords='data',
                       arrowprops=dict(facecolor='tab:gray', ec='none',
                                       arrowstyle='simple'))
        ax[1].text(vec[i-1], vec[j-1], feature, ha='center',
                   fontsize=10)
    ax[1].set_xlim(-0.5, 0.5)
    ax[1].set_ylim(-0.5, 0.5)
    ax[1].set_xlabel(f'SV{i}')
    ax[1].set_ylabel(f'SV{j}')
    for spine in ['top', 'right', 'left', 'bottom']:
        ax[1].spines[spine].set_visible(False)
    ax[1].set_title('Original Coordinates Projection')

    plt.show()

def plot_svd_zoomed(X_new, df_sv, sv_comp, i, j):
    """Plot the transformed dataset and original projection
    on the ith and jth SVs."""
    
    length = np.linalg.norm(df_sv[[f'SV{i}', f'SV{j}']], axis=1)
    index = length.argsort()[-15:]
    p = sv_comp.T
    features = df_sv.index
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw=dict(wspace=0.3))
    ax[0].scatter(X_new[:, i-1], X_new[:, j-1])
    ax[0].set_xlabel('SV1')
    ax[0].set_ylabel('SV2')
    ax[0].set_title('Transformed Dataset')

    for feature, vec in zip(features[index], p[index]):
        ax[1].annotate('', xy=(vec[i-1], vec[j-1]),  xycoords='data',
                       xytext=(0, 0), textcoords='data',
                       arrowprops=dict(facecolor='tab:gray', ec='none',
                                       arrowstyle='simple'))
        ax[1].text(vec[i-1], vec[j-1], feature, ha='center',
                   fontsize=10)
    ax[1].set_xlim(-0.30, 0.30)
    ax[1].set_ylim(-0.30, 0.30)
    ax[1].set_xlabel(f'SV{i}')
    ax[1].set_ylabel(f'SV{j}')
    for spine in ['top', 'right', 'left', 'bottom']:
        ax[1].spines[spine].set_visible(False)
    ax[1].set_title('Original Coordinates Projection')

    plt.show()
    
def plot_variance(exp_var_ratio):
    ''' Plot the explained variance of each SV as well as cumulative 
    variance.
    
    Parameters
    ----------
    exp_var_ratio : explained variance ratio from svd
    
    Return
    ----------
    sv_cutoff :
    
    '''
    latent_features = range(0, len(exp_var_ratio)+1, 10)
    plt.figure(figsize=(10, 5))
    plt.bar(height=exp_var_ratio, x=range(
        0, len(exp_var_ratio)), color='#FF5A5F')
    plt.xticks(latent_features)
    plt.ylabel('Variance')
    plt.xlabel('Singular Values')

    sv_cumsum = pd.DataFrame(exp_var_ratio.cumsum())
    cutoff_value = 0.90
    sv_cutoff = sv_cumsum[sv_cumsum.iloc[:, 0] <= cutoff_value].index.max() + 2
    actual_var = sv_cumsum.iloc[sv_cutoff - 1, :].values[0]

    fig, ax = plt.subplots(1, figsize=(10, 5))
    ax.plot(range(1, len(exp_var_ratio) + 1),
            exp_var_ratio.cumsum(), 'o-', color='#00A699')
    
    ax.plot()
    ax.axhline(actual_var, xmax=sv_cutoff/200, ls='--', color='#484848')
    ax.axvline(sv_cutoff, ymax=actual_var-.05, ls='--', color='#484848')
    ax.set_xlabel('Number of SVs')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.annotate(f'n = {sv_cutoff}', xy=(sv_cutoff+3, 0.2), weight='bold', size=13)
    plt.show()
    return sv_cutoff
    
def plot_SVcomponents(df_sv, sv_comp):
    """Plot the top 15 features that contribute to the SV."""
    
    length = np.linalg.norm(df_sv[['SV1', 'SV2']], axis=1)
    index = length.argsort()[-15:]
    p = sv_comp.T
    features = df_sv.index

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    id1 = np.abs(df_sv['SV1']).sort_values(ascending=False)[:15].index
    top10 = df_sv['SV1'].loc[id1].sort_values()
    ax[0, 0].barh(top10.index, width=top10.values, color='#00A699')
    ax[0, 0].set_title('SV1')
    ax[0, 0].set_xlabel('Weight')

    id2 = np.abs(df_sv['SV2']).sort_values(ascending=False)[:15].index
    top10 = df_sv['SV2'].loc[id2].sort_values()
    ax[0, 1].barh(top10.index, width=top10.values, color='#00A699')
    ax[0, 1].set_title('SV2')
    ax[0, 1].set_xlabel('Weight')

    id3 = np.abs(df_sv['SV3']).sort_values(ascending=False)[:15].index
    top10 = df_sv['SV3'].loc[id3].sort_values()
    ax[1, 0].barh(top10.index, width=top10.values, color='#00A699')
    ax[1, 0].set_title('SV3')
    ax[1, 0].set_xlabel('Weight')
    
    id4 = np.abs(df_sv['SV4']).sort_values(ascending=False)[:15].index
    top10 = df_sv['SV4'].loc[id4].sort_values()
    ax[1, 1].barh(top10.index, width=top10.values, color='#00A699')
    ax[1, 1].set_title('SV4')
    ax[1, 1].set_xlabel('Weight')
    
    plt.tight_layout()
    plt.show()

def agglo_cluster(X_new):
    """Return the model of the agglomerative clustering."""
    
    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    model.fit(X_new)
    return model

def plot_agglo(X_new, model):
    """Plot the dendogram for the agglomerative clustering."""
    
    def get_distances(X,model,mode='l2'):
        distances = []
        weights = []
        children=model.children_
        dims = (X.shape[1],1)
        distCache = {}
        weightCache = {}
        for childs in children:
            c1 = X[childs[0]].reshape(dims)
            c2 = X[childs[1]].reshape(dims)
            c1Dist = 0
            c1W = 1
            c2Dist = 0
            c2W = 1
            if childs[0] in distCache.keys():
                c1Dist = distCache[childs[0]]
                c1W = weightCache[childs[0]]
            if childs[1] in distCache.keys():
                c2Dist = distCache[childs[1]]
                c2W = weightCache[childs[1]]
            d = np.linalg.norm(c1-c2)
            cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)
    
            X = np.vstack((X,cc.T))
    
            newChild_id = X.shape[0]-1
                
            if mode=='l2':  
                added_dist = (c1Dist**2+c2Dist**2)**0.5 
                dNew = (d**2 + added_dist**2)**0.5
            elif mode == 'max':  
                dNew = max(d,c1Dist,c2Dist)
            elif mode == 'actual':  
                dNew = d
    
            wNew = (c1W + c2W)
            distCache[newChild_id] = dNew
            weightCache[newChild_id] = wNew
    
            distances.append(dNew)
            weights.append(wNew)
            
        return distances, weights
    
    distance, weight = get_distances(X_new, model)
    linkage_matrix = (np.column_stack([model.children_, distance, weight])
                      .astype(float))
    plt.figure(figsize=(20,10))
    dendrogram(linkage_matrix)
    plt.show()
    
    return linkage_matrix
    
def agglo_fcluster(X_new, linkage_matrix, t):
    """Plot the clusters."""
    
    y_pred = fcluster(linkage_matrix, t=t, criterion='distance')
    plt.scatter(X_new[:,0], X_new[:,1], c=y_pred);
    return y_pred

    
def cluster_range(X, clusterer, k_start, k_stop):
    """
    Return clusters and internal validation values
    
    Accepts the design matrix `X`, clustering object `clusterer`, initial and
    final values to step through, `k_start` and `k_stop` and optionally 
    `actual` labels then computes for the internal validation values.
    
    Parameters
    ----------
    X : np.array 
        Design matrix
    clusterer : kMeans clustering object
        Clustering object
    k_start : int
        Inital value to step through
    k_stop : int
        Final value to step through
    
    Returns
    -------
    res : dict
        Dictionary containing internal validation values
    """
    ys = []
    inertias = []
    chs = []
    shs = []
    centroids = []
    res = {}
    
    for k in range(k_start, k_stop+1):
        clusterer_k = clone(clusterer)
        clusterer_k.set_params(n_clusters=k, random_state=1337)
        clusterer_k.fit(X)
        y = clusterer_k.predict(X)
        
        #internal validation values
        ys.append(y)
        inertias.append(clusterer_k.inertia_)
        centroids.append(clusterer_k.cluster_centers_)
        chs.append(calinski_harabasz_score(X, y))
        shs.append(silhouette_score(X, y))
    
    # Record results
    res['ys'] = ys
    res['inertias'] = inertias
    res['chs'] = chs
    res['centroids'] = centroids
    res['shs'] = shs
    return res

def plot_clusters(X, ys, d1=0, d2=1):
    """Plot clusters given the design matrix and cluster labels"""
    # Get max and mid k's
    mx = len(ys) + 1
    md = mx//2 + 2
    
    # Initialize figure
    fig, ax = plt.subplots(2, mx//2, dpi=150, 
                           sharex=True, sharey=True, 
                           figsize=(15, 5), 
                           gridspec_kw=dict(wspace=0.075, hspace=0.15))
    
    # Iterate through all y's
    for k,y in zip(range(2, mx+1), ys):
        # Plot points
        if k < md:
            ax[0][k%md-2].scatter(X[:, d1], X[:, d2], c=y, s=1, alpha=0.8)
            ax[0][k%md-2].set_title('$k=%d$'%k)
            
        else:
            ax[1][k%md].scatter(X[:, d1], X[:, d2], c=y, s=1, alpha=0.8)
            ax[1][k%md].set_title('$k=%d$'%k)
    
    # Set figure title
    fig.suptitle(f" SV {d1+1} & {d2+1}", fontsize=15, weight='bold')
    plt.show()
    
def plot_num_clusters(res_kmeans):
    k = 3
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(range(2, 12), res_kmeans['inertias'], 'o-', color='#2663cf', 
                    lw=3.0)
    axes[1].plot(range(2, 12), res_kmeans['chs'], 'o-', color='red', 
                    lw=3.0)
    axes[2].plot(range(2, 12), res_kmeans['shs'], 'o-', color='#fe8702', lw=3.0)
    
    for ax in axes.flatten():
        ax.axvline(k, linestyle='--', lw=2.5, color='k')
        trans = transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        ax.text(k+0.250, 0.9, "k = {}".format(k), 
                transform=trans, color='k', weight='bold', 
                fontsize=10)
    
    for ax, label in zip(axes.flatten(), ['Inertia', 'CH index', 
                                          'SH coefficient']):
        ax.set_xlabel('Number of clusters', fontsize=12)
        ax.set_title(f'{label}', fontsize=12)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    plt.show()
    
def plot_elbow_kmeans(scaled_df):
    """Plot KMeans from k=2 to k=10 and elbow plot."""
    
    sse = {}
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=1337)
        kmeans.fit(scaled_df)
        sse[k] = kmeans.inertia_
    plt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    plt.show()        
        
    
def df_with_labels(df_id, scaled_df, y_pred_150, y_pred_250, y_pred_300):
    """Return dataframe with cluster labels from kmean and agglomerative 
    clustering results."""
    
    km = KMeans(n_clusters=3).fit(scaled_df)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = df_id['id'].values
    cluster_map['cluster_kmeans'] = km.labels_
    
    cluster_df = df_id.merge(cluster_map, 
                             how='left', 
                             left_on='id', 
                             right_on='data_index')
    
    cluster_agg = pd.DataFrame()
    cluster_agg['id'] = df_id['id'].values
    cluster_agg['cluster_agg_d150'] = y_pred_150
    cluster_agg['cluster_agg_d250'] = y_pred_250
    cluster_agg['cluster_agg_d300'] = y_pred_300
    
    labelled_df = cluster_df.merge(cluster_agg,
                                   how='left',
                                   left_on='id',
                                   right_on='id')
    
    labelled_df = labelled_df.drop('data_index', axis=1)
    
    return labelled_df
    
    

    
    
    
    
    
    