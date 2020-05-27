# ---
# jupyter:
#   jupytext:
#     cell_markers: '{{{,}}}'
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# {{{
import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

# #!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

# #!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')
# }}}

# {{{
# possibly need to clean up the table a bit before. add countries to cities and remove footnotes

# read and format population table
population = pd.read_excel('table08.xlsx', skiprows=4)
population = population[['Unnamed: 0', 'Both sexes\nLes deux sexes']]
population.rename(columns={"Unnamed: 0": "City", "Both sexes\nLes deux sexes": "Population Size"}, inplace = True)
# only keep cities with population data
print(population.shape)
population.dropna(inplace = True)
population = population[population['Population Size'] != '...']
print(population.shape)
population.head()
# }}}

# {{{
# add longitude and latitude to table
# initialize column, only necessary because it is using the column before it's first assignment
population['Country'] = np.nan

for i in population.index:
    city = population.at[i,'City']
    country = population.at[i,'Country']
    if not pd.notnull(country):
        geolocator = Nominatim(user_agent="foursquare_agent")
        location = geolocator.geocode(city)
        if not location == None:
            # get longitude and latitude 
            latitude = location.latitude
            longitude = location.longitude
            coordinates = '%s,%s' %(latitude, longitude)
            # get country of city, might be interesting for later
            location_reverse = geolocator.reverse(coordinates, language='en')
            loc = location_reverse.raw
            if 'country' in loc['address'].keys():
                country = loc['address']['country']
            else:
                country = 'None'
        else:
            latitude = np.nan
            longitude = np.nan
        population.at[i, 'Country'] = country
        population.at[i, 'Latitude'] = latitude
        population.at[i, 'Longitude'] = longitude

print(population.shape)
population.head()

# }}}

# {{{
# drop cities without location
population.dropna(inplace=True)
population.head()

def bin_pop_size(x):
    if x < 500000:
        bin_pop = 1
    elif x >= 500000 and x < 1000000:
        bin_pop = 2
    elif x >= 1000000 and x < 5000000:
        bin_pop = 3
    elif x >= 5000000 and x < 10000000:
        bin_pop = 4
    elif x >= 10000000 and x < 20000000:
        bin_pop = 5
    elif x >= 20000000:
        bin_pop = 6
    return bin_pop

# create a column indicating the bin based on population size
population['population_bin'] = population['Population Size'].apply(lambda x: bin_pop_size(x))

# safe table
population.to_excel('population_table.xlsx')

# look at table which we will use for the rest
print(population.shape)
population.head()

# }}}

# Okay, now that we have the cities location, their population size and their country  we can as foursquares to get all venues in a 5km radius around the city center

population = pd.read_excel('population_table.xlsx')
population.drop(['Unnamed: 0'], axis = 1, inplace = True)
population.head()

# {{{
# setting up the credentials for Foursqaure
CLIENT_ID = 'RH1MSDYU04B3JZQ31VMVTEQVZONTRLRKVP4DTM4A2XVQSZW3' # your Foursquare ID
CLIENT_SECRET = '1GIGN1TNCP5E0KUQCG10RHZWKTTH4EGAPGV4K2EHKSATD3TL' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

LIMIT = 5000
radius = 5000

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
# }}}

# {{{
# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()
        
        if 'groups' in results['response'].keys() and len(results["response"]['groups'][0]['items']) > 0:
            results = results["response"]['groups'][0]['items']
            
            
            # return only relevant information for each nearby venue
            venues_list.append([(
                name, 
                lat, 
                lng, 
                v['venue']['name'], 
                v['venue']['location']['lat'], 
                v['venue']['location']['lng'],  
                v['venue']['categories'][0]['name']) for v in results])
        
        
        else:   
            venues_list.append([(
                name, 
                lat, 
                lng, 
                np.nan, 
                np.nan, 
                np.nan,  
                np.nan)])
    
    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['City', 
                      'City Latitude', 
                      'City Longitude', 
                      'Venue', 
                      'Venue Latitude', 
                      'Venue Longitude', 
                      'Venue Category']
    
    return(nearby_venues)
# }}}

# {{{
city_venues = getNearbyVenues(names=population['City'],
                                   latitudes=population['Latitude'],
                                   longitudes=population['Longitude']
                                  )

city_venues.head()

# }}}

city_venues.columns = ['City', 
                      'City Latitude', 
                      'City Longitude', 
                      'Venue', 
                      'Venue Latitude', 
                      'Venue Longitude', 
                      'Venue Category']

# {{{
# safe table
#city_venues.to_excel('city_venues.xlsx')

# look at table which we will use for the rest
print(city_venues.shape)
city_venues.head()
# }}}

city_venues.head()

population.head()


def group_category(x, category_group):
    if category_group.upper() in x.upper():
        category = category_group
    else:
        category = x
    return category


# {{{
# group Venue Categories
# city_venues.dropna(inplace = True)
city_venues['grouped Venue Category'] = city_venues['Venue Category']
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Restaurant'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Store'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'House'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Museum'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Airport'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Joint'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Boutique'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Art'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Shop'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Bar'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Stadium'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Field'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Place'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Court'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Cafe'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Market'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Service'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Studio'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Gym'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Park'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Club'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Center'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Garden'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Salon'))
city_venues['grouped Venue Category'] = city_venues['grouped Venue Category'].apply(lambda x: group_category(x, 'Rink'))






city_venues.head()
# }}}

# create a dictionary grouping some of the rest of the categories
grouped_categries_dict = {
    cd /bio
}

print(len(set(city_venues['grouped Venue Category'])))
print(len(set(city_venues['Venue Category'])))


set(city_venues['grouped Venue Category'])

# {{{
# one hot encoding
city_onehot = pd.get_dummies(city_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
city_onehot['City'] = city_venues['City'] 

# move neighborhood column to the first column
#fixed_columns = [city_onehot.columns[-1]] + list(city_onehot.columns[:-1])
#city_onehot = city_onehot[fixed_columns]

city_onehot.head()
# }}}



city_onehot.loc[city_onehot['City'] == 'BERLIN']

# group by neighborhood but keep number not mean
toronto_grouped = toronto_onehot.groupby('Neighborhood').mean().reset_index()
toronto_grouped.head()

# {{{
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# }}}
