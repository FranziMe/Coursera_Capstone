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

# ## Step 1: Data aquisition
# 1.1 read and clean city/population data from UN  
# 1.2 get coordinates for all cities using `geopy.geocoder`  
# 1.3 bin population size  
# 1.4 for each city get all venues within 5km radius of city center using `foursquare` API

# ### 1.1 read and clean city/population data from UN

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

# ### 1.2 get coordinates for all cities using `geopy.geocoder`

# add longitude and latitude to table
# initialize column, only necessary because it is using the column before it's first assignment
population['Country'] = ''

# {{{
for i in population.index:
    city = population.at[i,'City']
    country = population.at[i,'Country']
    if country == '':
        #print(city)
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
            country = np.nan
        population.at[i, 'Country'] = country
        population.at[i, 'Latitude'] = latitude
        population.at[i, 'Longitude'] = longitude

print(population.shape)
population.head()
# }}}

population.tail()


# ### 1.3 bin population size  

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

# ### 1.4 for each city get all venues within 5km radius of city center using `foursquare` API

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

# read in table
city_venues = pd.read_excel('city_venues.xlsx', index_col= 0)
city_venues.dropna(inplace = True)
city_venues.head()

population.head()

# ## Step 2: Data wrangeling
# 2.1 redefine groups  
# 2.2 get one-hot encoding for Venue Category and/or Grouped Venue Category  
# 2.3 summarize by ciy either mean or sum  
# 2.4 merge with population data  

# ### 2.1 redefine groups (not doing that for now)



# ### 2.2 get one-hot encoding for Venue Category (and or Grouped Venue Category)

# {{{
# one hot encoding
city_onehot = pd.get_dummies(city_venues[['Venue Category']], prefix="", prefix_sep="")

city_onehot.drop(['City'], axis = 1, inplace = True)
# add neighborhood column back to dataframe
city_onehot['City'] = city_venues['City'] 

# move neighborhood column to the first column
fixed_columns = [city_onehot.columns[-1]] + list(city_onehot.columns[:-1])
city_onehot = city_onehot[fixed_columns]

city_onehot.head()
# }}}

# ### 2.3 summarize by ciy either sum (or mean)  
#

city_grouped = city_onehot.groupby('City').sum().reset_index()
city_grouped.head()

# ### 2.4 merge with population data  

# add population and location data to data, and save table
population_venue_data = population.merge(city_grouped, on = 'City')
population_venue_data.to_excel('population_venue_data.xlsx')
population_venue_data.head()

# ## Step 3: Visualize data
# 3.1 PCA  
# 3.2 TSNE  
# 3.3 barplot or boxplot for group Venues  
# 3.4 plot number of venues vs. number of categories (showing diversity)  

# ### 3.1 PCA

# {{{
# look at data with PCA and TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# remove city and country for the PCA
X = population_venue_data.drop(population_venue_data.iloc[:, 0:6], inplace=False, axis=1)
print(X.shape)

x_pca = StandardScaler().fit_transform(X) # normalizing the features
pca_cities = PCA(n_components=2)
principalComponents_cities = pca_cities.fit_transform(x_pca)
PC_cities_Df = pd.DataFrame(data = principalComponents_cities
             , columns = ['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_cities.explained_variance_ratio_))

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Population City Dataset",fontsize=20)
targets = population_venue_data['population_bin'].unique()
colors = ['black','red', 'green', 'blue', 'yellow']
for target, color in zip(targets,colors):
    indicesToKeep = population_venue_data['population_bin'] == target
    plt.scatter(PC_cities_Df.loc[indicesToKeep, 'principal component 1']
               , PC_cities_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
# }}}

# ### 3.2 TSNE

# {{{
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

X = population_venue_data.drop(population_venue_data.iloc[:, 0:6], inplace=False, axis=1)

tsne = TSNE(perplexity = 30)
X_embedded = tsne.fit_transform(X)

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Population City Dataset",fontsize=20)

targets = population_venue_data['population_bin'].unique()
colors = ['black','red', 'green', 'blue', 'yellow']

for target, color in zip(targets,colors):
    indicesToKeep = population_venue_data['population_bin'] == target
    plt.scatter(X_embedded[indicesToKeep,0]
               , X_embedded[indicesToKeep,1], c = color, s = 50)

plt.legend(targets,prop={'size': 15})

#sns.scatterplot(, X_embedded[:,1])
# }}}

# ## Step 4: Write function to find similar city
# 4.1 input: population/venue data, favorite city, number of similar cities, choice of algorithm, select or deselect categories  
# 4.2 remove or select specified categories  
# 4.3 use hierachical clustering to find x similar cities  
# 4.4 return similar cities  
# 4.5 make a heatmap showing similar cities and features (!=0)  

# ## Step 5: Apply functions
# 5.1 show the result for 5 examples  
# 5.2 pretend to select a city  
# 5.3 use `foursquare` to get more information on an example city  

# ## Possible discussion points and future directions. 
# - select more cities and create an average of input cities
# - select more cities and use K-means clustering with selected cities as input
#   - show top venues for each cluster
#   - build recommender engine for selecting multiple cities


