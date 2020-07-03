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

# # Capstone Project: find my next travel destination

# In this Ipython notebook I will first aquire necessary data, explore the data, write functions to find similar cities based on the diversity of venues per city and demonstrate the how the functions work based on a few examples

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import plotly.express as px

import seaborn as sns
# %matplotlib inline
import matplotlib.pyplot as plt

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
population = pd.read_excel('table08_clean.xlsx')
population.head()
# }}}
# ### 1.2 get coordinates for all cities using `geopy.geocoder`

# add longitude and latitude to table
# initialize column, only necessary because it is using the column before it's first assignment
population['got_coordinates'] = ''

# {{{
for i in population.index:
    city = population.at[i,'City']
    country = population.at[i,'Country']
    got_coordinates = population.at[i,'got_coordinates']
    if got_coordinates == '':
        print(city)
        geolocator = Nominatim(user_agent="foursquare_agent")
        location = geolocator.geocode('%s, %s' %(city, country))
        if not location == None:
            # get longitude and latitude 
            latitude = location.latitude
            longitude = location.longitude
        else:
            latitude = np.nan
            longitude = np.nan
            country = np.nan
        population.at[i, 'got_coordinates'] = 'yes'            
        population.at[i, 'Latitude'] = latitude
        population.at[i, 'Longitude'] = longitude

print(population.shape)
population.head()
# }}}

population.drop('got_coordinates', axis = 1, inplace = True)
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

CLIENT_ID = '' # your Foursquare ID
CLIENT_SECRET = '' # your Foursquare Secret
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

def getNearbyVenues(names, latitudes, longitudes, radius=5000, LIMIT=5000):
    
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
            # if there are no venues add nan for the city
            venues_list.append([(
                name, 
                lat, 
                lng, 
                np.nan, 
                np.nan, 
                np.nan,  
                np.nan)])
    
    # create a dataframe from results
    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    # add column name
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
# run the function
city_venues = getNearbyVenues(names=population['City'],
                                   latitudes=population['Latitude'],
                                   longitudes=population['Longitude']
                                  )

city_venues.head()

# }}}

# {{{
# safe table
# city_venues.to_excel('city_venues.xlsx')

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
# 2.3 summarize venues by city using sum  
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

# ### 2.3 summarize venues by city using sum 
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

population_venue_data = pd.read_excel('population_venue_data.xlsx')
population_venue_data.drop(['Unnamed: 0'], axis = 1, inplace = True)
population_venue_data.head()

# ### 3.1 PCA

# {{{
# look at data with PCA and TSNE

# remove city and country for the PCA
X = population_venue_data.drop(population_venue_data.iloc[:, [0,2]], inplace=False, axis=1)
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

# save plot
plt.savefig('figures/pca.png', format = 'png')
plt.savefig('figures/pca.pdf', format = 'pdf')

# }}}

# ### 3.2 TSNE

# {{{
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

X = population_venue_data.drop(population_venue_data.iloc[:, 0:3], inplace=False, axis=1)

tsne = TSNE(perplexity = 20)
X_embedded = tsne.fit_transform(StandardScaler().fit_transform(X))

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('TSNE - 1',fontsize=20)
plt.ylabel('TSNE - 2',fontsize=20)
plt.title("TSNE of Population City Dataset",fontsize=20)

targets = population_venue_data['population_bin'].unique()
colors = ['black','red', 'green', 'blue', 'yellow']

for target, color in zip(targets,colors):
    indicesToKeep = population_venue_data['population_bin'] == target
    plt.scatter(X_embedded[indicesToKeep,0]
               , X_embedded[indicesToKeep,1], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.savefig('figures/tsne.png', format = 'png')
plt.savefig('figures/tsne.pdf', format = 'pdf')

# }}}

# ### 3.3.1 histogram of how often a venue is present

# {{{
occurence_of_venues = population_venue_data.drop(population_venue_data.iloc[:, 0:6], inplace=False, axis=1).sum(axis =0)
occurence_of_venues

plt.figure(figsize=(10,5))

sns.set_style('darkgrid')
sns.distplot(occurence_of_venues)
plt.xlabel("Total occurence of Venue")


plt.savefig('figures/occurrence_per_venue.png', format = 'png')
plt.savefig('figures/occurrence_per_venue.pdf', format = 'pdf')

# }}}

# ### 3.3.1 histogram of how many venues each city has

# {{{
venues_per_city = population_venue_data.drop(population_venue_data.iloc[:, 0:6], inplace=False, axis=1).sum(axis = 1)
venues_per_city

plt.figure(figsize=(10,5))

sns.set_style('darkgrid')
sns.distplot(venues_per_city)
plt.xlabel("Venues per City")

plt.savefig('figures/venues_per_city.png', format = 'png')
plt.savefig('figures/venues_per_city.pdf', format = 'pdf')

# }}}

# ### 3.4 plot number of venues vs. number of categories (showing diversity)  

# {{{
# venues_per_city from previous cell
# calculate category per city

tmp = population_venue_data.drop(population_venue_data.iloc[:, 0:6], inplace=False, axis=1)
category_per_city = tmp.mask(tmp > 0, 1).sum(axis = 1)
category_per_city.head()
# }}}
# {{{
from scipy import stats 
plt.figure(figsize=(10,10))

sns.set_style('darkgrid')
ax = sns.regplot(x=category_per_city, y=venues_per_city)
plt.xlabel("Venue Categories per City")
plt.ylabel("Venues per City")
pearsonR = stats.pearsonr(category_per_city, venues_per_city)
ax.text(10, 200, 'pearson R: %s' %(np.round(pearsonR[0], 2)), horizontalalignment='left', size='larger', color='black')


plt.savefig('figures/venues_per_city_vs_categories.png', format = 'png')
plt.savefig('figures/venues_per_city_vs_categories.pdf', format = 'pdf')

# }}}


np.round(pearsonR[0], 2)


# The figure shows a clear indication that most cities are balanced in their repertoire of different venues the more venues they have

# ## Step 4: Write function to find similar city
# 4.1 input: population/venue data, favorite city, number of similar cities, choice of algorithm, select or deselect categories  
# 4.2 remove or select specified categories  
# 4.3 use hierachical clustering to find x similar cities  
# 4.4 return similar cities  
# 4.5 make a heatmap showing similar cities and features (!=0)  

# ### Function find_next_vacation()

def find_next_vacation(input_city, population_venue_data, k, distance_method): #, select, selected_features, distance_score):

    # include the selected features
    
    # only work on venue data
    tmp = population_venue_data.drop(population_venue_data.iloc[:, 0:6], inplace=False, axis=1)
    # remove or select given features
    print("selected features") 
    
    # calculate similarity to input_city
    distance_matrix = pd.DataFrame(squareform(pdist(tmp, distance_method)))
    distance_matrix.columns = population_venue_data['City']
    distance_matrix.index = population_venue_data['City']
    print("calculated distance matrix") 
    
    if not input_city in population_venue_data['City'].tolist():
        print("ERROR, we could not find your city")
        return
    else:
        # subset to k most similar cities
        next_vacations_cities = distance_matrix.sort_values(by = input_city)[input_city].head(n = k + 1).index.tolist()
        distance_to_input_city = distance_matrix.sort_values(by = input_city)[input_city].head(n = k + 1)
        next_vacations = population_venue_data[population_venue_data['City'].isin(next_vacations_cities)]
        print("found possible vacation destinations")
        
        # clean df (i.e. remove 0 features)
        next_vacations = next_vacations.loc[:, (next_vacations.loc[next_vacations['City'] == input_city,:] != 0).any(axis=0)]
        # sort by city
        next_vacations = next_vacations.merge(distance_to_input_city, on = 'City')
        next_vacations = next_vacations.sort_values(by = input_city)
        
        
        ###################################
        # create heatmap with clustering  #
        ###################################
        print("plotting heatmap")
        plt.figure()
        # plot heatmap of data
        heatmap_data = next_vacations.drop(next_vacations.iloc[:, 0:5], inplace=False, axis=1)
        heatmap_data.drop([input_city], axis = 1, inplace = True)
        heatmap_data.index = next_vacations['City']
        
        heatmap_data.sort_values(by = input_city, axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')
        ax = sns.clustermap(heatmap_data, method = 'ward', metric = distance_method, col_cluster = False, figsize = (25,10))
        # show, save heatmap and start new plot
        
        plt.savefig('figures/heatmap_%s.png' %(input_city.replace(' ', '.')), format = 'png')
        plt.savefig('figures/heatmap_%s.pdf' %(input_city.replace(' ', '.')), format = 'pdf')

        ###################################
        # plot dots hue is distance score #
        ###################################
        
        print("plotting world map")
        plt.figure()
        plot_data = next_vacations.iloc[:, 0:6]
        plot_data = plot_data.merge(distance_to_input_city, on = 'City')
        
        plot_data['log10(Population Size)'] = np.log10(plot_data['Population Size'])
        input_city_data = plot_data[plot_data['City'] == input_city]
        plot_data = plot_data[plot_data['City'] != input_city]
        
        sns.set(rc={'figure.figsize':(15,10)})
        ax = sns.scatterplot(x='Longitude', y='Latitude', hue = input_city, size = 'log10(Population Size)', data = plot_data)
        
        # long and lat max
        plt.xlim(population_venue_data['Longitude'].min(), population_venue_data['Longitude'].max())
        plt.ylim(population_venue_data['Latitude'].min(), population_venue_data['Latitude'].max())
        
        # city label names
        for i in range(0, plot_data.shape[0]):
            ax.text(plot_data.iloc[i, :]['Longitude'], plot_data.iloc[i, :]['Latitude'], plot_data.iloc[i, :]['City'], horizontalalignment='left', size='medium', color='black', weight='light', verticalalignment='bottom')
        
        # add input city to plot
        ax = sns.scatterplot(x='Longitude', y='Latitude', color = 'red', data = input_city_data )
        for i in range(0, input_city_data.shape[0]):
            ax.text(input_city_data.iloc[i, :]['Longitude'], input_city_data.iloc[i, :]['Latitude'], input_city_data.iloc[i, :]['City'], horizontalalignment='left', size='medium', color='red', weight='light', verticalalignment='bottom')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
        
        plt.savefig('figures/worldmap_%s.png' %(input_city.replace(' ', '.')), format = 'png')
        plt.savefig('figures/worldmap_%s.pdf' %(input_city.replace(' ', '.')), format = 'pdf')

    
    # plot cities by their location, color with similarity score
    return next_vacations


def get_information_on_next_vacation(input_city, population_venue_data, LIMIT = 20):
    # use foursquares to get dataframe of venues for next city vacations
    
    latitudes = population_venue_data[population_venue_data['City'] == input_city]['Latitude']
    longitudes = population_venue_data[population_venue_data['City'] == input_city]['Longitude']
    
    venues_list=[]
    for name, lat, lng in zip(input_city, latitudes, longitudes):
            
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
                input_city, 
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
    
    city_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    city_venues.columns = ['City', 
                      'City Latitude', 
                      'City Longitude', 
                      'Venue', 
                      'Venue Latitude', 
                      'Venue Longitude', 
                      'Venue Category']
    
    return(city_venues)

# ## Step 5: Apply functions
# 5.1 show the result for 5 examples  
# 5.2 pretend to select a city  
# 5.3 use `foursquare` to get more information on an example city  

# ### 5.1 show the result for 5 examples

# ### 5.1.1 BERLIN, Germany

future_vacation_destination_Berlin = find_next_vacation('BERLIN', population_venue_data, 10, 'euclidean')
future_vacation_destination_Berlin.head()

# ### 5.1.2 Chicago (IL), USA

future_vacation_destination = find_next_vacation('Chicago (IL)', population_venue_data, 10, 'euclidean')
future_vacation_destination.head()

# ### 5.1.3 Leiden, Netherlands

future_vacation_destination = find_next_vacation('Leiden', population_venue_data, 10, 'euclidean')
future_vacation_destination.head()

# ### 5.1.4 TOKYO, Japan

future_vacation_destination = find_next_vacation('TOKYO', population_venue_data, 10, 'euclidean')
future_vacation_destination.head()

# ### 5.1.5 WIEN, Austria

future_vacation_destination = find_next_vacation('WIEN', population_venue_data, 10, 'euclidean')
future_vacation_destination.head()

# ### 5.2 pretend to select a city  
#

# Our first example input 'BERLIN' revealed, that s-Gravenhage (The Netherlands) is quite similar to Berlin (Germany). It has a similar amount of coffee shops, parks and other cultural as well as cousine related venues. Furthermore, s-Gravenhage is smaller than Berlin and far less known. Let's use the function `get_information_on_next_vacation` to find out more about s-Gravenhage.

# ### 5.3 use `foursquare` to get more information on an example city  

future_vacation_destination_info_Gravenhage = get_information_on_next_vacation('s-Gravenhage', future_vacation_destination_Berlin, 20)
future_vacation_destination_info_Gravenhage

# It looks like our next vacation will be filled with food and a little bit of culture.

# save output tables of Berlin and Gravenhage for report
future_vacation_destination_Berlin.iloc[:,[0,1,2,3,4,5,6, 23, 48,64]].to_csv('tables/Berlin.txt', sep = '&', index = False)
future_vacation_destination_info_Gravenhage.to_csv('tables/Gravenhage.txt', sep = '&', index = False)


# ## Possible discussion points and future directions. 
# - select more cities and create an average of input cities
# - better input data (missing couple countries)
# - write a FLASK App
# - include feature selection- select more cities and use K-means clustering with selected cities as input
#   - show top venues for each cluster
#   - build recommender engine for selecting multiple cities
#
#
# - on the same principle it would be possible to investigate the next country 
# - one big problem when using the foursquare API is the limit of 100 venues per request, so some cities might be under estimated
#


