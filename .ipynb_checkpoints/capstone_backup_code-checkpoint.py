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



def group_category(x, category_group):
    if category_group.upper() in x.upper():
        category = category_group
    else:
        category = x
    return category


# {{{
# group Venue Categories
city_venues.dropna(inplace = True)
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
defined_categories = pd.read_excel('categories.xlsx')
defined_categories.head(n = 20)

city_venues.head()

city_venues = city_venues.merge(defined_categories, how = 'left', on = ['grouped Venue Category'])

city_venues_tmp = city_venues[['Venue Category','grouped Venue Category', 'defined Venue Category']].drop_duplicates()
city_venues_tmp.head()

city_venues_tmp.to_excel('categories_for_refinement.xlsx')

print(len(set(city_venues['grouped Venue Category'])))
print(len(set(city_venues['Venue Category'])))
print(len(set(city_venues['defined Venue Category'])))


city_venues.loc[city_venues['defined Venue Category'] == 'House'].head()

# {{{
# one hot encoding
city_onehot = pd.get_dummies(city_venues[['defined Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
city_onehot['City'] = city_venues['City'] 

# move neighborhood column to the first column
fixed_columns = [city_onehot.columns[-1]] + list(city_onehot.columns[:-1])
city_onehot = city_onehot[fixed_columns]

city_onehot.head()
# }}}
# group by neighborhood but keep number not mean
city_grouped = city_onehot.groupby('City').sum().reset_index()
city_grouped.head()

# add population and location data to data, and save table
population_venue_data = population.merge(city_grouped, on = 'City')
population_venue_data.to_excel('population_venue_data.xlsx')
population_venue_data.head()


# set up data for modeling
X = population_venue_data.drop(population_venue_data.iloc[:, 0:6], inplace=False, axis=1)
y_bin = population_venue_data['population_bin']
y_pop = population_venue_data['Population Size']
X.head()

# split data for test and training for linear regression to predict population size
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_pop, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# {{{
# look at data with PCA and TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

x_pca = StandardScaler().fit_transform(X) # normalizing the features
pca_cities = PCA(n_components=2)
principalComponents_cities = pca_cities.fit_transform(x_pca)
PC_cities_Df = pd.DataFrame(data = principalComponents_cities
             , columns = ['principal component 1', 'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_cities.explained_variance_ratio_))
# }}}

PC_cities_Df.shape
population_venue_data.head()


# {{{
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

# {{{
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Longitude',fontsize=20)
plt.ylabel('Latitude',fontsize=20)
plt.title("Map of Population City Dataset",fontsize=20)
targets = population_venue_data['population_bin'].unique()
colors = ['black','red', 'green', 'blue', 'yellow']
for target, color in zip(targets,colors):
    indicesToKeep = population_venue_data['population_bin'] == target
    plt.scatter(population_venue_data.loc[indicesToKeep, 'Longitude']
               , population_venue_data.loc[indicesToKeep, 'Latitude'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})

# }}}

# multiple linear regression to find population size
from sklearn import linear_model
regr = linear_model.LinearRegression()
x_mlr = np.asanyarray(X_train)
y_mlr = np.asanyarray(y_train)
regr.fit (x_mlr, y_mlr)
# The coefficients
print ('Coefficients: ', regr.coef_)

# {{{
y_hat_mlr= regr.predict(X_test)
x_mlr_test = np.asanyarray(X_test)
y_mlr_test = np.asanyarray(y_test)
print("Residual sum of squares: %.2f"
      % np.mean((y_hat_mlr - y_mlr_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_mlr_test, y_mlr_test))
# }}}

# {{{
# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.plot(np.log(y_mlr_test), np.log(y_hat_mlr), 'o', color='black'); #maybe color by population bin, or something
print(np.corrcoef(y_mlr_test,y_hat_mlr))

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Longitude',fontsize=20)
plt.ylabel('Latitude',fontsize=20)
plt.title("Map of Population City Dataset",fontsize=20)
targets = population_venue_data['population_bin'].unique()
colors = ['black','red', 'green', 'blue', 'yellow']
for target, color in zip(targets,colors):
    indicesToKeep = X_test['population_bin'] == target
    plt.scatter(population_venue_data.loc[indicesToKeep, 'Longitude']
               , population_venue_data.loc[indicesToKeep, 'Latitude'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})


# }}}

X_test.head()

# Conclusion: Variance score is extremely low. The prediction does not work well as visiualized in Figure and indicated by the correlation coefficient. Maybe feature selection will help. Like only using one or two features. Automation to find best model

# {{{



# maybe a pca or tsne?

# try the different models predicting the population bin

# evaluate
# }}}


# {{{
# set number of clusters
kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
# }}}
