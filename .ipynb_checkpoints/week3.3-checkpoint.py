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

# ## Part 1

# only keeping necessary code from part 1

import pandas as pd
df = pd.read_html('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')[0]
df.drop(df[df['Borough'] == "Not assigned" ].index , inplace=True)
print(df.head())
df.shape

# ## Part 2

# only keeping necessary code from part 2



# loop over whole data frame
# ```
#
# import geocoder # import geocoder
#
# latitudes = []
# longitudes = []
#
# for postal_code in df["Postal Code"]:
#     # initialize your variable to None
#     lat_lng_coords = None
#      
#     # loop until you get the coordinates
#     while(lat_lng_coords is None):
#         g = geocoder.google('{}, Toronto, Ontario'.format(postal_code))
#         lat_lng_coords = g.latlng
#
#     latitudes += lat_lng_coords[0]
#     longitudes += lat_lng_coords[1]
#
# df['Latitude'] = latitudes
# df['Longitude'] = longitudes
# ```

# The above code ran forever, hence I downloaded the table given in the assignment description

# {{{
# # !wget https://cocl.us/Geospatial_data
# }}}

postal_codes = pd.read_csv('Geospatial_data')
postal_codes.head()

# Now merging the data frames of Boroughs and Neighborhoods with the Latitude and Longitude for each Postal Code

df = pd.merge(df, postal_codes, on='Postal Code')
df.head()
