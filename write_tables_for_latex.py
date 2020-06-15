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

# This script reads in tables and writes out a file containing tables formated in Latex style for easy copy paste into Latex script

# {{{
# table with population
# table with population and coordinates
# table with venues
# full input table for running the function
# }}}

import pandas as pd

pop_data = pd.read_excel('table08_clean.xlsx')
pop_data.head()

pop_coordinates = pd.read_excel('population_table.xlsx')
pop_coordinates.drop(['Unnamed: 0'], axis = 1, inplace = True)
pop_coordinates.head()

venues = pd.read_excel('city_venues.xlsx')
venues.drop(['Unnamed: 0'], axis = 1, inplace = True)
venues.head()

one_hot = pd.read_excel('population_venue_data.xlsx')
one_hot.drop(['Unnamed: 0'], axis = 1, inplace = True)
one_hot.head()
one_hot.shape

one_hot.iloc[:,[0,1,2,3,4,5,6, 254 ,757]].head()

# {{{
# write tables
# column separater = &

pop_data.to_csv('tables/pop_data.txt', sep = '&', index = False)
pop_coordinates.to_csv('tables/pop_coordinates.txt', sep = '&', index = False)
venues.to_csv('tables/venues.txt', sep = '&', index = False)
one_hot.iloc[:,[0,1,2,3,4,5,6, 254 ,757]].to_csv('tables/one_hot.txt', sep = '&', index = False)

# }}}




