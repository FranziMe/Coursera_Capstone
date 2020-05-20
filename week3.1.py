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

import pandas as pd

# 1. Download the data from Wikipedia

df = pd.read_html('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')[0]
df.head()

# 2. Remove un assigned pastal codes

df.drop(df[df['Borough'] == "Not assigned" ].index , inplace=True)
df.head()

# 3. Check if any there are any NaN neighborhoods

df[df['Neighborhood'] == "NaN" ].index

# There are no unassigned neighborhoods so nothing left to be done

df.shape


