#%%
import pandas as pd
import numpy as np
import os

DIR = os.path.dirname(__file__)
file_name = "{}/movie_metadata.csv".format(DIR)
df = pd.read_csv(file_name)

# drop any rows that have blanks
df.dropna(inplace = True)

# removing duplicates
df.drop_duplicates()

# removing trailing characters from movie title
df['movie_title'] = df['movie_title'].str.replace(r'åÊ', '')

# splitting the genres based on | and creating a list
df['genres'] = df['genres'].str.split('|')

# exploding the list so for every list item, a new row is created
df = df.explode('genres')

# deleting rows where gross or budget is blank
df = df[df.gross.notnull()]
df = df[df.budget.notnull()]

# replacing NA values with 0 for aspect_ratio
df.drop(columns='aspect_ratio')

# only two blanks for critics. Decided to drop them since it's so few
df = df[df.num_critic_for_reviews.notnull()]

#only one blank for duration. dropping
df = df[df.duration.notnull()]

# update facenumber_in_poster
df['facenumber_in_poster'] = df['facenumber_in_poster'].replace(0, '.')


# # update float and int values that have 0 to '.'
avg_columns = ['director_facebook_likes', 'actor_3_facebook_likes', 'actor_1_facebook_likes',
               'cast_total_facebook_likes', 'movie_facebook_likes']
# df.select_dtypes(include=[np.float64])
for i in avg_columns:
    df[i] = df[i].replace(0, df[i].mean())

# M = GP = PG, X = NC-17. We want to replace M and GP with PG, replace X with NC-17
df['content_rating'] = df['content_rating'].replace({'M': 'PG', 'GP': 'PG', 'X': 'NC-17'})


# making unclear ratings into 1 new rating
unclear_rating = ['Not Rated', 'Unrated', 'Passed']

df['unclear_rating'] = df['content_rating'].apply(lambda x: 'Match' if x in unclear_rating else 'mismatch')



df.to_csv('output_data.csv')
# %%