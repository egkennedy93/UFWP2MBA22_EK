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

# iterating through each row 
for row_index, row in enumerate(df.itertuples()):
    for index, i in enumerate(row.genres):
        column_name = 'genres_{}'.format(index+1)
        df.at[row_index, column_name] = i
df.drop(columns='genres')


# deleting rows where gross or budget is blank
df = df[df.gross.notnull()]
df = df[df.budget.notnull()]

# dropping the aspect_ratio column
df.drop(columns='aspect_ratio')

# only two blanks for critics. Decided to drop the blanks since it's so few
df = df[df.num_critic_for_reviews.notnull()]

#only one blank for duration. dropping the blank rows
df = df[df.duration.notnull()]


# # update float and int values that have 0 to '.'
avg_columns = ['director_facebook_likes', 'actor_3_facebook_likes', 'actor_1_facebook_likes',
               'cast_total_facebook_likes', 'movie_facebook_likes']
# df.select_dtypes(include=[np.float64])
for i in avg_columns:
    df[i] = df[i].replace(0, df[i].mean())

# M = GP = PG, X = NC-17. We want to replace M and GP with PG, replace X with NC-17
df['content_rating'] = df['content_rating'].replace({'M': 'PG', 'GP': 'PG', 'X': 'NC-17'})


# making unclear ratings into 1 new rating
df['content_rating'] = df['content_rating'].replace({'Not Rated': 'R', 'Unrated': 'R', 'Passed': 'R'})



df.to_csv('output_data.csv')

