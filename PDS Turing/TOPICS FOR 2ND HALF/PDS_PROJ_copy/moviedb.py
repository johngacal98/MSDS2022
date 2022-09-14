import numpy as np
import pickle
import shutil
import os
import pandas as pd
from tempfile import TemporaryDirectory
from numpy.testing import assert_equal, assert_almost_equal, assert_raises
from collections import Counter
import matplotlib.pyplot as plt

class MovieDB:
    def __init__(self, path):
        self.data_dir = path
        os.makedirs(self.data_dir, exist_ok=True)
        
        return
    
    def add_movie(self, title, year, genre, director):
        if os.path.isfile(os.path.join(self.data_dir,'movies.csv')):
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        else:
            self.movies = pd.DataFrame(columns=['movie_id', 'title', 'year', 'genre', 'director_id'])
            self.movies.to_csv(os.path.join(self.data_dir,'movies.csv'), index=False)
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        if os.path.isfile(os.path.join(self.data_dir,'directors.csv')):
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
            
        else:
            self.directors = pd.DataFrame(columns=['director_id', 'given_name', 'last_name'])
            self.directors.to_csv(os.path.join(self.data_dir,'directors.csv'), index=False) 
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
            
        ###------------directors.csv appending-------------------   
        last_name, given_name = director.split(',')
        given_name = given_name.strip()
        last_name = last_name.strip()
        
        name1_bool = self.df_dir['given_name'].str.lower().isin([given_name.lower()])
        name2_bool = self.df_dir['last_name'].str.lower().isin([last_name.lower()])
        
        if len(self.df_dir[(name1_bool)&(name2_bool)]) == 1:
            director_id = self.df_dir[(name1_bool)&(name2_bool)]['director_id'].item()
        else:
            new_id = len(self.df_dir) + 1

            add_dir = {'director_id': new_id, 
                      'given_name': given_name,
                      'last_name': last_name}
            df_add = pd.DataFrame(add_dir, index=[0])
            df_add.to_csv(os.path.join(self.data_dir,'directors.csv'), mode='a', header=False, index=False)
            director_id = new_id
            
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
        ##------------movies.csv appending-------------------
        title_bool = self.df_movies['title'].str.lower().isin([title.lower().strip()])
        year_bool = self.df_movies['year'].isin([year])
        genre_bool = self.df_movies['genre'].str.lower().isin([genre.lower().strip()])
        director_id_bool = self.df_movies['director_id'].isin([director_id])
        
        if len(self.df_movies[(title_bool)&(year_bool)&(genre_bool)&(director_id_bool)] == 1):
            raise MovieDBError
        else:
            movie_id = len(self.df_movies) + 1

            add_mov = {'movie_id': movie_id,
                      'title': title,
                      'year': year, 
                      'genre': genre,
                      'director_id': director_id}
            df_addmov = pd.DataFrame(add_mov, index=[0])
            df_addmov.to_csv(os.path.join(self.data_dir,'movies.csv'), mode='a', header=False, index=False)
           
        self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
        
        return movie_id
    
    def add_movies(self,movies):
        # check if movies.csv exists already, if not create a new file
        
        if os.path.isfile(os.path.join(self.data_dir,'movies.csv')):
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        else:
            self.movies = pd.DataFrame(columns=['movie_id', 'title', 'year', 'genre', 'director_id'])
            self.movies.to_csv(os.path.join(self.data_dir,'movies.csv'), index=False)
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        
        # check if directors.csv exists already, if not create a new file
        if os.path.isfile(os.path.join(self.data_dir,'directors.csv')):
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
            
        else:
            self.directors = pd.DataFrame(columns=['director_id', 'given_name', 'last_name'])
            self.directors.to_csv(os.path.join(self.data_dir,'directors.csv'), index=False) 
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
            
            
        # create temporary variables for this function
        
        counter = 0
        new_movie_ids = []
        
        display(self.df_dir)
        # go through all the dictionary of movies in the list
        for movie in movies:
            # keep on repopulating DF for any new movies or directors added
            df_movies_dir = self.df_movies.merge(self.df_dir, how='left', on='director_id')
            
            # check if the dictionary has all the required keys
            if 'title' in movie and 'year' in movie and 'genre' in movie and 'director' in movie:
                # create temporary variables when checking /creating a movie in the list
                # use strip() to clean spaces
                last_name, given_name = movie['director'].split(',')
                given_name = given_name.strip()
                last_name = last_name.strip()
                movie['title'] = movie['title'].strip()
                movie['genre'] = movie['genre'].strip()
                
                
                # check if the movie already in movies.csv and directors.csv
                if movie['title'].lower() in df_movies_dir['title'].str.lower().to_list() and \
                        movie['year'] in df_movies_dir['year'].to_list() and \
                        movie['genre'].lower() in df_movies_dir['genre'].str.lower().to_list() and \
                        given_name.lower() in df_movies_dir['given_name'].str.lower().to_list() and \
                        last_name.lower() in df_movies_dir['last_name'].str.lower().to_list():
                    m = movie['title']
                    print(f'Warning: movie {m} is already in the database. Skipping...')
                    
                else:
                    ###------------directors.csv appending-------------------   
                    name1_bool = self.df_dir['given_name'].str.lower().isin([given_name.lower()])
                    name2_bool = self.df_dir['last_name'].str.lower().isin([last_name.lower()])

                    if len(self.df_dir[(name1_bool)&(name2_bool)]) == 1:
                        director_id = self.df_dir[(name1_bool)&(name2_bool)]['director_id'].item()
                        
                    else:
                        new_id = len(pd.read_csv(os.path.join(self.data_dir,'directors.csv'))) + 1
                        add_dir = {'director_id': new_id, 
                                  'given_name': given_name,
                                  'last_name': last_name}
                        df_add = pd.DataFrame(add_dir, index=[0])
                        df_add.to_csv(os.path.join(self.data_dir,'directors.csv'), mode='a', header=False, index=False)
                        self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))

                        director_id = new_id
                    
                    ##------------movies.csv appending-------------------
                    movie_id = len(pd.read_csv(os.path.join(self.data_dir,'movies.csv'))) + 1
                    add_mov = {'movie_id': movie_id,
                              'title': movie['title'],
                              'year': movie['year'], 
                              'genre': movie['genre'],
                              'director_id': director_id}
                    df_addmov = pd.DataFrame(add_mov, index=[0])
                    df_addmov.to_csv(os.path.join(self.data_dir,'movies.csv'), mode='a', header=False, index=False)
                    self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
                    
                    # append to new_movie_ids list
                    new_movie_ids.append(movie_id)
            else:
                print(f'Warning: movie index {counter} has invalid or incomplete information. Skipping...')
            
            # temp variable just for the index of the list of movies "to be added"
            counter += 1
        
        df_movies_dir = self.df_movies.merge(self.df_dir, how='left', on='director_id')
        display(df_movies_dir)
        
        return new_movie_ids


    def search_movies(self, title = None, year = None, genre = None, director_id = None):
        if os.path.isfile(os.path.join(self.data_dir,'movies.csv')):
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        else:
            self.movies = pd.DataFrame(columns=['movie_id', 'title', 'year', 'genre', 'director_id'])
            self.movies.to_csv(os.path.join(self.data_dir,'movies.csv'), index=False)
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))

        # check if the parameter is not present or has the correct data type
        if (type(title) == str or title == None) and \
            (type(year) == int or year == None) and \
            (type(genre) == str or genre == None) and \
            (type(director_id) == int or director_id == None):
            
            # handle none parameters
            temp_title = title
            temp_genre = genre
            if title != None:
                temp_title = temp_title.strip().lower()
            if temp_genre != None:
                temp_genre = temp_genre.strip().lower()
                
            df_temp = self.df_movies[ 
                        (self.df_movies['title'].str.lower() == temp_title) |
                        (self.df_movies['year'] == year) | 
                        (self.df_movies['genre'].str.lower() == temp_genre) | 
                        (self.df_movies['director_id'] == director_id)
                        ]
            
            # convert the column of movie_id to a list
            temp_list = df_temp['movie_id'].tolist()
            
            # check if at least one of the parameters is present
            if title == None and year == None and genre == None and director_id == None:
                raise MovieDBError
            else:
                return temp_list
    def delete_movie(self, movie_id):
        if os.path.isfile(os.path.join(self.data_dir,'movies.csv')):
            df_to_del = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        else:
            self.movies = pd.DataFrame(columns=['movie_id', 'title', 'year', 'genre', 'director_id'])
            self.movies.to_csv(os.path.join(self.data_dir,'movies.csv'), index=False)
            df_to_del = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
        

        if movie_id not in df_to_del['movie_id'].values:
            raise MovieDBError
        else:
            index = df_to_del[df_to_del['movie_id']==movie_id].index.item()
            df_done = df_to_del.drop(index)
            df_done.to_csv(os.path.join(self.data_dir,'movies.csv'), index=False)
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            return
    def export_data(self):
        if os.path.isfile(os.path.join(self.data_dir,'movies.csv')):
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        else:
            self.movies = pd.DataFrame(columns=['movie_id', 'title', 'year', 'genre', 'director_id'])
            self.movies.to_csv(os.path.join(self.data_dir,'movies.csv'), index=False)
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        if os.path.isfile(os.path.join(self.data_dir,'directors.csv')):
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
            
        else:
            self.directors = pd.DataFrame(columns=['director_id', 'given_name', 'last_name'])
            self.directors.to_csv(os.path.join(self.data_dir,'directors.csv'), index=False) 
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
            
        merged = self.df_movies.merge(self.df_dir, on='director_id', how='left').sort_values(by='movie_id')
        merged = merged.drop(['movie_id','director_id'], axis=1)
        merged.rename(columns={'given_name':'director_given_name', 'last_name':'director_last_name'}, inplace=True)
        return merged[['title', 'year', 'genre', 'director_last_name','director_given_name']]
    def token_freq(self): 
        if os.path.isfile(os.path.join(self.data_dir,'movies.csv')):
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        else:
            self.movies = pd.DataFrame(columns=['movie_id', 'title', 'year', 'genre', 'director_id'])
            self.movies.to_csv(os.path.join(self.data_dir,'movies.csv'), index=False)
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        list_mov = self.df_movies['title'].str.lower().str.split().tolist()
        li_to_count = [item for sublist in list_mov for item in sublist]
        return Counter(li_to_count)

    def generate_statistics(self, stat):
        if os.path.isfile(os.path.join(self.data_dir,'movies.csv')):
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        else:
            self.movies = pd.DataFrame(columns=['movie_id', 'title', 'year', 'genre', 'director_id'])
            self.movies.to_csv(os.path.join(self.data_dir,'movies.csv'), index=False)
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        if os.path.isfile(os.path.join(self.data_dir,'directors.csv')):
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
            
        else:
            self.directors = pd.DataFrame(columns=['director_id', 'given_name', 'last_name'])
            self.directors.to_csv(os.path.join(self.data_dir,'directors.csv'), index=False) 
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
        
        if stat not in ['movie', 'genre', 'director', 'all']:
            raise MovieDBError
        else:   
            
            if stat == 'movie':
                return dict(self.df_movies.groupby('year')['movie_id'].count())
            elif stat == 'genre':

                d = {}

                for a,b in self.df_movies.groupby(['genre'])['year']:

                    val = dict(self.df_movies[self.df_movies['genre']== a].groupby('year')['movie_id'].count())
                    d[a]= val
                return d
            elif stat =='director':
                df_merged = self.df_movies.merge(self.df_dir, on='director_id', how='left')

                full_name = df_merged['last_name'] + ', ' + df_merged['given_name']
                df_merged['full_name'] = full_name

                d={}
                for a in df_merged.full_name.unique():
                    val = dict(df_merged[df_merged['full_name']==a].groupby('year')['movie_id'].count())
                    d[a] = val

                return d
            elif stat == 'all':
                all_d = {}

                for i in ['movie', 'genre', 'director']:
                    if i == 'movie':
                        all_val = dict(self.df_movies.groupby('year')['movie_id'].count())
                    elif i == 'genre':
                        d_1 = {}
                        for a,b in self.df_movies.groupby(['genre'])['year']:
                            val = dict(self.df_movies[self.df_movies['genre']== a].groupby('year')['movie_id'].count())
                            d_1[a]= val
                        all_val = d_1
                    elif i == 'director':
                        df_merged = self.df_movies.merge(self.df_dir, on='director_id', how='left')

                        full_name = df_merged['last_name'] + ', ' + df_merged['given_name']
                        df_merged['full_name'] = full_name

                        d_2={}
                        for a in df_merged.full_name.unique():
                            val = dict(df_merged[df_merged['full_name']==a].groupby('year')['movie_id'].count())
                            d_2[a] = val
                        all_val = d_2

                    all_d[i] = all_val
                return all_d
        
    def plot_statistics(self, stat):
        # check if movies.csv exists already, if not create a new file
        if os.path.isfile(os.path.join(self.data_dir,'movies.csv')):
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        else:
            self.movies = pd.DataFrame(columns=['movie_id', 'title', 'year', 'genre', 'director_id'])
            self.movies.to_csv(os.path.join(self.data_dir,'movies.csv'), index=False)
            self.df_movies = pd.read_csv(os.path.join(self.data_dir,'movies.csv'))
            
        
        # check if directors.csv exists already, if not create a new file
        if os.path.isfile(os.path.join(self.data_dir,'directors.csv')):
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
            
        else:
            self.directors = pd.DataFrame(columns=['director_id', 'given_name', 'last_name'])
            self.directors.to_csv(os.path.join(self.data_dir,'directors.csv'), index=False) 
            self.df_dir = pd.read_csv(os.path.join(self.data_dir,'directors.csv'))
        
        # create merged DF of movies and director
        df_movies_dir = self.df_movies.merge(self.df_dir, how='left', on='director_id')
        
        
        if stat == 'movie':
            gr = self.df_movies.groupby('year')['movie_id'].count()
            fig, ax = plt.subplots(1,1,figsize=(15,10))
            ax.set_ylabel('movies')
            ax.bar(gr.index,gr.values)
        elif stat == 'genre':
            temp_df = pd.DataFrame(df_movies_dir.groupby(['year', 'genre'])['movie_id'].count().reset_index())
            temp_df = temp_df.pivot(index = 'year', columns = 'genre', values = 'movie_id')
            
            plt.rcParams['figure.figsize'] = [15, 10]
            ax = temp_df.plot(kind='line', ylabel='movies', marker='o')
        elif stat == 'director':
            full_name = df_movies_dir['last_name'] + ', ' + df_movies_dir['given_name']
            df_movies_dir['full_name'] = full_name
            temp_df = pd.DataFrame(df_movies_dir.groupby(['full_name', 'year'])['movie_id'].count().reset_index())
            
            # get the top 5 directors
            temp_df_dir = pd.DataFrame(temp_df.groupby(['full_name'])['movie_id'].sum().reset_index())
            top_5_dir = temp_df_dir.sort_values(['movie_id','full_name'], ascending=[0,1]).head(5)
            top_5_dir = top_5_dir['full_name'].to_list()
#             display(top_5_dir)
            
            temp_df = temp_df.pivot(index = 'year', columns = 'full_name', values = 'movie_id')
#             display(temp_df)
#             top_5_dir = temp_df.sum(axis=0).sort_values(ascending=False).head(5)
            
            temp_df = temp_df[top_5_dir]
            
            plt.rcParams['figure.figsize'] = [15, 10]
            ax = temp_df.plot(kind='line', ylabel='movies', marker = 'o')
            
#         plt.show()
        return ax
        
class MovieDBError(Exception):
    pass 