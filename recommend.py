# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:59:33 2020

"""



from datetime import datetime

import os
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

if __name__ == "__main__":    
    # Check if csv file is in the directory
    if not os.path.isfile("../Data/movies.csv"): 
        startTime = datetime.now()
        dataset = pd.read_csv('Data/movies.csv', encoding = 'unicode_escape')
    # Print the dataset
    # Then check for null values (if any) in the dataset
    
    print(dataset)
    print("---------------------------------------------------")
    print("Number of NaN values: ")
    print(str(dataset.isnull().sum()))
    print("---------------------------------------------------")
    
    # - Convert data to a DataFrame file
    # - Find all movies with a score of 7.0 or better and add their data
    # to a new list
    # - The original csv file has the movies sorted in chronological order
    # (by year), so we now randomize the data using random.sample then 
    # convert our data into a np array
    df = pd.DataFrame(data=dataset)
    data_holder = []
    
    for value in df.values:
        if value[10] >= 7.0:
            data_holder += [value]
        
    data = random.sample(data_holder, k=len(data_holder))
    data = np.array(data)
    print("Number of Movies with a score of 7.0 or better:" , len(data))
    print("---------------------------------------------------")

    # create a dummy dataset value
    # Note oversized for getting a training set
    best_picks = []

    # iterate through dataset and determine the top scored movies
    '''
    avg_topScore = 0;
    it = 0
    for mov in data_holder:
    	if mov[10] >= avg_topScore:
    		best_picks += [mov]
    		avg_topScore = (avg_topScore + mov[10]) / (it + 2)
    		print("topscore: ", avg_topScore)
    		# todo: make this averaging actually do something notable


    # print any findings
    print("Number of movies greater than base score (init 7):  ", len(best_picks))
    print("----------------------------------------------------")
    '''


    genres = [x[4] for x in data]
    
    vectorize = TfidfVectorizer()
    #Construct the required TF-IDF matrix by fitting and transforming the data
    movie_genres = vectorize.fit_transform(genres)

    cosine_simularity = linear_kernel(movie_genres, movie_genres)

    def get_genre_recommendation(movie, cosine_simularity=cosine_simularity):
        # Find index of the original movie name
        index_movie = np.nonzero(data == movie)[0]
        # Calculate the score by finding movies with similar genres
        movie_scores = list(enumerate(cosine_simularity[index_movie][0]))
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        movie_scores = movie_scores[0:11]
        # Get the movie indices of the top 10
        movie_indices = [i[0] for i in movie_scores]
    
        movies_recommended = []
        for index in movie_indices:
            movies_recommended += [(data[index][6], data[index][14])]
        return movies_recommended

    print(get_genre_recommendation("Platoon"))



    	
    
    
    #print("Time taken for execution of above code = "+str(datetime.now() - startTime))

