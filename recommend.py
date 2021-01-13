# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:59:33 2020

@author: joshb
"""



from datetime import datetime

import os
import numpy as np
import pandas as pd
import random

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


    # Create a training data set of 1530 and a testing data set of 525
    # from the 2055 well rated movies


    	
    
    
    #print("Time taken for execution of above code = "+str(datetime.now() - startTime))
