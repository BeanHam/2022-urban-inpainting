import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def main():
    
    ## parameters
    root = "D:/nyc_taxi/ori_data"
    chunk_size=2000000
    batch_no=1
    init_year = '2009'
    file_names = ['2009_yellow_taxi.csv',
                  '2011_yellow_taxi.csv',
                  '2012_yellow_taxi.csv',
                  '2013_yellow_taxi.csv',
                  '2014_yellow_taxi.csv',
                  '2015_yellow_taxi.csv',
                  '2016_yellow_taxi.csv',]
    
    for file in file_names:
        year = file[:4]
        ## reset batch_no
        if year != init_year:
            init_year = year
            batch_no = 1
        for chunk in pd.read_csv(root+file, chunksize=chunk_size):
            chunk.to_csv(root+'chunk_data/'+year+'/chunk'+str(batch_no)+'.csv',index=False)
            batch_no+=1
        print(file + 'Done!')
        
if __name__ == "__main__":
    main()