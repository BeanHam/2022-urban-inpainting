import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import argparse
import json
import time

import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point, MultiLineString
from shapely.ops import polygonize

warnings.filterwarnings("ignore")

def change_time_format(time):
    if 'PM' in time:
        return str(int(time[:2])+12)
    else:
        return time[:2]
    
def count_occurence(data, grid_lat, grid_long, image_size):
    '''
    Counts occurence in grid data.
    
    Arg:
        - data: hourly data
        - grid_lat: latitude segments
        - grid_long: longitude segments
    
    Output:
        - grid count
    '''
    
    ## make geopoints
    points = gpd.GeoDataFrame({"x":data.lat,"y":data.long})
    points['geometry'] = points.apply(lambda p: Point(p.x, p.y), axis=1)
    
    ## make geogrids
    hlines = [((x1, yi), (x2, yi)) for x1, x2 in list(zip(grid_lat[:-1], grid_lat[1:])) for yi in grid_long]
    vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(grid_long[:-1], grid_long[1:]) for xi in grid_lat]
    grids = list(polygonize(MultiLineString(hlines + vlines)))
    id = [i for i in range(len(grids))]
    grid_frame = gpd.GeoDataFrame({"id":id,"geometry":grids})
    
    ## count occurrence in grids
    pointInPolys = sjoin(points, grid_frame, how='left')
    count_table = pointInPolys.groupby(['id']).size().reset_index(name='count')
    count_dict = dict(count_table.values)
    grid_data = np.zeros([image_size, image_size])
    for key in count_dict.keys():
        grid_data[key//image_size, key%image_size] = count_dict[key]
    
    return grid_data

def main():
    
    ## parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('parameter_path', 
                        default='parameter.txt', 
                        help='the path to parameter json file')
    opt = parser.parse_args()
    
    #####################################################
    ## load parameters
    parameter_path = opt.parameter_path
    with open(parameter_path) as json_file:
        parameters = json.load(json_file)
    root = parameters['root']
    lat_upper = parameters['lat_upper']
    lat_bottom = parameters['lat_bottom']
    long_left = parameters['long_left']
    long_right = parameters['long_right']
    image_size = parameters['image_size']
    date_formats = parameters['date_formats']
    column_names = parameters['column_names']
    uni_columns = parameters['uni_columns']
    years = parameters['years']
    
    #####################################################
    ## process data
    ## iterate data chunks for each year
    for year in years:
        
        if os.path.isfile(f"D:/nyc_taxi/grid_data/{year}_grid_data.npy"):
            print(f"{year} grid data exists! Start next year...")
            print('-----------------------------')
            next
        else:
            print('Prepare '+year)
            
            ## initial time
            t0 = time.time()
            
            ## get all chunk files
            dirt = root + str(year) + '/'
            files = os.listdir(dirt)
            
            ## get variable names
            variables = column_names[year]
            
            ## iterate each chunk file
            print(f"Load {year} data...")
            whole_data = []
            for file in files:
                data = pd.read_csv(dirt + file)
                data = data[variables]
                data.columns = uni_columns
                data = data.loc[(data.lat<= lat_upper) &\
                                (data.lat>= lat_bottom) &\
                                (data.long<= long_right) &\
                                (data.long>= long_left)]
                whole_data.append(data.values)
            whole_data = pd.DataFrame(np.concatenate(whole_data), columns=uni_columns)
            
            ## extract date & time
            print(f"Prepare {year} data...")
            if year != '2016':
                dates = [record[5:10] for record in whole_data.pickup_datetime]
                times = [record[11:13] for record in whole_data.pickup_datetime]
                whole_data['date'] = dates
                whole_data['time'] = times
            else:
                dates = [record[:5] for record in whole_data.pickup_datetime]
                times = [record[11:13]+record[-2:] for record in whole_data.pickup_datetime]
                times = np.array(list(map(change_time_format, times)))
                
                ## 12 -> 12AM; should be changed to 00
                ## 24 -> 12PM; should stay as 12
                times[times == '12'] = '00'
                times[times == '24'] = '12'
                whole_data['date'] = dates
                whole_data['time'] = times
            
            ## unique dates & unique times & segmentations
            UNIQUE_DATES = np.unique(dates)
            UNIQUE_TIME = np.unique(times)
            lat_segments = np.linspace(lat_upper, lat_bottom,image_size+1)
            long_segments = np.linspace(long_left, long_right,image_size+1)
            
            ## iterate unique date & time
            print(f"Grid {year} data...")
            gridded_data = []
            for uni_date in UNIQUE_DATES:
                for uni_time in UNIQUE_TIME:
                    query = f"date == '{uni_date}' & time == '{uni_time}'"
                    hourly_data = whole_data.query(query)
                    if len(hourly_data) == 0:
                        gridded_data.append(np.zeros([image_size, image_size]))
                    else:
                        gridded_data.append(count_occurence(hourly_data, lat_segments, long_segments, image_size))
            gridded_data = np.stack(gridded_data)
            
            ## final time
            t1 = time.time()
            
            ## save data
            np.save('D:/nyc_taxi/grid_data/'+year+'_grid_data',gridded_data)
            print(year+' Done!')
            print('# of ori records: {}'.format(len(whole_data)))
            print('# of gridded records: {}'.format(len(gridded_data)))
            print('Total time: {}'.format(round((t1-t0)/60),4))
            print('-----------------------------')
    
if __name__ == "__main__":
    main()