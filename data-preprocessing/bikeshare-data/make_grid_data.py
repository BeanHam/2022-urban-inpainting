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
                        default='parameters.json', 
                        help='the path to parameter json file')
    parser.add_argument('year', default=2021)
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
    year = opt.year
    
    #####################################################
    ## process data
    ## iterate data chunks for each year
    dirt = root + str(year) + '/'
    files = os.listdir(dirt)
    
    ## get variable names
    variables = [
        ['starttime', 'start station latitude', 'start station longitude'],
        ['started_at', 'start_lat', 'start_lng']
    ]
    uni_columns = ['time', 'lat', 'long']
    
    ## iterate each chunk file
    print(f"Load {str(year)} data...")
    whole_data = []
    for i in range(len(files)):
        file = files[i]
        data = pd.read_csv(dirt + file)
        try: data = data[variables[0]]
        except: data = data[variables[1]]
        data.columns = uni_columns
        data = data.loc[(data.lat<= lat_upper) &\
                        (data.lat>= lat_bottom) &\
                        (data.long<= long_right) &\
                        (data.long>= long_left)]
        whole_data.append(data.values)
    whole_data = pd.DataFrame(np.concatenate(whole_data), columns=uni_columns)
    
    ## unique dates & hours
    dates = [record.split()[0] for record in whole_data.time]
    hours = [record.split()[1].split(':')[0] for record in whole_data.time]
    whole_data['date'] = dates
    whole_data['time'] = hours
    UNIQUE_DATES = np.unique(dates)
    UNIQUE_HOURS = np.unique(hours)
    lat_segments = np.linspace(lat_upper, lat_bottom,image_size+1)
    long_segments = np.linspace(long_left, long_right,image_size+1)
    
    ## iterate unique date & time
    counter = 1
    gridded_data = []
    for uni_date in UNIQUE_DATES:
        for uni_time in UNIQUE_HOURS:
            if counter % 500 == 0:
                print(f'{counter}/{len(UNIQUE_DATES)*len(UNIQUE_HOURS)}...')
            query = f"date == '{uni_date}' & time == '{uni_time}'"
            hourly_data = whole_data.query(query)
            if len(hourly_data) == 0:
                gridded_data.append(np.zeros([image_size, image_size]))
            else:
                gridded_data.append(count_occurence(hourly_data, lat_segments, long_segments, image_size))
            counter += 1
    gridded_data = np.stack(gridded_data)
    
    ## save data
    np.save('D:/nyc_bikeshare/grid_data/'+str(year)+'_grid_data.npy',gridded_data)
    print(str(year)+' Done!')
    print('# of ori records: {}'.format(len(whole_data)))
    print('# of gridded records: {}'.format(len(gridded_data)))
    print('-----------------------------')

    ## clear memory
    del whole_data, gridded_data
            
if __name__ == "__main__":
    main()