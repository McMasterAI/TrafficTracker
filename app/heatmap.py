# add alpha (transparency) to a colormap
import matplotlib; matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import time

def points_to_grid_values(data, class_filter=None, date_filter=None):
    filtered_data = data.loc[data['Class'].isin(class_filter)] if class_filter else data

    if date_filter:
        start_date = datetime.timestamp(datetime.strptime(date_filter[0], '%Y-%m-%d')) if date_filter[0] else 946702800.0 # Jan 1 2000
        end_date = datetime.timestamp(datetime.strptime(date_filter[0], '%Y-%m-%d')) if date_filter[1] else time.time()
        filtered_data = filtered_data[filtered_data['created_time'].between(start_date,end_date)]

    x = [int(e) for e in filtered_data.Pos_x.to_list()[1:]]
    y = [int(e) for e in filtered_data.Pos_y.to_list()[1:]]
    w = [int(e) for e in filtered_data.width.to_list()[1:]]
    h = [int(e) for e in filtered_data.height.to_list()[1:]]

    cell_size = 12
    width = 1920
    height = 1080   # bug with grid lines
    # map_factor = max(width,height) / min(width,height)

    adjusted_width = width//cell_size
    adjusted_height = height//cell_size
    heatmap_data = [[0 for j in range(adjusted_width)] for i in range(adjusted_height)]

    for i in range(len(x)):
        for col in range(x[i]//cell_size, (x[i]+w[i])//cell_size):
            for row in range(y[i]//cell_size, (y[i]+h[i])//cell_size):
                if (row < adjusted_height and col < adjusted_width): 
                    # adjusted_row = min(int(row*map_factor), max(width,height)//cell_size-1)
                    # print(adjusted_row)
                    heatmap_data[row][col] += 1
    return heatmap_data

# TODO: remove chart numbers/ grid
def create(heatmap_data, filename=None):
#     wd = matplotlib.cm.winter._segmentdata # only has r,g,b  
#     wd['alpha'] = ((0.0, 0.0, 0.3), 
#                    (0.3, 0.3, 1.0),
#                    (1.0, 1.0, 1.0))
#     # modified colormap with changing alpha
#     al_winter = LinearSegmentedColormap('AlphaWinter', wd)

    fig = plt.figure()
    plt.imshow(
        heatmap_data, 
        alpha=0.3,
        zorder=2,
        )
    plt.axis('off')

    # plt.show()
    # hmax = sns.heatmap(
    #     heatmap_data, 
    #     # cmap=get_alpha_blend_cmap("rocket_r", 0.5), 
    #     alpha = 0.3, # whole heatmap is translucent
    #     annot = False,
    #     zorder = 2,
    #     xticklabels = False,
    #     yticklabels = False,
    #     cbar=False,
    #     linewidths=0.0,
    #     edgecolor="none",
    #     square=True)

    map_img = mpimg.imread('./imgs/streetView.png') 
    plt.imshow(
        map_img,
        # aspect = plt.get_aspect(),
        # extent = plt.get_xlim() + plt.get_ylim(),
        extent=[0,len(heatmap_data[0])] + [len(heatmap_data),0],
        zorder = 1) #put the map under the heatmap

    # fig.savefig(filename, dpi=1000) # higer dpi is slow saving
    fig.savefig(filename)
    plt.close(fig)
