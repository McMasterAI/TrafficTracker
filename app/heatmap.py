# add alpha (transparency) to a colormap
import matplotlib; matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image

img_filename = './imgs/streetView.png'

def points_to_grid_values(data):

    x = [int(e) for e in data.Pos_x.to_list()[1:]]
    y = [int(e) for e in data.Pos_y.to_list()[1:]]
    w = [int(e) for e in data.width.to_list()[1:]]
    h = [int(e) for e in data.height.to_list()[1:]]

    im = Image.open(img_filename)
    width, height = im.size
    cell_size = 12

    adjusted_width = width//cell_size
    adjusted_height = height//cell_size
    heatmap_data = [[0 for j in range(adjusted_width)] for i in range(adjusted_height)]

    for i in range(len(x)):
        for col in range(x[i]//cell_size, (x[i]+w[i])//cell_size):
            for row in range(y[i]//cell_size, (y[i]+h[i])//cell_size):
                if (row < adjusted_height and col < adjusted_width): 
                    heatmap_data[row][col] += 1

    # # data trimming
    # target = max([max(row) for row in heatmap_data])
    # for i in range(len(heatmap_data)):
    #     for j in range(len(heatmap_data[i])):
    #         if heatmap_data[i][j] > 0.70*target:
    #             heatmap_data[i][j] = 0
    #         if heatmap_data[i][j] < 0.4*target:
    #             heatmap_data[i][j] *=2

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

    map_img = mpimg.imread(filename if filename else img_filename) 
    plt.imshow(
        map_img,
        extent=[0,len(heatmap_data[0])] + [len(heatmap_data),0],
        zorder = 1) #put the map under the heatmap
        
    fig.savefig(filename)
    plt.close(fig)
