# add alpha (transparency) to a colormap
import matplotlib; matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def points_to_grid_values(data, width=100, height=100):
    x = data.x.to_list()
    y = data.y.to_list()
    w = data.w.to_list()
    h = data.h.to_list()

    #declare empty matrix
    heatmap_data = [[0 for j in range(width)] for i in range(height)]

    cell_size = 1
    #for every point
    for i in range(len(x)):
        #note that x,y represents top left corner, which starts at (0,0)
        for col in range(x[i]//cell_size, (x[i]+w[i])//cell_size):
            for row in range(y[i]//cell_size, (y[i]+h[i])//cell_size):
                if (row < width and col < height):
                    heatmap_data[row][col]+=1
    return heatmap_data

# TODO: remove chart numbers/ grid
def create(heatmap_data, filename=None):
#     wd = matplotlib.cm.winter._segmentdata # only has r,g,b  
#     wd['alpha'] = ((0.0, 0.0, 0.3), 
#                    (0.3, 0.3, 1.0),
#                    (1.0, 1.0, 1.0))
#     # modified colormap with changing alpha
#     al_winter = LinearSegmentedColormap('AlphaWinter', wd)

    # get the map image as an array so we can plot it
    map_img = mpimg.imread('./imgs/intersection.jpg') 

    fig = plt.figure()
    hmax = sns.heatmap(
        heatmap_data,
        # cmap = "winter", # cm name or object
        alpha = 0.5, # whole heatmap is translucent
        annot = False,
        zorder = 2,
        xticklabels = False,
        yticklabels = False,
        cbar=False)

    # heatmap uses pcolormesh instead of imshow, so we can't pass through 
    # extent as a kwarg, so we can't mmatch the heatmap to the map. Instead, 
    # match the map to the heatmap:
    hmax.imshow(
        map_img,
        aspect = hmax.get_aspect(),
        extent = hmax.get_xlim() + hmax.get_ylim(),
        zorder = 1) #put the map under the heatmap

    fig.savefig(filename)
    plt.close(fig)
