# add alpha (transparency) to a colormap
import matplotlib; matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def points_to_grid_values(data):
    x = [int(e) for e in data.Pos_x.to_list()[1:]]
    y = [int(e) for e in data.Pos_y.to_list()[1:]]
    w = [int(e) for e in data.width.to_list()[1:]]
    h = [int(e) for e in data.height.to_list()[1:]]

    cell_size = 12
    width = 1920
    height = 1920   # bug with grid lines

    adjusted_width = width//cell_size
    adjusted_height = height//cell_size
    heatmap_data = [[0 for j in range(adjusted_width)] for i in range(adjusted_height)]

    for i in range(len(x)):
        for col in range(x[i]//cell_size, (x[i]+w[i])//cell_size):
            for row in range(y[i]//cell_size, (y[i]+h[i])//cell_size):
                if (row < adjusted_height and col < adjusted_width): 
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
    hmax = sns.heatmap(
        heatmap_data,
        # cmap = "winter", # cm name or object
        alpha = 0.5, # whole heatmap is translucent
        annot = False,
        zorder = 2,
        xticklabels = False,
        yticklabels = False,
        cbar=False,
        square=True)
    # sns.axes_style("white")
    # sns.heatmap(
    #     heatmap_data,
    #     zorder = 2,
    #     xticklabels=False,
    #     yticklabels=False,
    #     cbar=False,
    #     square=True)

    map_img = mpimg.imread('./imgs/intersection2.jpeg') 
    hmax.imshow(
        map_img,
        aspect = hmax.get_aspect(),
        extent = hmax.get_xlim() + hmax.get_ylim(),
        zorder = 1) #put the map under the heatmap

    fig.savefig(filename)
    plt.close(fig)
