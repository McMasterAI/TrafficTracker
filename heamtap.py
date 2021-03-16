# add alpha (transparency) to a colormap
import matplotlib.cm
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import matplotlib.image as mpimg 
import numpy.random as random 
import seaborn as sns; sns.set()

# Read the csv file
data = pd.read_csv('traffic_data.csv', names = ['key', 'datetime', 'x','y', 'w', 'h', 'class', 'object_id', 'location', 'direction'])

x = data.x.to_list()
y = data.y.to_list()
w = data.w.to_list()
h = data.h.to_list()

#declare empty matrix
height = 100
width = 100
heatmap_data = [[0 for j in range(width)] for i in range(height)]

cell_size = 1
#for evert point
for i in range(len(x)):
    #note that x,y represents top left corner, which starts at (0,0)
    for col in range(x[i]//cell_size, (x[i]+w[i])//cell_size):
        for row in range(y[i]//cell_size, (y[i]+h[i])//cell_size):
            if (row < 100 and col < 100):
                heatmap_data[row][col]+=1

wd = matplotlib.cm.winter._segmentdata # only has r,g,b  
wd['alpha'] =  ((0.0, 0.0, 0.3), 
               (0.3, 0.3, 1.0),
               (1.0, 1.0, 1.0))

# modified colormap with changing alpha
al_winter = LinearSegmentedColormap('AlphaWinter', wd) 

# get the map image as an array so we can plot it 

map_img = mpimg.imread('inference\\images\\street.jpeg') 

# making and plotting heatmap 

hmax = sns.heatmap(heatmap_data,
            #cmap = al_winter, # this worked but I didn't like it
            cmap = matplotlib.cm.winter,
            alpha = 0.5, # whole heatmap is translucent
            annot = False,
            zorder = 2,
            )

# heatmap uses pcolormesh instead of imshow, so we can't pass through 
# extent as a kwarg, so we can't mmatch the heatmap to the map. Instead, 
# match the map to the heatmap:

# hmax.imshow(map_img,
#           aspect = hmax.get_aspect(),
#           extent = hmax.get_xlim() + hmax.get_ylim(),
#           zorder = 1) #put the map under the heatmap

hmax.imshow(map_img,
          aspect = hmax.get_aspect(),
          extent = hmax.get_xlim() + hmax.get_ylim(),
          zorder = 1, interpolation = 'nearest') #put the map under the heatmap

from matplotlib.pyplot import show 
show()