from flask import Flask, jsonify, render_template, send_from_directory
import json
import os
import pandas as pd

# custom modules
import heatmap

app = Flask(__name__)
app.config['IMG_FOLDER'] = './imgs'
app.config['DATA_FOLDER'] = './test_data'

@app.after_request
def add_header(response):   # since heatmap img gets updated, force the browser not to cache it
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def home():
    return "Hello World"

@app.route('/heatmap')
def _heatmap():
    # data_filename = os.path.join(app.config['DATA_FOLDER'], 'heatmap_table_test_data.json')
    data_filename = os.path.join(app.config['DATA_FOLDER'], 'traffic_data.csv')
    save_png_filename = 'heatmap.png'

    data_points = get_data_points(data_filename)
    grid_values = heatmap.points_to_grid_values(data_points)
    heatmap.create(grid_values, os.path.join(app.config['IMG_FOLDER'], save_png_filename))
    
    return render_template("index.html", image_filename=save_png_filename)

@app.route('/send_heatmap_file/<filename>')
def send_heatmap_file(filename):
    return send_from_directory(app.config['IMG_FOLDER'], filename)

def get_data_points(filename):
    data = pd.read_csv(filename, names = ['created_time', 'Pos_x','Pos_y', 'width', 'height', 'Class', 'Object_id', 'location_id'])
    # with open(data_filename) as data_file:
    #     data = json.loads(data_file.read())
    # resolution = data['grid_size']
    # points = [list(e['position'].values())[0:2] for e in data['heatmapTable']]
    # data = points_to_grid_values(points, width=resolution['width'],height=resolution['width'])
    return data

if __name__ == '__main__':
    app.run(debug=True)