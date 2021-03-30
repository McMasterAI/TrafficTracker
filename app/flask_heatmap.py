from flask import Flask, jsonify, render_template, send_from_directory, request
import json
import os
import pandas as pd
from datetime import datetime


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

# TODO: query endpoint instead of forced template state
@app.route('/heatmap', methods=['POST','GET'])
def _heatmap():
    data_filename = os.path.join(app.config['DATA_FOLDER'], 'traffic_data2.csv')
    save_png_filename = 'heatmap.png'

    data_points = get_data_points(data_filename)
    template_params = {
        "image_filename":save_png_filename,
        "object_classes":['Person','Bicycle','Car','Motorcycle','Bus','Truck'],
        "checked":[],
        "start_date":'',
        "end_date":''
        }

    if request.method == 'POST':
        object_classes = [object_class.lower() for object_class in list(request.form.values())[:-2]]
        start_date = request.form.get('date_start') 
        end_date = request.form.get('date_end')
        grid_values = heatmap.points_to_grid_values(data_points, object_classes, [start_date, end_date])
        template_params["checked"] = object_classes
        template_params["start_date"] = start_date
        template_params["end_date"] = end_date
        
    else:
        default_objects_classes = ['truck','car','bus']
        grid_values = heatmap.points_to_grid_values(data_points, default_objects_classes)

    heatmap.create(grid_values, os.path.join(app.config['IMG_FOLDER'], save_png_filename))  
    return render_template("index.html", **template_params)

@app.route('/send_heatmap_file/<filename>')
def send_heatmap_file(filename):
    return send_from_directory(app.config['IMG_FOLDER'], filename)

def get_data_points(filename):
    data = pd.read_csv(filename, names = ['created_time', 'Pos_x','Pos_y', 'width', 'height', 'Class', 'Object_id', 'location_id'])
    data['created_time'] = data['created_time'].iloc[1:].astype(float)
    return data

if __name__ == '__main__':
    app.run(debug=True)