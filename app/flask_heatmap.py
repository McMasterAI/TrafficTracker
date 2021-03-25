from flask import Flask, jsonify, render_template, send_from_directory, request
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

@app.route('/heatmap', methods=['POST','GET'])
def _heatmap():
    if request.method == 'POST':
        object_class = request.form.get('object_class')
        start_date = request.form.get('date_start')
        end_date = request.form.get('date_end')
        return f'{object_class} {start_date} {end_date}'
    else:
        data_filename = os.path.join(app.config['DATA_FOLDER'], 'new_testData.csv')
        save_png_filename = 'heatmap.png'

        data_points = get_data_points(data_filename)
        grid_values = heatmap.points_to_grid_values(data_points)
        heatmap.create(grid_values, os.path.join(app.config['IMG_FOLDER'], save_png_filename))
        
        template_params = {
            "image_filename":save_png_filename,
            "object_classes":['Person','Bicycle','Car','Motorcycle','Bus','Truck']}

        return render_template("index.html", **template_params)

@app.route('/send_heatmap_file/<filename>')
def send_heatmap_file(filename):
    return send_from_directory(app.config['IMG_FOLDER'], filename)

def get_data_points(filename):
    data = pd.read_csv(filename, names = ['created_time', 'Pos_x','Pos_y', 'width', 'height', 'Class', 'Object_id', 'location_id'])
    return data

if __name__ == '__main__':
    app.run(debug=True)