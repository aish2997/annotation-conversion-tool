from dataclasses import dataclass
from flask import Flask, flash,render_template,request, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
import converter

import os

app = Flask(__name__, static_folder = os.path.join(os.getcwd()))

@app.route('/uploadImages', methods = ['GET', 'POST'])
def uploadImages():
   if request.method == 'POST':
    image_files = request.files.getlist('image_file')
    if len (image_files)>0:
        for f in image_files:
            f.save('./image_data/'+ secure_filename(f.filename))
    # flash("Images Uploaded Successfully")        
    return redirect(url_for('test'))

@app.route('/uploadAnnotations', methods = ['GET', 'POST'])
def uploadAnnotations():
   if request.method == 'POST':
    annotation_files = request.files.getlist('annotation_file')
    if len (annotation_files)>0:
        for f in annotation_files:
            f.save('./annotation_data/'+ secure_filename(f.filename))
             
    return redirect(url_for('test'))

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
    image_files = request.files.getlist('image_file')
    annotation_files = request.files.getlist('annotation_file')
    source = request.form.get('source')
    format = request.form.get('format')
    if len (annotation_files)>0:
        for f in annotation_files:
            f.save('./annotation_data/'+ secure_filename(f.filename))
            
    if len (image_files)>0:
        for f in image_files:
            f.save('./image_data/'+ secure_filename(f.filename))
            
    url = source+'2'+format
    return redirect(url_for(url))
@app.route('/unavailable')
def unavailable():
    return render_template("public/unavailable.html")

@app.route('/')
def test():
    converter.clean('output')
    return render_template("public/index.html")

# @app.route('/')
# def welcome():
#     return render_template("public/upload.html")

@app.route('/polygon_vgg_json2coco_json', methods = ['GET', 'POST'])
def polygon_vgg_json2coco_json():
    converter.polygon_vgg_json2coco_json()
    converter.clean('image_data')
    converter.clean('annotation_data')
    return redirect(url_for('download', filename = 'polygon_vgg_json2coco_json.json'))

@app.route('/pascalvoc_xml2tf_csv', methods = ['GET', 'POST'])
def pascalvoc_xml2tf_csv():
    converter.pascalvoc_xml2tf_csv()
    converter.clean('image_data')
    converter.clean('annotation_data')
    return redirect(url_for('download', filename = 'pascalvoc_xml2tf_csv.csv'))

@app.route('/polygon_vgg_json2tf_csv', methods = ['GET', 'POST'])
def polygon_vgg_json2tf_csv():
    converter.polygon_vgg_json2tf_csv()
    converter.clean('image_data')
    converter.clean('annotation_data')
    return redirect(url_for('download', filename = 'polygon_vgg_json2tf_csv.csv'))

@app.route('/rectangle_vgg_json2tf_csv', methods = ['GET', 'POST'])
def rectangle_vgg_json2tf_csv():
    converter.rectangle_vgg_json2tf_csv()
    converter.clean('image_data')
    converter.clean('annotation_data')
    return redirect(url_for('download', filename = 'rectangle_vgg_json2tf_csv.csv'))

@app.route('/rectangle_vgg_json2rectangle_vgg_json', methods = ['GET', 'POST'])
def rectangle_vgg_json2rectangle_vgg_json():
    converter.merge_json()
    converter.clean('image_data')
    converter.clean('annotation_data')
    return redirect(url_for('download', filename = 'rectangle_vgg_json2rectangle_vgg_json.json'))

@app.route('/polygon_vgg_json2polygon_vgg_json', methods = ['GET', 'POST'])
def polygon_vgg_json2polygon_vgg_json():
    converter.merge_json()
    converter.clean('image_data')
    converter.clean('annotation_data')
    return redirect(url_for('download', filename = 'polygon_vgg_json2polygon_vgg_json.json'))

@app.route('/pascalvoc_xml2polygon_vgg_json', methods = ['GET', 'POST'])
def pascalvoc_xml2polygon_vgg_json():
    converter.pascalvoc_xml2polygon_vgg_json()
    converter.clean('image_data')
    converter.clean('annotation_data')
    return redirect(url_for('download', filename = 'pascalvoc_xml2polygon_vgg_json.json'))


@app.route('/pascalvoc_xml2coco_json', methods = ['GET', 'POST'])
def pascalvoc_xml2coco_json():
    converter.pascalvoc_xml2coco_json()
    converter.clean('image_data')
    converter.clean('annotation_data')
    return redirect(url_for('download', filename = 'pascalvoc_xml2coco_json.json'))

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    downloads = os.path.join(os.getcwd(),'output')
    return send_from_directory(directory=downloads, filename=filename,as_attachment=True, cache_timeout=0)

if __name__=='__main__':
    # app.static_folder = 'static'
    app.run(host='0.0.0.0', port=5001, debug=True)
