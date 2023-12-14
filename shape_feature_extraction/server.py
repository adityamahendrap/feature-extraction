from flask import Flask, render_template, request, redirect, flash
import secrets
from werkzeug.utils import secure_filename 
import cv2
import extraction as ext

app = Flask(__name__)
app.secret_key = secrets.token_hex(16) 

def save_request_file(request):
    if 'file' not in request.files:
        flash('No file part')
        print('No file part')
        return redirect(request.url)
        
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        print('No selected file')
        return redirect(request.url)

    path = 'static/upload/' + secure_filename(file.filename)
    file.save(path)
    return path

def save_image(edges, metode):
    path = 'static/upload/' + metode + '_' + secrets.token_hex(16) + '.jpg'
    cv2.imwrite(path, edges)
    return path

def process_method(img, method):
    method_mapping = {
        'robert': ext.get_edges_with_robert,
        'sobel': ext.get_edges_with_sobel,
        'prewitt': ext.get_edges_with_prewitt,
        'canny': ext.get_edges_with_canny,
        'hough_transform': ext.get_edges_with_canny
    }

    if method in method_mapping:
        edges = method_mapping[method](img)
        if method is 'hough_transform':
            edges = ext.get_hough_transform(edges)
        result_path = save_image(edges, method)
        contours = ext.get_contours(edges)
        features = ext.get_shape_features(contours)
        return {
            'features': features,
            'result_path': result_path,
            'metode': 'Hough Transform' if method is 'hough_transform' else method.capitalize()
        }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/shape_feature_extraction', methods=['POST'])
def shape_feature_extraction():
    data = request.form
    initial_path = save_request_file(request)
    img = cv2.imread(initial_path, 0)
    
    if(data['metode'] == 'hough_transform'):
        edges = ext.get_edges_with_canny(img)
        edges = ext.get_hough_transform(edges)
        result_path = save_image(edges, data['metode'])
        contours = ext.get_contours(edges)
        features = ext.get_shape_features(contours)
        return render_template('result.html', initial_path=initial_path, features=features, result_path=result_path, metode='Hough Transform')
    elif(data['metode'] == 'robert'):
        edges = ext.get_edges_with_robert(img)
        result_path = save_image(edges, data['metode'])
        contours = ext.get_contours(edges)
        features = ext.get_shape_features(contours)
        return render_template('result.html', initial_path=initial_path, features=features, result_path=result_path, metode='Robert')
    elif(data['metode'] == 'sobel'):
        edges = ext.get_edges_with_sobel(img)
        result_path = save_image(edges, data['metode'])
        contours = ext.get_contours(edges)
        features = ext.get_shape_features(contours)
        return render_template('result.html', initial_path=initial_path, features=features, result_path=result_path, metode='Sobel')
    elif(data['metode'] == 'prewitt'):
        edges = ext.get_edges_with_prewitt(img)
        result_path = save_image(edges, data['metode'])
        contours = ext.get_contours(edges)
        features = ext.get_shape_features(contours)
        return render_template('result.html', initial_path=initial_path, features=features, result_path=result_path, metode='Prewitt')
    elif(data['metode'] == 'canny'):
        edges = ext.get_edges_with_canny(img)
        result_path = save_image(edges, data['metode'])
        contours = ext.get_contours(edges)
        features = ext.get_shape_features(contours)
        return render_template('result.html', initial_path=initial_path, features=features, result_path=result_path, metode='Canny')

@app.route('/shape_feature_extraction/all', methods=['POST'])
def shape_feature_extraction_all():
    initial_path = save_request_file(request)
    img = cv2.imread(initial_path, 0)
    
    robert_result = process_method(img, 'robert')
    sobel_result = process_method(img, 'sobel')
    prewitt_result = process_method(img, 'prewitt')
    canny_result = process_method(img, 'canny')
    hough_transform_result = process_method(img, 'hough_transform')
    
    return render_template('result_all.html', initial_path=initial_path, robert_result=robert_result, sobel_result=sobel_result, prewitt_result=prewitt_result, canny_result=canny_result, hough_transform_result=hough_transform_result)
    
    