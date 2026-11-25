
import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO  # YOLOv11
import cv2

# Flask app setup
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = ("C:\\Users\\Hiba\\brain_tumor_detection\\static\\uploads")
MODEL_PATH = ("C:\\Users\\Hiba\\brain_tumor_detection\\models\\best.pt")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the YOLOv11 model
model = YOLO(MODEL_PATH)

# Helper function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error="No file selected")

        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run YOLO model on the uploaded image
            results = model(filepath)
            results_image = results[0].plot()  # Visualize detections
            
            # Save results image
            results_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{filename}")
            cv2.imwrite(results_path, results_image)

            return redirect(url_for('result', filename=f"result_{filename}"))

    return render_template("index.html")

@app.route("/result/<filename>")
def result(filename):
    # Display the result page
    return render_template("results.html", result_image=url_for('static', filename=f"uploads/{filename}"))

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
