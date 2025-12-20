import os
from flask import Flask, request, jsonif
from werkzeug.utils import secure_filename
# Import the main function from  analysis script
from analyzer import analyze_video

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
# Define the allowed video file extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- API Endpoint ---
@app.route('/analyze', methods=['POST'])
def upload_and_analyze_video():
    """Endpoint to upload a video and receive analysis results."""
    

    # Check if a file was sent in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Check if the file type is allowed and save it
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(video_path)
        
        
        
        try:
            
            
            # --- Run Your Analysis ---
            # Call the function from analyzer.py
            analysis_result = analyze_video(video_path, frame_stride=1)

           
            
            # --- Cleanup ---
            os.remove(video_path)
            
            # Return the JSON result
            return jsonify(analysis_result), 200
            
        except Exception as e:
            # If an error occurs, still try to clean up
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({"error": "Failed during video analysis", "details": str(e)}), 500

    return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':

    app.run()
