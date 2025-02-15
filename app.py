from flask import Flask, jsonify, render_template
from flask_cors import CORS
import subprocess
import os
import signal

app = Flask(__name__)
CORS(app)

# Define the paths to the scripts
HAND_CONTROLLER_SCRIPT = os.path.abspath("F:/Hand gesture virtual mouse/Hand Controller.py")
GAZE_CONTROLLER_SCRIPT = os.path.abspath("F:/Hand gesture virtual mouse/Gaze Controller.py")

# Store process references for stopping them later
processes = {"hand": None, "gaze": None}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-hand', methods=['POST'])
def start_hand_controller():
    try:
        # Start Hand Controller script as a subprocess
        if processes["hand"] is None:
            processes["hand"] = subprocess.Popen(
                ["python", HAND_CONTROLLER_SCRIPT], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            return jsonify(message="Hand Controller Started"), 200
        else:
            return jsonify(message="Hand Controller is already running"), 400
    except Exception as e:
        return jsonify(message=f"Failed to start Hand Controller: {e}"), 500

@app.route('/start-gaze', methods=['POST'])
def start_gaze_controller():
    try:
        # Start Gaze Controller script as a subprocess
        if processes["gaze"] is None:
            processes["gaze"] = subprocess.Popen(
                ["python", GAZE_CONTROLLER_SCRIPT], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            return jsonify(message="Gaze Controller Started"), 200
        else:
            return jsonify(message="Gaze Controller is already running"), 400
    except Exception as e:
        return jsonify(message=f"Failed to start Gaze Controller: {e}"), 500

@app.route('/stop-hand', methods=['POST'])
def stop_hand_controller():
    try:
        # Stop the Hand Controller script
        if processes["hand"] is not None:
            processes["hand"].terminate()  # Terminate the subprocess
            processes["hand"].wait()  # Wait for process to finish cleanup
            processes["hand"] = None
            return jsonify(message="Hand Controller Stopped"), 200
        else:
            return jsonify(message="Hand Controller is not running"), 400
    except Exception as e:
        return jsonify(message=f"Failed to stop Hand Controller: {e}"), 500

@app.route('/stop-gaze', methods=['POST'])
def stop_gaze_controller():
    try:
        # Stop the Gaze Controller script
        if processes["gaze"] is not None:
            processes["gaze"].terminate()  # Terminate the subprocess
            processes["gaze"].wait()  # Wait for process to finish cleanup
            processes["gaze"] = None
            return jsonify(message="Gaze Controller Stopped"), 200
        else:
            return jsonify(message="Gaze Controller is not running"), 400
    except Exception as e:
        return jsonify(message=f"Failed to stop Gaze Controller: {e}"), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

