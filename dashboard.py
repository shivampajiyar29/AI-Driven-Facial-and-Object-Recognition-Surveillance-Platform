from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import time
from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION
from control_state import load_state, update_state

app = Flask(__name__)
CORS(app)

# MongoDB connection
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[MONGO_DB]
    detections_collection = db[MONGO_COLLECTION]
    client.admin.command('ismaster')
    print("✅ MongoDB connected successfully")
    MONGO_CONNECTED = True
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    detections_collection = None
    MONGO_CONNECTED = False

@app.route('/')
def home():
    return render_template('dashboard.htm')

@app.route('/api/stats')
def get_stats():
    try:
        # Load current system control state
        state = load_state()
        
        # Get today's date range
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_str = today_start.strftime('%Y%m%d')
        
        # Initialize counts
        wanted_count = 0
        recent_detections = []
        
        if MONGO_CONNECTED and detections_collection is not None:
            # Count today's detections
            wanted_count = detections_collection.count_documents({
                'timestamp': {'$regex': f'^{today_str}'},
                'type': 'wanted_face'
            })
            
            # Get recent detections (last 24 hours)
            recent_detections = list(detections_collection.find().sort('timestamp', -1).limit(10))
        
        # Format detections for frontend
        formatted_detections = []
        for detection in recent_detections:
            name = detection.get('name', 'Unknown')
            timestamp = detection.get('timestamp', '')
            # Format timestamp for display
            try:
                dt = datetime.strptime(timestamp, '%Y%m%d-%H%M%S')
                display_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                display_time = timestamp
                
            formatted_detections.append({
                'name': name,
                'timestamp': display_time,
                'type': detection.get('type', 'face'),
                'alert': detection.get('alert', False),
                'snapshot': detection.get('snapshot', '')
            })
        
        # Count objects from captures folder
        object_count = 0
        if os.path.exists('captures/objects'):
            object_count = len([f for f in os.listdir('captures/objects') if f.endswith('.jpg')])
        
        # Count total face captures
        face_count = 0
        if os.path.exists('captures/faces'):
            face_count = len([f for f in os.listdir('captures/faces') if f.endswith('.jpg')])
        
        # Count active alerts
        active_alerts = len([d for d in formatted_detections if d['alert']])
        
        return jsonify({
            'wantedDetections': wanted_count,
            'objectDetections': object_count,
            'totalFaces': face_count,
            'activeAlerts': active_alerts,
            'recentDetections': formatted_detections,
            'systemStatus': state.get('systemStatus', 'running'),
            'faceDetection': state.get('faceDetection', True),
            'objectDetection': state.get('objectDetection', True),
            'alarmEnabled': state.get('alarmEnabled', True),
            'lastUpdate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cameraStatus': 'connected',
            'mongoStatus': 'connected' if MONGO_CONNECTED else 'disconnected'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections/hourly')
def hourly_detections():
    """Get hourly detection counts for the last 24 hours"""
    try:
        hours = []
        wanted_counts = []
        object_counts = []
        
        if not MONGO_CONNECTED:
            return jsonify({'error': 'MongoDB not connected'}), 500
            
        for i in range(24):
            hour_time = datetime.now() - timedelta(hours=23-i)
            hour_str = hour_time.strftime('%Y%m%d-%H')
            
            # Count wanted detections in this hour
            wanted_count = detections_collection.count_documents({
                'timestamp': {'$regex': f'^{hour_str}'},
                'type': 'wanted_face'
            })
            
            # Count object detections in this hour
            object_count = detections_collection.count_documents({
                'timestamp': {'$regex': f'^{hour_str}'},
                'type': 'object'
            })
            
            hours.append(hour_time.strftime('%H:%M'))
            wanted_counts.append(wanted_count)
            object_counts.append(object_count)
        
        return jsonify({
            'hours': hours,
            'wanted': wanted_counts,
            'objects': object_counts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/control', methods=['POST'])
def system_control():
    """Handle system control commands"""
    data = request.json
    command = data.get('command')

    # Load current state
    state = load_state()

    if command == 'silence_alarm':
        # Temporarily silence alarm for 10 minutes
        silence_until = time.time() + 10 * 60
        update_state({
            'silenceUntil': silence_until,
            'lastCommand': 'silence_alarm',
            'lastCommandAt': time.time()
        })
        return jsonify({'status': 'success', 'message': 'Alarm silenced for 10 minutes'})

    elif command == 'capture_snapshot':
        update_state({
            'lastCommand': 'capture_snapshot',
            'lastCommandAt': time.time()
        })
        return jsonify({'status': 'success', 'message': 'Manual snapshot requested'})

    elif command == 'toggle_face_detection':
        new_value = not state.get('faceDetection', True)
        update_state({
            'faceDetection': new_value,
            'lastCommand': 'toggle_face_detection',
            'lastCommandAt': time.time()
        })
        status = "enabled" if new_value else "disabled"
        return jsonify({'status': 'success', 'message': f'Face detection {status}'})

    elif command == 'toggle_object_detection':
        new_value = not state.get('objectDetection', True)
        update_state({
            'objectDetection': new_value,
            'lastCommand': 'toggle_object_detection',
            'lastCommandAt': time.time()
        })
        status = "enabled" if new_value else "disabled"
        return jsonify({'status': 'success', 'message': f'Object detection {status}'})

    elif command == 'toggle_alarm':
        new_value = not state.get('alarmEnabled', True)
        update_state({
            'alarmEnabled': new_value,
            'silenceUntil': 0.0,
            'lastCommand': 'toggle_alarm',
            'lastCommandAt': time.time()
        })
        status = "enabled" if new_value else "disabled"
        return jsonify({'status': 'success', 'message': f'Alarm {status}'})

    elif command == 'emergency_stop':
        update_state({
            'systemStatus': 'stopped',
            'lastCommand': 'emergency_stop',
            'lastCommandAt': time.time()
        })
        return jsonify({'status': 'success', 'message': 'System emergency stopped'})

    elif command == 'start_system':
        update_state({
            'systemStatus': 'running',
            'lastCommand': 'start_system',
            'lastCommandAt': time.time()
        })
        return jsonify({'status': 'success', 'message': 'System started'})

    else:
        return jsonify({'status': 'error', 'message': 'Unknown command'})

@app.route('/api/detections')
def get_detections():
    """Get paginated detections"""
    try:
        if not MONGO_CONNECTED or detections_collection is None:
            return jsonify({'error': 'MongoDB not connected'}), 500
            
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        skip = (page - 1) * limit
        
        detections = list(detections_collection.find()
                         .sort('timestamp', -1)
                         .skip(skip)
                         .limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for detection in detections:
            detection['_id'] = str(detection['_id'])
            timestamp = detection.get('timestamp', '')
            try:
                dt = datetime.strptime(timestamp, '%Y%m%d-%H%M%S')
                detection['display_time'] = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                detection['display_time'] = timestamp
        
        return jsonify(detections)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Face Detection Dashboard...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔍 Make sure MongoDB is running and main.py is started")
    print("=" * 50)
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)