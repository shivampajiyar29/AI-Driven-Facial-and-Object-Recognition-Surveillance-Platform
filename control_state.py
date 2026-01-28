import json
import time
import os

STATE_FILE = "control_state.json"

def load_state():
    """Load control state from JSON file"""
    default_state = {
        "systemStatus": "running",
        "faceDetection": True,
        "objectDetection": True,
        "alarmEnabled": True,
        "silenceUntil": 0.0,
        "lastCommand": "",
        "lastCommandAt": 0.0
    }
    
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # Ensure all keys exist
                for key, value in default_state.items():
                    if key not in state:
                        state[key] = value
                return state
    except Exception as e:
        print(f"[CONTROL] Error loading state: {e}")
    
    return default_state.copy()

def update_state(updates):
    """Update control state and save to JSON file"""
    try:
        state = load_state()
        state.update(updates)
        
        # Always update timestamp for commands
        if 'lastCommand' in updates:
            state['lastCommandAt'] = time.time()
        
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
            
        return True
    except Exception as e:
        print(f"[CONTROL] Error updating state: {e}")
        return False