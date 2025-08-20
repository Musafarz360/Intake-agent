#!/usr/bin/env python3
"""
Simple Outbound Caller Web UI
"""

import os
import json
import subprocess
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.local')

# Create Flask app
app = Flask(__name__)

class CallManager:
    def __init__(self):
        self.active_calls = {}
    
    def start_call(self, phone_number):
        """Start an outbound call"""
        try:
            metadata = f'{{"phone_number": "{phone_number}"}}'
            
            cmd = [
                'lk', 'dispatch', 'create',
                '--new-room',
                '--agent-name', 'outbound-caller',
                '--metadata', metadata
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Simple parsing
                output = result.stdout
                dispatch_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                room_name = f"room_{datetime.now().strftime('%H%M%S')}"
                
                call_info = {
                    'dispatch_id': dispatch_id,
                    'room_name': room_name,
                    'phone_number': phone_number,
                    'start_time': datetime.now().isoformat(),
                    'status': 'active'
                }
                
                self.active_calls[dispatch_id] = call_info
                
                return {
                    'success': True,
                    'dispatch_id': dispatch_id,
                    'room_name': room_name
                }
            else:
                return {
                    'success': False,
                    'error': f'Command failed: {result.stderr}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_call_notes(self):
        """Get call notes from JSON files"""
        try:
            import pathlib
            
            notes_dir = pathlib.Path(__file__).parent / "call_notes"
            if not notes_dir.exists():
                return []
            
            call_notes = []
            
            for json_file in notes_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        note_data = json.load(f)
                        note_data['file_name'] = json_file.name
                        call_notes.append(note_data)
                except Exception:
                    continue
            
            # Sort by timestamp (newest first)
            call_notes.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return call_notes
            
        except Exception:
            return []

# Initialize call manager
call_manager = CallManager()

@app.route('/')
def index():
    """Main UI page"""
    return render_template('index.html')

@app.route('/make_call', methods=['POST'])
def make_call():
    """Start an outbound call"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number')
        
        if not phone_number:
            return jsonify({
                'success': False,
                'error': 'Phone number is required'
            }), 400
        
        result = call_manager.start_call(phone_number)
        return jsonify(result)
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/call_notes')
def get_call_notes():
    """Get all call notes"""
    try:
        notes = call_manager.get_call_notes()
        return jsonify(notes)
    except Exception as e:
        return jsonify([]), 500

@app.route('/view_note/<filename>')
def view_note(filename):
    """View a specific call note"""
    try:
        import pathlib
        
        notes_dir = pathlib.Path(__file__).parent / "call_notes"
        file_path = notes_dir / filename
        
        if not file_path.exists():
            return jsonify({
                'success': False,
                'error': 'Note file not found'
            }), 404
        
        with open(file_path, 'r', encoding='utf-8') as f:
            note_data = json.load(f)
        
        return jsonify({
            'success': True,
            'note': note_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Outbound Caller Web UI...")
    print("Make sure your agent is running with: python agent.py dev")
    app.run(debug=True, host='0.0.0.0', port=5000)
