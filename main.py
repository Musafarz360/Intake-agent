#!/usr/bin/env python3
"""
Enhanced Outbound Caller Web UI with Advanced Features
"""

import os
import json
import subprocess
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import uuid
import threading
import time

# Load environment variables
load_dotenv('.env.local')

# Create Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this')

class EnhancedCallManager:
    def __init__(self):
        self.active_calls = {}
        self.scheduled_calls = {}
        self.patient_database = {}
        self.scheduler_running = False
        self.start_scheduler()
    
    def start_call(self, phone_number, patient_name="", priority="normal", notes=""):
        """Start an outbound call with enhanced metadata"""
        try:
            call_id = str(uuid.uuid4())
            metadata = json.dumps({
                "phone_number": phone_number,
                "patient_name": patient_name,
                "priority": priority,
                "notes": notes,
                "call_id": call_id
            })
            
            cmd = [
                'lk', 'dispatch', 'create',
                '--new-room',
                '--agent-name', 'outbound-caller',
                '--metadata', metadata
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                dispatch_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                room_name = f"room_{datetime.now().strftime('%H%M%S')}"
                
                call_info = {
                    'call_id': call_id,
                    'dispatch_id': dispatch_id,
                    'room_name': room_name,
                    'phone_number': phone_number,
                    'patient_name': patient_name,
                    'priority': priority,
                    'notes': notes,
                    'start_time': datetime.now().isoformat(),
                    'status': 'active',
                    'duration': 0
                }
                
                self.active_calls[dispatch_id] = call_info
                
                # Store in patient database
                if phone_number not in self.patient_database:
                    self.patient_database[phone_number] = {
                        'name': patient_name,
                        'total_calls': 0,
                        'last_call': None,
                        'notes': []
                    }
                
                self.patient_database[phone_number]['total_calls'] += 1
                self.patient_database[phone_number]['last_call'] = datetime.now().isoformat()
                self.patient_database[phone_number]['notes'].append(notes)
                
                return {
                    'success': True,
                    'call_id': call_id,
                    'dispatch_id': dispatch_id,
                    'room_name': room_name,
                    'message': f'Call initiated to {phone_number}'
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
    
    def schedule_call(self, phone_number, scheduled_time, patient_name="", priority="normal", notes=""):
        """Schedule a call for later"""
        try:
            schedule_id = str(uuid.uuid4())
            scheduled_datetime = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
            
            scheduled_call = {
                'schedule_id': schedule_id,
                'phone_number': phone_number,
                'scheduled_time': scheduled_datetime.isoformat(),
                'patient_name': patient_name,
                'priority': priority,
                'notes': notes,
                'status': 'scheduled'
            }
            
            self.scheduled_calls[schedule_id] = scheduled_call
            return {
                'success': True,
                'schedule_id': schedule_id,
                'message': f'Call scheduled for {scheduled_datetime.strftime("%Y-%m-%d %H:%M")}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def start_scheduler(self):
        """Start the background scheduler"""
        if not self.scheduler_running:
            self.scheduler_running = True
            scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            scheduler_thread.start()
            print("📅 Scheduler started for automatic calls")
    
    def _scheduler_loop(self):
        """Background loop to check and execute scheduled calls"""
        while self.scheduler_running:
            try:
                current_time = datetime.now()
                calls_to_execute = []
                
                # Find calls that are ready to execute
                for schedule_id, call_data in self.scheduled_calls.items():
                    try:
                        scheduled_time = datetime.fromisoformat(call_data["scheduled_time"])
                        # Make both times timezone-naive for comparison
                        if scheduled_time.tzinfo:
                            scheduled_time = scheduled_time.replace(tzinfo=None)
                        current_naive = current_time.replace(tzinfo=None)
                        
                        if (call_data["status"] == "scheduled" and scheduled_time <= current_naive):
                            calls_to_execute.append((schedule_id, call_data))
                    except Exception as e:
                        print(f"Error processing scheduled call {schedule_id}: {e}")
                        continue
                
                # Execute ready calls
                for schedule_id, call_data in calls_to_execute:
                    print(f"🕐 Executing scheduled call to {call_data['phone_number']}")
                    
                    # Mark as executing
                    self.scheduled_calls[schedule_id]["status"] = "executing"
                    
                    # Make the call
                    result = self.start_call(
                        call_data["phone_number"],
                        call_data["patient_name"],
                        call_data["priority"],
                        f"Scheduled call: {call_data['notes']}"
                    )
                    
                    # Update status
                    if result["success"]:
                        self.scheduled_calls[schedule_id]["status"] = "completed"
                        print(f"✅ Scheduled call executed successfully")
                    else:
                        self.scheduled_calls[schedule_id]["status"] = "failed"
                        print(f"❌ Scheduled call failed: {result.get('error', 'Unknown error')}")
                
                # Clean up old completed/failed calls (older than 24 hours)
                cutoff_time = current_time - timedelta(hours=24)
                expired_calls = []
                for schedule_id, call_data in self.scheduled_calls.items():
                    try:
                        scheduled_time = datetime.fromisoformat(call_data["scheduled_time"])
                        if scheduled_time.tzinfo:
                            scheduled_time = scheduled_time.replace(tzinfo=None)
                        if (call_data["status"] in ["completed", "failed"] and scheduled_time < cutoff_time):
                            expired_calls.append(schedule_id)
                    except:
                        continue
                
                for schedule_id in expired_calls:
                    del self.scheduled_calls[schedule_id]
                
                # Sleep for 30 seconds before checking again
                time.sleep(30)
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(60)  # Wait longer if there's an error
    
    def get_scheduled_calls(self):
        """Get all scheduled calls"""
        return [
            {**call_data, "schedule_id": schedule_id}
            for schedule_id, call_data in self.scheduled_calls.items()
        ]
    
    def cancel_scheduled_call(self, schedule_id):
        """Cancel a scheduled call"""
        if schedule_id in self.scheduled_calls:
            self.scheduled_calls[schedule_id]["status"] = "cancelled"
            return {"success": True, "message": "Call cancelled"}
        return {"success": False, "error": "Call not found"}
    
    def get_call_notes(self):
        """Get call notes with enhanced processing"""
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
                        
                        # Calculate call duration if available
                        if 'start_time' in note_data and 'end_time' in note_data:
                            try:
                                start = datetime.fromisoformat(note_data['start_time'])
                                end = datetime.fromisoformat(note_data['end_time'])
                                note_data['duration'] = (end - start).total_seconds()
                            except:
                                note_data['duration'] = 0
                        
                        call_notes.append(note_data)
                except Exception:
                    continue
            
            # Sort by timestamp (newest first)
            call_notes.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return call_notes
            
        except Exception:
            return []
    
    def get_analytics(self):
        """Get call analytics and statistics"""
        try:
            notes = self.get_call_notes()
            
            if not notes:
                return {
                    'total_calls': 0,
                    'avg_duration': 0,
                    'calls_today': 0,
                    'calls_this_week': 0,
                    'top_reasons': [],
                    'patient_count': 0
                }
            
            total_calls = len(notes)
            today = datetime.now().date()
            week_ago = today - timedelta(days=7)
            
            calls_today = 0
            calls_this_week = 0
            total_duration = 0
            reasons = {}
            patients = set()
            
            for note in notes:
                try:
                    note_date = datetime.fromisoformat(note.get('timestamp', '')).date()
                    
                    if note_date == today:
                        calls_today += 1
                    
                    if note_date >= week_ago:
                        calls_this_week += 1
                    
                    if note.get('phone_number'):
                        patients.add(note['phone_number'])
                    
                    if note.get('patient_info', {}).get('reason_for_visit'):
                        reason = note['patient_info']['reason_for_visit']
                        reasons[reason] = reasons.get(reason, 0) + 1
                    
                    if note.get('duration'):
                        total_duration += note['duration']
                        
                except:
                    continue
            
            avg_duration = total_duration / total_calls if total_calls > 0 else 0
            top_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'total_calls': total_calls,
                'avg_duration': round(avg_duration, 2),
                'calls_today': calls_today,
                'calls_this_week': calls_this_week,
                'top_reasons': top_reasons,
                'patient_count': len(patients)
            }
            
        except Exception:
            return {
                'total_calls': 0,
                'avg_duration': 0,
                'calls_today': 0,
                'calls_this_week': 0,
                'top_reasons': [],
                'patient_count': 0
            }
    
    def get_patient_database(self):
        """Get patient database information"""
        return self.patient_database

# Initialize enhanced call manager
call_manager = EnhancedCallManager()

@app.route('/')
def index():
    """Main UI page with analytics"""
    analytics = call_manager.get_analytics()
    return render_template('index.html', analytics=analytics)

@app.route('/make_call', methods=['POST'])
def make_call():
    """Start an outbound call with enhanced data"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number')
        patient_name = data.get('patient_name', '')
        priority = data.get('priority', 'normal')
        notes = data.get('notes', '')
        
        if not phone_number:
            return jsonify({
                'success': False,
                'error': 'Phone number is required'
            }), 400
        
        result = call_manager.start_call(phone_number, patient_name, priority, notes)
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

@app.route('/schedule_call', methods=['POST'])
def schedule_call():
    """Schedule a call for later execution"""
    try:
        data = request.get_json()
        phone_number = data.get('phone_number')
        scheduled_time = data.get('scheduled_time')
        patient_name = data.get('patient_name', '')
        priority = data.get('priority', 'normal')
        notes = data.get('notes', '')
        
        if not phone_number or not scheduled_time:
            return jsonify({
                'success': False,
                'error': 'Phone number and scheduled time are required'
            }), 400
        
        result = call_manager.schedule_call(phone_number, scheduled_time, patient_name, priority, notes)
        return jsonify(result)
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/scheduled_calls')
def get_scheduled_calls():
    """Get all scheduled calls"""
    try:
        scheduled_calls = call_manager.get_scheduled_calls()
        return jsonify(scheduled_calls)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cancel_call/<schedule_id>', methods=['POST'])
def cancel_scheduled_call(schedule_id):
    """Cancel a scheduled call"""
    try:
        result = call_manager.cancel_scheduled_call(schedule_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analytics')
def get_analytics():
    """Get call analytics"""
    try:
        analytics = call_manager.get_analytics()
        return jsonify(analytics)
    except Exception as e:
        return jsonify({}), 500

@app.route('/patients')
def get_patients():
    """Get patient database"""
    try:
        patients = call_manager.get_patient_database()
        return jsonify(patients)
    except Exception as e:
        return jsonify({}), 500

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
    print("🚀 Starting Enhanced Outbound Caller Web UI...")
    print("Make sure your agent is running with: python agent.py dev")
    app.run(debug=True, host='0.0.0.0', port=5000)
