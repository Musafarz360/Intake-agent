#!/usr/bin/env python3
"""
Enhanced Outbound Caller Web UI

This Flask app provides a simple dashboard to:
- Dispatch immediate outbound intake calls via the LiveKit CLI (`lk dispatch create`)
- Schedule calls for later (lightweight, in‑memory scheduler thread)
- Browse generated TXT reports and JSON no‑answer notes under `call_notes/`
- View basic analytics (counts and top reasons, best‑effort from notes)

New contributors: start from the `EnhancedCallManager` for call dispatch and
the `index` route for the UI entry. The system expects the agent worker to be
running separately (see README: `python agent.py dev`).
"""

import os
import json
import subprocess
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import uuid
import threading
import time
import pathlib

# Load environment variables
load_dotenv('.env.local')

# Create Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-this-secret-key-in-production')

 
class EnhancedCallManager:
    """Lightweight coordinator for call dispatch, scheduling, and notes.

    Notes:
    - Uses the external `lk` CLI in the current PATH to dispatch the agent
    - Keeps state in memory; restarting the server clears scheduled calls
    - A background thread polls for due schedules every ~30s
    """
    def __init__(self):
        self.active_calls = {}
        self.scheduled_calls = {}
        self.patient_database = {}
        self.scheduler_running = False
        self.start_scheduler()
        
    
    def _parse_dispatch_output(self, stdout_text):
        """Best-effort parser for LiveKit CLI dispatch output.
        Tries JSON first, then falls back to regex-based extraction.
        Returns a dict with optional keys: dispatch_id, room_name.
        """
        info = {}
        if not stdout_text:
            return info
        try:
            # Try entire output as JSON
            data = json.loads(stdout_text)
            # Common keys
            info['dispatch_id'] = (
                data.get('id')
                or data.get('dispatchId')
                or data.get('dispatch_id')
            )
            # Room can be nested or flat
            info['room_name'] = (
                (data.get('room') or {}).get('name') if isinstance(data.get('room'), dict) else data.get('room')
            ) or data.get('room_name') or data.get('roomName')
            return {k: v for k, v in info.items() if v}
        except Exception:
            pass
        # Try line-by-line JSON
        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    info['dispatch_id'] = info.get('dispatch_id') or data.get('id') or data.get('dispatchId') or data.get('dispatch_id')
                    rn = (data.get('room') or {}).get('name') if isinstance(data.get('room'), dict) else data.get('room')
                    info['room_name'] = info.get('room_name') or rn or data.get('room_name') or data.get('roomName')
            except Exception:
                continue
        # Regex fallback for common patterns
        try:
            import re
            # room name key-value
            m = re.search(r"room[\s_-]?name\s*[:=]\s*([A-Za-z0-9_-]+)", stdout_text, re.IGNORECASE)
            if m:
                info['room_name'] = info.get('room_name') or m.group(1)
            # dispatch id key-value
            m = re.search(r"dispatch[\s_-]?id\s*[:=]\s*([A-Za-z0-9_-]+)", stdout_text, re.IGNORECASE)
            if m:
                info['dispatch_id'] = info.get('dispatch_id') or m.group(1)
            # UUID fallback
            if 'dispatch_id' not in info:
                m = re.search(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b", stdout_text)
                if m:
                    info['dispatch_id'] = m.group(0)
        except Exception:
            pass
        return {k: v for k, v in info.items() if v}
    
    def start_call(self, phone_number, patient_name="", priority="normal", doctor_note: str = ""):
        """Start an outbound call with enhanced metadata"""
        try:
            call_id = str(uuid.uuid4())
            metadata = json.dumps({
                "phone_number": phone_number,
                "patient_name": patient_name,
                "priority": priority,
                "doctor_note": doctor_note,
                "call_id": call_id,
            })
            
            cmd = [
                'lk', 'dispatch', 'create',
                '--new-room',
                '--agent-name', 'outbound-caller',
                '--metadata', metadata
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Try to parse real IDs from CLI output
                parsed = self._parse_dispatch_output(result.stdout)
                dispatch_id = parsed.get('dispatch_id') or f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                room_name = parsed.get('room_name') or f"room_{datetime.now().strftime('%H%M%S')}"
                
                call_info = {
                    'call_id': call_id,
                    'dispatch_id': dispatch_id,
                    'room_name': room_name,
                    'phone_number': phone_number,
                    'patient_name': patient_name,
                    'priority': priority,
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
                
                return {
                    'success': True,
                    'call_id': call_id,
                    'dispatch_id': dispatch_id,
                    'room_name': room_name,
                    'message': f'Call initiated to {phone_number}',
                    'raw_output': result.stdout.strip() if result.stdout else ''
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

    
    
    def schedule_call(self, phone_number, scheduled_time, patient_name="", priority="normal"):
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
        """Start the background scheduler.

        A daemon thread wakes every ~30 seconds and dispatches any calls whose
        scheduled time is now past. This is a best‑effort, in‑process scheduler
        intended for single-instance dev/test usage.
        """
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
                    # Build doctor_note and retry metadata if provided
                    doc_note = call_data.get('doctor_note') or ''
                    retries = int(str(call_data.get('retries') or 1))
                    # Pass doctor_note and retry overrides through metadata
                    result = self.start_call(
                        call_data["phone_number"],
                        call_data.get("patient_name", ""),
                        call_data.get("priority", "normal"),
                        doc_note
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
        """Get call notes with enhanced processing.

        Collects both JSON no‑answer notes and generated TXT reports, normalizes
        a few fields for display, and sorts newest first.
        """
        try:
            import pathlib
            
            notes_dir = pathlib.Path(__file__).parent / "call_notes"
            if not notes_dir.exists():
                return []
            
            call_notes = []
            
            # Load JSON files (if any)
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
            
            # Load TXT medical reports
            for txt_file in notes_dir.glob("*.txt"):
                try:
                    # Extract metadata from filename
                    filename = txt_file.name
                    if filename.startswith("medical_report_"):
                        # Parse phone number and timestamp from filename
                        parts = filename.replace("medical_report_", "").replace(".txt", "").split("_")
                        if len(parts) >= 2:
                            phone_number = parts[0]
                            timestamp_str = "_".join(parts[1:])
                            
                            # Try to parse timestamp
                            try:
                                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").isoformat()
                            except:
                                timestamp = datetime.fromtimestamp(txt_file.stat().st_mtime).isoformat()
                            
                            # Read file content to extract patient info
                            with open(txt_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            # Extract patient name and chief complaint from content
                            patient_name = "Unknown"
                            chief_complaint = "Not specified"
                            
                            # Parse content for patient info
                            lines = content.split('\n')
                            for line in lines:
                                if "Patient Name:" in line:
                                    patient_name = line.split("Patient Name:")[1].strip()
                                elif "Primary Concern:" in line:
                                    chief_complaint = line.split("Primary Concern:")[1].strip()
                                elif "Chief Complaint:" in line:
                                    chief_complaint = line.split("Chief Complaint:")[1].strip()
                            
                            note_data = {
                                'file_name': filename,
                                'phone_number': phone_number,
                                'timestamp': timestamp,
                                'patient_info': {
                                    'name': patient_name,
                                    'chief_complaint': chief_complaint
                                },
                                'duration': 0,
                                'status': 'completed'
                            }
                            
                            call_notes.append(note_data)
                except Exception as e:
                    print(f"Error processing TXT file {txt_file}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            call_notes.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return call_notes
            
        except Exception:
            return []
    
    def get_analytics(self):
        """Get call analytics and statistics (best‑effort)."""
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
        appointment_date = data.get('appointment_date', '')
        priority = data.get('priority', 'normal')
        doctor_note = data.get('doctor_note', '')
        
        
        if not phone_number:
            return jsonify({
                'success': False,
                'error': 'Phone number is required'
            }), 400
        
        result = call_manager.start_call(phone_number, patient_name, priority, doctor_note)
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
        # New optional fields for richer scheduling
        template_id = data.get('template_id') or data.get('template') or ''
        sms_reminder = data.get('sms_reminder') or ''
        retries = int(str(data.get('retries') or data.get('retry_count') or 1))
        encounter_id = data.get('encounter_id') or ''
        instructions = data.get('instructions') or ''
        doctor_note = data.get('doctor_note') or instructions or ''
 
        if not phone_number or not scheduled_time:
            return jsonify({
                'success': False,
                'error': 'Phone number and scheduled time are required'
            }), 400
 
        # Include the extra fields in schedule storage via kwargs
        # We pass a dict so the manager can hold them until execution
        result = call_manager.schedule_call(phone_number, scheduled_time, patient_name, priority)
        # Augment the stored schedule with extra metadata if creation succeeded
        if result.get('success'):
            sid = result.get('schedule_id')
            if sid in call_manager.scheduled_calls:
                call_manager.scheduled_calls[sid].update({
                    'template_id': template_id,
                    'sms_reminder': sms_reminder,
                    'retries': retries,
                    'encounter_id': encounter_id,
                    'instructions': instructions,
                    'doctor_note': doctor_note,
                })
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
        # If JSON file, parse and return
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                note_data = json.load(f)
            return jsonify({
                'success': True,
                'note': note_data
            })
        # If TXT medical report, parse minimal fields and return content
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Extract lightweight metadata
            patient_name = None
            chief_complaint = None
            for line in content.split('\n'):
                if line.startswith('Patient Name:') and not patient_name:
                    patient_name = line.split('Patient Name:')[1].strip()
                if (line.startswith('Primary Concern:') or line.startswith('Chief Complaint:')) and not chief_complaint:
                    chief_complaint = line.split(':', 1)[1].strip()
            return jsonify({
                'success': True,
                'note': {
                    'file_name': filename,
                    'content': content,
                    'patient_info': {
                        'name': patient_name,
                        'chief_complaint': chief_complaint
                    }
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Unsupported note format'
            }), 400
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New: list TXT medical reports
@app.route('/reports')
def list_reports():
    try:
        import pathlib
        from datetime import datetime

        notes_dir = pathlib.Path(__file__).parent / "call_notes"
        if not notes_dir.exists():
            return jsonify([])

        reports = []
        for txt_file in notes_dir.glob("*.txt"):
            try:
                filename = txt_file.name
                # Try to extract phone and timestamp from the filename pattern
                phone_number = ""
                timestamp = datetime.fromtimestamp(txt_file.stat().st_mtime).isoformat()
                if filename.startswith("medical_report_") and filename.endswith(".txt"):
                    parts = filename.replace("medical_report_", "").replace(".txt", "").split("_")
                    if len(parts) >= 2:
                        phone_number = parts[0]
                        try:
                            timestamp = datetime.strptime("_".join(parts[1:]), "%Y%m%d_%H%M%S").isoformat()
                        except Exception:
                            pass

                reports.append({
                    'file_name': filename,
                    'phone_number': phone_number,
                    'timestamp': timestamp,
                    'size': txt_file.stat().st_size,
                })
            except Exception:
                continue

        # newest first
        reports.sort(key=lambda r: r.get('timestamp', ''), reverse=True)
        return jsonify(reports)
    except Exception:
        return jsonify([]), 500

# New: fetch a single TXT report content
@app.route('/report/<path:filename>')
def get_report(filename):
    try:
        import pathlib
        # basic validation to prevent path traversal
        if (".." in filename) or ("/" in filename) or ("\\" in filename):
            return jsonify({'success': False, 'error': 'Invalid filename'}), 400
        if not filename.endswith('.txt'):
            return jsonify({'success': False, 'error': 'Unsupported file type'}), 400

        notes_dir = pathlib.Path(__file__).parent / "call_notes"
        file_path = notes_dir / filename
        if not file_path.exists():
            return jsonify({'success': False, 'error': 'Report not found'}), 404

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return jsonify({'success': True, 'file_name': filename, 'content': content})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Outbound Caller Web UI...")
    print("Make sure your agent is running with: python agent.py dev")
    app.run(debug=True, host='0.0.0.0', port=5000)
