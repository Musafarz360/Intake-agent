from __future__ import annotations
"""
Voice medical intake agent (outbound caller)

This module contains:
- Environment setup and validation
- A specialized `OutboundCaller` agent that runs a structured, voice-first
  pre‑visit intake conversation and generates a professional TXT report
- The `entrypoint` for LiveKit Agents workers that configures audio/LLM stack,
  dials the patient via SIP, and orchestrates the session lifecycle

New contributors: skim the high-level comments around session setup, event
handlers, and the dialing loop to understand the flow quickly.
"""

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from datetime import datetime
from typing import Any

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import (
    deepgram,
    google,
    silero,
    noise_cancellation,
)
 
# Load environment variables
# We read from a local .env file to keep API keys out of source control
import pathlib
current_dir = pathlib.Path(__file__).parent
env_path = current_dir / ".env.local"

logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=str(env_path))

# Load all required environment variables
# These drive SIP dialing and AI stack configuration
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
outbound_call_timeout_s = int(os.getenv("OUTBOUND_CALL_TIMEOUT", "45"))
outbound_retry_count = int(os.getenv("OUTBOUND_RETRY_COUNT", "1"))
outbound_retry_delay_s = int(os.getenv("OUTBOUND_RETRY_DELAY", "30"))
 
# Validate required environment variables
# Fail fast if anything essential is missing so deploys surface issues early
if not deepgram_api_key:
    logger.error("DEEPGRAM_API_KEY is required")
    raise ValueError("DEEPGRAM_API_KEY environment variable is required")

if not google_api_key:
    logger.error("GOOGLE_API_KEY is required")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Validate SIP trunk id
if not outbound_trunk_id:
    logger.error("SIP_OUTBOUND_TRUNK_ID is required")
    raise ValueError("SIP_OUTBOUND_TRUNK_ID environment variable is required")

def save_no_answer_note(phone_number: str, reason: str, dial_info: dict[str, Any] | None = None) -> str | None:
    """Save a JSON call note for no-answer/failed call attempts."""
    try:
        import pathlib
        notes_dir = pathlib.Path(__file__).parent / "call_notes"
        notes_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().isoformat()
        filename = f"call_notes_{phone_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = notes_dir / filename

        safe_dial = dial_info or {}
        note = {
            "phone_number": phone_number,
            "timestamp": timestamp,
            "call_duration": "not_connected",
            "status": "no_answer",
            "patient_info": {
                "name": safe_dial.get("patient_name", ""),
                "appointment_date": "",
                "reason_for_visit": safe_dial.get("notes", ""),
                "medications": "",
                "emergency_contact": "",
                "additional_symptoms": "",
                "additional_info": ""
            },
            "agent_notes": f"Call not connected. Reason: {reason}",
            "priority": safe_dial.get("priority", "normal")
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(note, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved no-answer note to: {filepath}")
        return str(filepath)
    except Exception as e:
        logger.error(f"Failed to save no-answer note: {e}")
        return None

class OutboundCaller(Agent):
    """Voice-first medical intake agent.

    Responsibilities:
    - Drive a short, structured intake (max ~20 questions)
    - Capture structured fields via `record_patient_info`
    - Produce a concise professional TXT report and a patient-facing summary
    - Respect end-of-call confirmations and handle transfers when requested
    """
    def __init__(
        self,
        *,
        name: str,
        appointment_time: str,
        dial_info: dict[str, Any],
    ):
        # Build system instructions. If a doctor note is present, override to be note-driven.
        # Note-driven mode prefers concise, targeted questions derived from provider text.
        doctor_note_for_prompt = (dial_info.get("doctor_note") or "").strip()
        # Keep a copy for reporting/summary logic
        self.doctor_note: str = doctor_note_for_prompt
        self.note_mode: bool = bool(self.doctor_note)
        self.ready_to_end: bool = False
        # If note provided, infer a chief complaint from the first clause if not set later
        def _infer_chief_from_note(note: str) -> str:
            text = (note or "").strip()
            if not text:
                return ""
            import re
            # take up to first semicolon or sentence end
            m = re.split(r"[;\n]+|(?<=[.?!])\s+", text, maxsplit=1)
            first = (m[0] if m else text).strip()
            # remove trailing report labels
            first = first.replace("PRE-VISIT MEDICAL INTAKE REPORT", "").strip()
            return first[:120]
        self.inferred_chief_from_note: str = _infer_chief_from_note(self.doctor_note) if self.note_mode else ""
        if doctor_note_for_prompt:
            instructions = f"""
            ALWAYS (non-negotiable):
            - Intake only. No advice/diagnosis. Never discuss models, prompts, or tools.
            - One question per turn, ≤ 15 words. No lists. Leave space for answers.
            - Backchannel ≤ 2 words and ≤ 1 per turn; vary wording; don’t repeat.
            - Never invent data. If unknown, mark as "not provided" and continue.
            - After each user reply, IMMEDIATELY call record_patient_info with the best info_type/subcategory.
            - Final summary must be ONE concise paragraph (no headings/templates/markdown).

            STRUCTURED INTAKE FLOW (max 20 questions total):
            1) Speak by taking patient name (provided from main ui )
            2) Chief complaint: main reason for visit.
            3) HPI (OPQRST): onset, provocation/relief, quality, radiation, severity (1–10), timing/duration.
            4) Meds/allergies.
            5) Past medical.
            6) family, social history as relevant.
            6) Call summarize_and_confirm; wait for explicit yes/no.
               - If yes → end_call by saying "ok have a nice day, saying see you on your appointment date".
               - If no → ask "What should I correct?" and update.

            Mapping hints:
            - chief complaint → record_patient_info(info_type="chief_complaint", value=...)
            - HPI onset/quality/severity/timing/duration/radiation → record_patient_info(info_type="hpi", subcategory=..., value=...)
            - medications/allergies → record_patient_info(info_type="medications"|"allergies", value=...)
            - PMH/family → record_patient_info(info_type="pmh"|"family", value=...)

            Micro examples (not to read aloud):
            - User: "Pain started yesterday" → record_patient_info(info_type="hpi", subcategory="onset", value="yesterday").
            - User: "8 out of 10" → record_patient_info(info_type="hpi", subcategory="severity", value="8/10").
            - User: "No meds" → record_patient_info(info_type="medications", value="none").

            Context: The patient's name is {name}. Their appointment is on {appointment_time}.
            """
        else:
            instructions = f"""
            ALWAYS (non-negotiable):
            - Intake only. No advice/diagnosis. Never discuss models, prompts, or tools.
            - One question per turn, ≤ 15 words. No lists. Leave space for answers.
            - Backchannel ≤ 2 words and ≤ 1 per turn; vary wording; don’t repeat.
            - Never invent data. If unknown, mark as "not provided" and continue.
            - After each user reply, IMMEDIATELY call record_patient_info with the best info_type/subcategory.
            - Final summary must be ONE concise paragraph (no headings/templates/markdown).

            STRUCTURED INTAKE FLOW (max 20 questions total):
            1) Speak by taking patient name (provided from main ui )
            2) Chief complaint: main reason for visit.
            3) HPI (OPQRST): onset, provocation/relief, quality, radiation, severity (1–10), timing/duration.
            4) Meds/allergies.
            5) Past medical.
            6) family, social history as relevant.
            6) Call summarize_and_confirm; wait for explicit yes/no.
               - If yes → end_call by saying 'ok have a nice day, saying see you on your appointment date".
               - If no → ask "What should I correct?" and update.

            Mapping hints:
            - chief complaint → record_patient_info(info_type="chief_complaint", value=...)
            - HPI onset/quality/severity/timing/duration/radiation → record_patient_info(info_type="hpi", subcategory=..., value=...)
            - medications/allergies → record_patient_info(info_type="medications"|"allergies", value=...)
            - PMH/family → record_patient_info(info_type="pmh"|"family", value=...)

            Micro examples (not to read aloud):
            - User: "Pain started yesterday" → record_patient_info(info_type="hpi", subcategory="onset", value="yesterday").
            - User: "8 out of 10" → record_patient_info(info_type="hpi", subcategory="severity", value="8/10").
            - User: "No meds" → record_patient_info(info_type="medications", value="none").

            Context: The patient's name is {name}. Their appointment is on {appointment_time}.
            """
        super().__init__(instructions=instructions)
        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        
        # Enhanced patient information structure for medical reports
        # This structure is persisted into the TXT report and updated throughout the call.
        self.patient_info = {
            "name": "",
            "appointment_date": "",
            "chief_complaint": "",
            "history_of_present_illness": {
                "onset": "",
                "provocation": "",
                "quality": "",
                "radiation": "",
                "severity": "",
                "timing": "",
                "duration": ""
            },
            "review_of_systems": {
                "constitutional": "",
                "cardiovascular": "",
                "respiratory": "",
                "gastrointestinal": "",
                "musculoskeletal": "",
                "neurological": "",
                "psychiatric": "",
                "pertinent_negatives": []
            },
            "medications": "",
            "allergies": "",
            "past_medical_history": "",
            "social_history": "",
            "family_history": "",
            "additional_notes": ""
        }
        
        # Template support removed
        self.template_opening = ""
        self.template_required = []
        self.template_followups = []
        self.template_closing = ""
        
        # Track interview progress
        # `question_count` can be used for analytics; `_count_recorded_items` tracks data density.
        self.interview_phase = "identification"
        self.question_count = 0
        self.max_questions = 20
        
        # Store call summary
        self.call_summary = None
        # Connection state
        self.connected = False
        
        # Template mode removed
        self.template_mode = False
        self.template_required_index = 0
        self.template_followup_index = 0
        self.last_question = None
        self.qa_log: list[dict[str, str]] = []
        # Planning guard to prevent tool execution during internal planning
        self.planning_mode: bool = False
        # Confirmation flow state
        self.awaiting_confirmation: bool = False
        self.user_confirmed: bool = False
        self.summary_done: bool = False
        # Identification steps
        self.id_name_done: bool = True
        self.id_chief_done: bool = False
        self.id_appt_done: bool = True
        # Turn-taking guard: prevent multiple questions in one turn
        self.awaiting_user_reply: bool = False

    def _count_recorded_items(self) -> int:
        """Count how many structured fields have been recorded to estimate progress."""
        count = 0
        # Top-level fields
        for key in ["chief_complaint", "medications", "allergies", "past_medical_history", "name", "appointment_date"]:
            val = self.patient_info.get(key)
            if isinstance(val, str) and val.strip():
                count += 1
        # HPI subfields
        for key, val in self.patient_info.get("history_of_present_illness", {}).items():
            if isinstance(val, str) and val.strip():
                count += 1
        # ROS subfields (ignore list of negatives for counting simplicity)
        for key, val in self.patient_info.get("review_of_systems", {}).items():
            if key == "pertinent_negatives":
                continue
            if isinstance(val, str) and val.strip():
                count += 1
        return count

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant
        self.connected = True

    async def hangup(self):
        """Helper function to hang up the call by deleting the room"""

        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(
            api.DeleteRoomRequest(
                room=job_ctx.room.name,
            )
        )

    @function_tool()
    async def transfer_call(self, ctx: RunContext):
        """Transfer the call to a human agent, called after confirming with the user"""

        transfer_to = self.dial_info["transfer_to"]
        if not transfer_to:
            return "cannot transfer call"

        logger.info(f"transferring call to {transfer_to}")

        # let the message play fully before transferring
        await ctx.session.generate_reply(
            instructions="let the user know you'll be transferring them"
        )

        job_ctx = get_job_context()
        try:
            await ctx.api.sip.transfer_sip_participant(
                api.TransferSIPParticipantRequest(
                    room_name=job_ctx.room.name,
                    participant_identity=self.participant.identity,
                    transfer_to=f"tel:{transfer_to}",
                )
            )

            logger.info(f"transferred call to {transfer_to}")
        except Exception as e:
            logger.error(f"error transferring call: {e}")
            await ctx.session.generate_reply(
                instructions="there was an error transferring the call."
            )
            await self.hangup()

    @function_tool()
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        who = self.participant.identity if self.participant else 'unknown'
        logger.info(f"ending the call for {who}")

        # Prevent premature end in note-mode unless a summary has been delivered
        if getattr(self, 'note_mode', False) and not getattr(self, 'ready_to_end', False):
            logger.info("Blocking premature end_call: waiting for summary/confirmation in note-mode")
            await ctx.session.generate_reply(
                instructions="Politely explain you'll continue with a couple of brief questions and summarize before ending."
            )
            return

        # Generate final call summary
        # Persist a last snapshot of collected fields and status prior to hangup.
        final_summary = {
            "patient_info": self.patient_info,
            "call_duration": "completed",
            "status": "call_ended",
            "timestamp": datetime.now().isoformat(),
            
        }
        
        # If no questions were asked and this was note-driven, include a concise explanation
        if getattr(self, 'doctor_note', None) and self._count_recorded_items() == 0:
            final_summary["status"] = "call_ended_no_answers_note_mode"
        
        logger.info(f"Final call summary: {final_summary}")
        
        # Save professional medical report to file
        saved_path = await self.save_medical_report(final_summary)
        # Persist summary to patient DB - removed
        # try:
        #     save_call_summary(Path(__file__).parent / "patients.db", who, self.generate_medical_report(), (self.dial_info or {}).get("doctor_note"))
        # except Exception as e:
        #     logger.warning(f"DB save summary error: {e}")
        
        # let the agent finish speaking
        await ctx.wait_for_playout()

        await self.hangup()

    def get_call_summary(self):
        """Get the current call summary"""
        if self.call_summary:
            return self.call_summary
        else:
            # Generate a basic summary from patient info
            return {
                "patient_info": self.patient_info.copy(),
                "call_duration": "in_progress",
                "status": "no_summary_generated",
                "timestamp": datetime.now().isoformat(),
                
            }

    async def save_medical_report(self, call_summary):
        """Save professional medical report to a TXT file"""
        try:
            # Create call_notes directory if it doesn't exist
            import pathlib
            notes_dir = pathlib.Path(__file__).parent / "call_notes"
            notes_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp and phone number
            phone_number = self.participant.identity if self.participant else "unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"medical_report_{phone_number}_{timestamp}.txt"
            filepath = notes_dir / filename
            
            # Generate professional medical report
            medical_report = self.generate_medical_report()
            
            # Create comprehensive text report
            # Prefer a computed question metric if the tracked count is zero
            
            text_report = f"""
{medical_report}

CALL INFORMATION:
Phone Number: {phone_number}
Call Date: {call_summary["timestamp"]}
Call Duration: {call_summary["call_duration"]}
Call Status: {call_summary["status"]}
Report Type: Pre-visit Medical Screening
Priority: Normal

PATIENT SUMMARY:
Name: {call_summary["patient_info"].get("name", "Not provided")}
Appointment Date: {call_summary["patient_info"].get("appointment_date", "Not provided")}
Chief Complaint: {call_summary["patient_info"].get("chief_complaint", "Not specified")}

NOTES:
This medical intake report was generated during a pre-visit screening call. The information collected helps healthcare providers prepare for the patient's appointment by understanding their current symptoms, medical history, and medication needs. All information should be verified during the actual medical visit.

Report generated by AI Medical Intake Specialist
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Save to TXT file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_report.strip())
            
            logger.info(f"Medical report saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving medical report: {e}")
            return None

    def generate_medical_report(self):
        """Generate a professional medical report in paragraph format.
        If a doctor note is present and little/no patient info was collected,
        generate a concise note-driven paragraph instead of the full template.
        """
        # Two modes:
        # 1) Note-driven: summarize conversation fields in one paragraph
        # 2) Standard: construct a narrative paragraph from HPI/ROS/Meds/History
        # If a doctor note is present, always produce a single-paragraph conversation summary
        minimal_info = not (
            self.patient_info.get('chief_complaint')
            or any(self.patient_info['history_of_present_illness'].values())
            or self.patient_info.get('medications')
            or self.patient_info.get('allergies')
            or self.patient_info.get('past_medical_history')
        )
        if self.doctor_note:
            summary_parts: list[str] = []
            if self.patient_info['name']:
                summary_parts.append(f"Patient identified as {self.patient_info['name']}.")
            elif (self.dial_info or {}).get('patient_name'):
                summary_parts.append(f"Patient identified as {(self.dial_info or {}).get('patient_name')}.")
            chief = self.patient_info.get('chief_complaint') or getattr(self, 'inferred_chief_from_note', '')
            if chief:
                summary_parts.append(f"Primary concern is {chief}.")
            hpi = self.patient_info['history_of_present_illness']
            hpi_bits: list[str] = []
            if hpi.get('quality'):
                hpi_bits.append(hpi['quality'])
            if hpi.get('onset'):
                hpi_bits.append(f"onset {hpi['onset']}")
            if hpi.get('severity'):
                hpi_bits.append(f"severity {hpi['severity']}")
            if hpi.get('duration'):
                hpi_bits.append(f"duration {hpi['duration']}")
            if hpi.get('timing'):
                hpi_bits.append(f"timing {hpi['timing']}")
            if hpi.get('radiation'):
                hpi_bits.append(f"radiation {hpi['radiation']}")
            if hpi_bits:
                summary_parts.append("History of present illness: " + ", ".join(hpi_bits) + ".")
            if self.patient_info.get('past_medical_history'):
                summary_parts.append(f"Relevant history includes {self.patient_info['past_medical_history']}.")
            if self.patient_info.get('medications'):
                summary_parts.append(f"Current medications include {self.patient_info['medications']}.")
            if self.patient_info.get('allergies'):
                summary_parts.append(f"Allergies: {self.patient_info['allergies']}.")
            ros = self.patient_info.get('review_of_systems', {})
            ros_bits: list[str] = []
            for key in ['respiratory', 'cardiovascular', 'gastrointestinal', 'musculoskeletal', 'neurological', 'psychiatric']:
                val = ros.get(key)
                if isinstance(val, str) and val.strip():
                    # Clean internal tags like "obstructive_sleep_apnea_risk_factors: "
                    clean_val = val.replace("obstructive_sleep_apnea_risk_factors: ", "").strip()
                    ros_bits.append(f"{key.capitalize()}: {clean_val}")
            if ros_bits:
                summary_parts.append("Review of systems: " + "; ".join(ros_bits) + ".")
            if self.patient_info.get('additional_notes'):
                summary_parts.append(self.patient_info['additional_notes'].replace("past_medical_history (anesthesia_problems): ", "Anesthesia: "))
            if minimal_info and not summary_parts:
                summary_parts.append("No patient responses were captured during this call.")
            # Build paragraph using conversation-derived fields only
            paragraph = " ".join(summary_parts).strip()
            import textwrap
            wrapped_paragraph = textwrap.fill(paragraph, width=92)

            effective_q = max(self.question_count, self._count_recorded_items())
            report = f"""
PRE-VISIT MEDICAL INTAKE REPORT

Note-Driven Conversation Summary:
{wrapped_paragraph}
            """
            return report.strip()
        # Incorporate QA log if present (template mode)
        qa_section = ""
        if getattr(self, 'qa_log', None):
            pairs = [f"- {p.get('question')}: {p.get('answer')}" for p in self.qa_log]
            qa_section = "\nTemplate Q&A Summary:\n" + "\n".join(pairs)
        
        # Build a narrative HPI paragraph matching the requested style
        onset = (self.patient_info['history_of_present_illness']['onset'] or '').strip()
        quality = (self.patient_info['history_of_present_illness']['quality'] or '').strip()
        severity = (self.patient_info['history_of_present_illness']['severity'] or '').strip()
        timing = (self.patient_info['history_of_present_illness']['timing'] or '').strip()
        radiation = (self.patient_info['history_of_present_illness']['radiation'] or '').strip()
        duration = (self.patient_info['history_of_present_illness']['duration'] or '').strip()
        chief_local = (self.patient_info['chief_complaint'] or self.inferred_chief_from_note).strip()
        # Pull constitutional/ROS for supportive symptoms and denials
        ros_map_local = self.patient_info.get('review_of_systems', {}) or {}
        constitutional = (ros_map_local.get('constitutional') or '').strip()
        respiratory_ros = (ros_map_local.get('respiratory') or '').strip().lower()
        cardiovascular_ros = (ros_map_local.get('cardiovascular') or '').strip().lower()

        # Sentences
        hpi_sentences: list[str] = []
        if onset or duration:
            onset_phrase = f"that symptoms began {onset}" if onset else "that symptoms have recently begun"
            dur_phrase = f"and have persisted for approximately {duration}" if duration else ""
            hpi_sentences.append(f"The patient reports {onset_phrase} {dur_phrase}.".replace("  ", " ").strip())
        # Severity and quality
        if severity or quality:
            sev_text = ''
            if severity:
                sev_val = severity if any(ch.isdigit() for ch in severity) else severity
                sev_text = f"with an intensity estimated at {sev_val} on a 10‑point scale"
            qual_text = f"The discomfort is described as {quality}" if quality else ''
            if qual_text and sev_text:
                hpi_sentences.append(f"{qual_text}, {sev_text}.")
            elif qual_text:
                hpi_sentences.append(f"{qual_text}.")
            elif sev_text:
                hpi_sentences.append(f"The discomfort is {sev_text}.")
        # Associated symptoms from constitutional/quality
        assoc_bits: list[str] = []
        for token in ["fatigue", "chill", "weakness", "malaise"]:
            if token in (constitutional or '').lower() or token in quality.lower():
                assoc_bits.append(token + ('' if token.endswith('s') else 's') if token == 'chill' else token)
        if assoc_bits:
            pretty = ", ".join(sorted(set(assoc_bits))).replace('chills', 'chills')
            hpi_sentences.append(f"In addition, the patient notes {pretty}, contributing to an overall sense of weakness.")
        # Course / timing and radiation
        if timing:
            hpi_sentences.append(f"The symptoms follow a fluctuating course, with {timing}.")
        if radiation:
            hpi_sentences.append(f"Pain/radiation noted: {radiation}.")
        # Denials from ROS
        denial_list: list[str] = []
        if 'chest pain' in cardiovascular_ros or 'negative for chest pain' in cardiovascular_ros or 'no chest pain' in cardiovascular_ros:
            denial_list.append('chest pain')
        if 'dyspnea' in respiratory_ros or 'shortness of breath' in respiratory_ros or 'no shortness of breath' in respiratory_ros or 'negative for dyspnea' in respiratory_ros:
            denial_list.append('shortness of breath')
        if denial_list:
            both = ' or '.join(denial_list)
            hpi_sentences.append(f"Despite symptom severity, the patient denies {both} or other acute cardiopulmonary complaints.")
        # If still empty, anchor with chief complaint
        if not hpi_sentences and chief_local:
            hpi_sentences.append(f"Presenting with {chief_local}.")
        hpi_text = ' '.join(hpi_sentences).strip()

        # Build medical history paragraph (only if present)
        medical_history_parts: list[str] = []
        pmh = (self.patient_info['past_medical_history'] or '').strip()
        fam = (self.patient_info['family_history'] or '').strip()
        if pmh:
            medical_history_parts.append(pmh.rstrip('. '))
        if fam:
            medical_history_parts.append(f"Family history: {fam.rstrip('. ')}")
        medical_history_text = '. '.join(medical_history_parts)

        # Build medications/allergies paragraph (expand with clarity)
        meds = (self.patient_info['medications'] or '').strip()
        algs = (self.patient_info['allergies'] or '').strip()
        medall_text = ''
        if meds and algs:
            medall_text = f"Current medications include {meds.rstrip('. ')}. Allergies: {algs.rstrip('. ')}."
        elif meds:
            medall_text = f"Current medications include {meds.rstrip('. ')}."
        elif algs:
            medall_text = f"Allergies: {algs.rstrip('. ')}."

        # Build review of systems narrative (positives and pertinent negatives)
        ros_map = self.patient_info.get('review_of_systems', {}) or {}
        ros_bits: list[str] = []
        for sys_key in [
            'constitutional', 'respiratory', 'cardiovascular', 'gastrointestinal',
            'musculoskeletal', 'neurological', 'psychiatric'
        ]:
            val = (ros_map.get(sys_key) or '').strip()
            if val:
                ros_bits.append(f"{sys_key.capitalize()}: {val.rstrip('. ')}.")
        negs = ros_map.get('pertinent_negatives') or []
        neg_text = ''
        if isinstance(negs, list) and negs:
            neg_text = "Pertinent negatives: " + ", ".join(negs) + "."
        ros_text = " ".join(ros_bits + ([neg_text] if neg_text else [])).strip()

        # Build one narrative paragraph across all sections
        ui_name_local = (self.dial_info or {}).get('patient_name') or ''
        chief = (self.patient_info['chief_complaint'] or self.inferred_chief_from_note).strip()
        narrative_bits: list[str] = []
        if (self.patient_info['name'] or ui_name_local):
            narrative_bits.append(f"Patient {self.patient_info['name'] or ui_name_local} presents for pre‑visit intake.")
        if self.patient_info['appointment_date']:
            narrative_bits.append(f"Upcoming appointment on {self.patient_info['appointment_date']}.")
        if chief:
            narrative_bits.append(f"Primary concern: {chief.rstrip('. ')}.")
        if hpi_text:
            narrative_bits.append(hpi_text.rstrip())
        if medical_history_text:
            narrative_bits.append(f"Relevant history: {medical_history_text.rstrip('.')}.")
        if medall_text:
            narrative_bits.append(medall_text.rstrip())
        if ros_text:
            narrative_bits.append(f"Review of systems: {ros_text.rstrip()}" )
        if self.patient_info.get('social_history'):
            narrative_bits.append(f"Social history: {self.patient_info['social_history'].rstrip('. ')}.")
        if self.patient_info.get('family_history'):
            narrative_bits.append(f"Family history: {self.patient_info['family_history'].rstrip('. ')}.")
        # Preliminary impression
        impression_bits: list[str] = []
        if onset:
            impression_bits.append("subacute onset")
        if severity:
            impression_bits.append("significant severity")
        if 'fever' in (chief.lower() + ' ' + quality.lower()):
            impression_bits.append("febrile symptoms")
        if impression_bits:
            narrative_bits.append((", ".join(impression_bits)).capitalize() + ". Correlate clinically.")
        narrative = " ".join([s for s in narrative_bits if s]).strip()
        report = "\n".join([
            "PRE-VISIT MEDICAL INTAKE REPORT",
            "",
            narrative or "",
        ])
        return report.strip()

    @function_tool()
    async def record_patient_info(self, ctx: RunContext, info_type: str, value: str, subcategory: str = None):
        """Record specific patient information during the medical interview
        
        Args:
            info_type: Type of information (name, appointment_date, chief_complaint, hpi, ros, medications, allergies, pmh, social, family, additional)
            value: The information provided by the patient
            subcategory: For structured data like HPI or ROS (onset, provocation, quality, etc.)
        """
        # Mapping notes:
        # - Unknown HPI/ROS keys fall back into `additional_notes` so nothing is lost.
        # - Common synonyms are normalized (e.g., description -> quality; dyspnea -> respiratory).
        try:
            if info_type == "name":
                self.patient_info["name"] = value
                self.id_name_done = True
            elif info_type == "appointment_date":
                self.patient_info["appointment_date"] = value
                self.id_appt_done = True
            elif info_type == "chief_complaint":
                self.patient_info["chief_complaint"] = value
                self.id_chief_done = True
            elif info_type == "infection":
                # Map generic infection/fever info into ROS constitutional
                self.patient_info["review_of_systems"]["constitutional"] = value
            elif info_type == "hpi" and subcategory:
                # Map common synonyms and ensure we don't drop unknown subcategories
                mapped = subcategory
                if subcategory == "description":
                    mapped = "quality"
                if mapped in self.patient_info["history_of_present_illness"]:
                    self.patient_info["history_of_present_illness"][mapped] = value
                else:
                    # Store unknown HPI data in additional notes so it's not lost
                    existing = self.patient_info.get("additional_notes", "")
                    joiner = "\n" if existing else ""
                    self.patient_info["additional_notes"] = f"{existing}{joiner}HPI ({subcategory}): {value}"
            elif info_type == "ros" and subcategory:
                if subcategory in self.patient_info["review_of_systems"]:
                    self.patient_info["review_of_systems"][subcategory] = value
                elif subcategory.lower() == "dyspnea":
                    # Map dyspnea to respiratory ROS
                    self.patient_info["review_of_systems"]["respiratory"] = value
            elif info_type == "pertinent_negative":
                self.patient_info["review_of_systems"]["pertinent_negatives"].append(value)
            elif info_type == "medications":
                self.patient_info["medications"] = value
            elif info_type == "allergies":
                self.patient_info["allergies"] = value
            elif info_type == "pmh" or info_type == "past_medical_history":
                self.patient_info["past_medical_history"] = value
            elif info_type == "social":
                self.patient_info["social_history"] = value
            elif info_type == "family":
                self.patient_info["family_history"] = value
            elif info_type == "additional":
                self.patient_info["additional_notes"] = value
            else:
                # Map some common unknown categories to safe fields
                low = (info_type or "").lower()
                if "pre-visit" in low:
                    # ignore availability/consent flags
                    return {"status": "ignored", "message": "availability noted"}
                if "infection" in low or "fever" in low or "post-op" in low or "postop" in low:
                    self.patient_info["review_of_systems"]["constitutional"] = value
                elif "anesthesia" in low:
                    pmh = self.patient_info.get("past_medical_history", "")
                    joiner = "; " if pmh else ""
                    self.patient_info["past_medical_history"] = f"{pmh}{joiner}Anesthesia: {value}".strip()
                elif "sleep" in low or "apnea" in low:
                    # Map OSA-related items to HPI timing/quality and ROS respiratory
                    existing_ros = self.patient_info["review_of_systems"].get("respiratory", "")
                    tag = (subcategory or "OSA").replace("_", " ")
                    # Build human-friendly fragment without internal tags
                    human_item = value if value and value.lower() not in {"yes", "no"} else tag
                    new_ros = (existing_ros + ("; " if existing_ros and human_item else "") + (human_item or "")).strip()
                    self.patient_info["review_of_systems"]["respiratory"] = new_ros
                    # If daytime sleepiness, also reflect in HPI timing
                    if (subcategory or "").lower().startswith("daytime"):
                        self.patient_info["history_of_present_illness"]["timing"] = "daytime sleepiness present"
                elif "chest" in low and "pain" in low:
                    self.patient_info["history_of_present_illness"]["quality"] = (self.patient_info["history_of_present_illness"].get("quality", "") + ("; " if self.patient_info["history_of_present_illness"].get("quality") else "") + value).strip()
                else:
                    # Fallback: stash in additional notes
                    existing = self.patient_info.get("additional_notes", "")
                    joiner = "\n" if existing else ""
                    self.patient_info["additional_notes"] = f"{existing}{joiner}{info_type}{(' ('+subcategory+')') if subcategory else ''}: {value}"
            
            who = self.participant.identity if self.participant else 'unknown'
            logger.info(f"Recorded {info_type}: {value} for {who}")
            return {
                "status": "info_recorded",
                "info_type": info_type,
                "value": value,
                "all_info": self.patient_info
            }
        except Exception as e:
            logger.error(f"Error recording patient info: {e}")
            return {"status": "error", "message": f"Error recording info: {str(e)}"}

    @function_tool()
    async def summarize_and_confirm(self, ctx: RunContext):
        """Generate comprehensive medical summary and ask patient to confirm"""
        # If user already confirmed, don't re-read summary
        if getattr(self, 'user_confirmed', False):
            return {
                "status": "already_confirmed",
                "message": "User has already confirmed the summary"
            }
        
        # If summary already delivered and awaiting confirmation, don't re-read
        if getattr(self, 'summary_done', False) and getattr(self, 'awaiting_confirmation', False):
            return {
                "status": "awaiting_confirmation",
                "message": "Summary already delivered, waiting for user confirmation"
            }
        
        # If summary already delivered, don't repeat speaking it
        if getattr(self, 'summary_done', False):
            return {
                "status": "summary_already_delivered",
                "summary": (self.call_summary or {}).get("medical_report", ""),
                "patient_info": self.patient_info,
                "call_summary": self.call_summary
            }
        # Generate professional medical report
        medical_report = self.generate_medical_report()
        
        # Create patient-friendly summary
        if self.doctor_note:
            # Note-driven: concise summary of collected information
            parts = []
            if self.patient_info['chief_complaint']:
                parts.append(f"Primary concern: {self.patient_info['chief_complaint']}")
            if self.patient_info['past_medical_history']:
                parts.append(f"Medical history: {self.patient_info['past_medical_history']}")
            if self.patient_info['medications']:
                parts.append(f"Medications: {self.patient_info['medications']}")
            if self.patient_info['allergies']:
                parts.append(f"Allergies: {self.patient_info['allergies']}")
            
            if parts:
                summary_text = ". ".join(parts) + "."
                patient_summary = f"I've collected: {summary_text}"
            else:
                patient_summary = "I've captured your details."
        else:
            effective_q = max(self.question_count, self._count_recorded_items())
            onset_txt = self.patient_info['history_of_present_illness']['onset'] or ''
            sev_txt = self.patient_info['history_of_present_illness']['severity'] or ''
            dur_txt = self.patient_info['history_of_present_illness']['duration'] or ''
            parts = [
                f"Primary concern: {self.patient_info['chief_complaint'] or 'not provided'}.",
            ]
            hpi_bits = []
            if onset_txt:
                hpi_bits.append(f"started {onset_txt}")
            if sev_txt:
                hpi_bits.append(f"severity {sev_txt}")
            if dur_txt:
                hpi_bits.append(f"duration {dur_txt}")
            if hpi_bits:
                parts.append("History of illness: " + ", ".join(hpi_bits) + ".")
            parts.append(f"Medical history: {self.patient_info['past_medical_history'] or 'none reported'}.")
            parts.append(f"Current medications: {self.patient_info['medications'] or 'none' }.")
            parts.append(f"Allergies: {self.patient_info['allergies'] or 'none' }.")
            
            patient_summary = " ".join(parts)
        
        who = self.participant.identity if self.participant else 'unknown'
        logger.info(f"Summarizing medical interview for {who}: {self.question_count} questions asked")
        
        # Store the summary for later retrieval
        self.call_summary = {
            "patient_info": self.patient_info.copy(),
            "call_duration": "in_progress",
            "status": "summary_generated",
            "timestamp": datetime.now().isoformat(),
            "medical_report": medical_report
        }
        self.ready_to_end = True
        self.awaiting_confirmation = True
        self.user_confirmed = False
        self.summary_done = True

        # Speak the patient-facing summary aloud so the user hears it
        try:
            await ctx.session.say(patient_summary)
            await ctx.session.say("Is this information correct?")
        except Exception as e:
            logger.error(f"Error speaking confirmation prompt: {e}")
        
        return {
            "status": "summary_ready",
            "summary": patient_summary,
            "patient_info": self.patient_info,
            "call_summary": self.call_summary,
            "question_count": self.question_count
        }

    @function_tool()
    async def save_notes(self, ctx: RunContext):
        """Save current medical report to TXT file"""
        try:
            current_summary = {
                "patient_info": self.patient_info.copy(),
                "call_duration": "in_progress",
                "status": "notes_saved",
                "timestamp": datetime.now().isoformat(),
                "question_count": self.question_count
            }
            
            filepath = await self.save_medical_report(current_summary)
            if filepath:
                return {
                    "status": "notes_saved",
                    "filepath": filepath,
                    "message": "Medical report has been saved as TXT file"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to save medical report"
                }
        except Exception as e:
            logger.error(f"Error in save_notes: {e}")
            return {
                "status": "error",
                "message": f"Error saving notes: {str(e)}"
            }

    @function_tool()
    async def detected_answering_machine(self, ctx: RunContext):
        """Called when the call reaches voicemail. Use this tool AFTER you hear the voicemail greeting"""
        # If not connected to a remote participant yet (e.g., between retries), ignore
        if not self.connected or not self.participant:
            logger.info("voicemail detected during dialing/retry; ignoring until connected")
            return {"status": "ignored_during_retry"}

        try:
            identity = self.participant.identity
            phone_number = identity or self.dial_info.get("phone_number", "unknown")
            logger.info(f"detected answering machine for {phone_number}")

            # Save a voicemail note, then hang up
            save_no_answer_note(phone_number, reason="voicemail", dial_info=self.dial_info)
        finally:
            await self.hangup()

    def update_interview_phase(self, new_phase: str):
        """Update the current interview phase"""
        self.interview_phase = new_phase
        logger.info(f"Interview phase updated to: {new_phase}")

    def increment_question_count(self):
        """Increment internal question counter (for analytics/limits)."""
        self.question_count += 1
        return self.question_count

async def entrypoint(ctx: JobContext):
    """LiveKit Agents worker entrypoint.

    High-level flow:
    1) Connect to the assigned room
    2) Parse metadata (phone number, optional doctor note)
    3) Configure VAD/STT/TTS/LLM stack for the session
    4) Start the agent session first, then dial via SIP with retries
    5) Once participant joins, engage note-driven or standard intro and run
    """
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    # when dispatching the agent, we'll pass it the approriate info to dial the user
    # dial_info is a dict with the following keys:
    # - phone_number: the phone number to dial
    # - transfer_to: the phone number to transfer the call to when requested
    
    metadata_str = ctx.job.metadata
    dial_info = {}
    logger.info(f"Raw metadata: {metadata_str}")
    
    if metadata_str:
        try:
            # Try to parse as JSON first
            import json
            dial_info = json.loads(metadata_str)
            logger.info(f"Parsed JSON metadata: {dial_info}")
        except json.JSONDecodeError:
            # Fallback to string parsing
            logger.info("JSON parsing failed, trying string parsing")
            clean_str = metadata_str.strip('{}')
            for item in clean_str.split(','):
                if ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip().strip('"')
                    value = value.strip().strip('"')
                    dial_info[key] = value
            logger.info(f"Parsed string metadata: {dial_info}")
    
    logger.info(f"Final dial_info: {dial_info}")
    
    # Validate required fields
    logger.info(f"Checking for phone_number in dial_info: {dial_info}")
    if not dial_info.get("phone_number"):
        logger.error(f"No phone_number found in metadata. Available keys: {list(dial_info.keys())}")
        logger.error(f"Metadata string was: {metadata_str}")
        ctx.shutdown()
        return
        
    participant_identity = phone_number = dial_info["phone_number"]
    # Get UI-provided name if available
    ui_name = (dial_info.get("patient_name") or "").strip()
    ui_appt = (dial_info.get("appointment_date") or "").strip()

    # Initialize the professional medical intake agent
    agent = OutboundCaller(
        name="Patient",  # Generic name since we'll get it from the patient
        appointment_time="upcoming appointment",
        dial_info=dial_info,
    )

    # Determine doctor-note mode early (affects session config)
    doctor_note = (dial_info.get("doctor_note") or "").strip()
    doctor_note_mode = bool(doctor_note)

    # Optimized session configuration for minimal delays
    # Tune VAD/STT endpointing carefully to balance latency with natural turns.
    session = AgentSession(
        vad=silero.VAD.load(
            activation_threshold=0.25,
            min_speech_duration=0.08,
            min_silence_duration=0.25,
            prefix_padding_duration=0.06,
            max_buffered_speech=20.0,
            force_cpu=False,
        ),
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
            endpointing_ms=85,
        ),
        tts=deepgram.TTS(
            #model="aura-asteria-en",
            model="aura-luna-en",
        ),
        llm=google.LLM(
            model="gemini-1.5-flash",  # Supports function calling with higher quotas
            temperature=0.7,  # Slightly lower for more consistent medical interviewing
            api_key=google_api_key,
        ),
        # Disable preemptive generation to ensure agent waits for user responses
        preemptive_generation=False,
    )

    # Add event handlers for better conversation flow
    note_questions: list[str] = []
    note_q_idx: int = 0
    started_questions: bool = False
    fallback_task = None

    def _questions_from_note(note: str) -> list[str]:
        # Simple heuristic conversion of provider note into questions without LLM
        text = (note or "").strip()
        # Sanitize: remove obvious template headings and duplicate words
        for junk in ["PRE-VISIT MEDICAL INTAKE REPORT", "Report Details:"]:
            text = text.replace(junk, " ")
        if not text:
            return []
        # Split by semicolons or periods for clauses
        import re
        parts = [p.strip() for p in re.split(r"[;\n]+|(?<=[.])\s+", text) if p.strip()]
        questions: list[str] = []
        for p in parts:
            low = p.lower()
            if not p:
                continue
            # Map common phrases to concise questions
            if "functional capacity" in low or "climb" in low:
                questions.append("Can you climb two flights of stairs or carry groceries without symptoms?")
            elif "heart" in low or "lung" in low or "kidney" in low:
                questions.append("Do you have a history of heart, lung, or kidney disease?")
            elif "chest pain" in low or "dyspnea" in low or "shortness of breath" in low:
                questions.append("Do you get chest pain or shortness of breath with exertion?")
            elif "sleep apnea" in low or "obstructive" in low:
                questions.append("Have you been told you have obstructive sleep apnea or do you snore loudly and stop breathing during sleep?")
            elif "anesthesia" in low:
                questions.append("Have you had any problems with anesthesia in the past?")
            elif "anticoagulant" in low or "antiplatelet" in low or "sglt2" in low or "current meds" in low or "meds" in low:
                questions.append("What prescription medications are you taking now, including any blood thinners or SGLT2 inhibitors?")
            elif "infection" in low or "fever" in low:
                questions.append("Have you had any recent infections or fevers?")
            elif "allerg" in low:
                questions.append("Do you have any medication or latex allergies?")
            else:
                # Fallback to a short, neutral question to avoid reading the note verbatim
                questions.append("Based on your doctor’s note, what symptoms are you experiencing right now?")
        # Deduplicate and cap to 12
        seen = set()
        final: list[str] = []
        for q in questions:
            if q.lower() not in seen:
                final.append(q)
                seen.add(q.lower())
            if len(final) >= 12:
                break
        return final

    async def _start_note_questions_if_needed():
        nonlocal started_questions, note_q_idx
        if started_questions or not (doctor_note_mode and note_questions):
            return
        started_questions = True
        # Personalized intro if name provided
        if ui_name:
            await session.say(f"Hello {ui_name}, this is your pre-visit intake assistant. I'm here to collect important medical information to help prepare for your appointment. Is now a good time to talk?")
        else:
            await session.say("Hello, this is your pre-visit intake assistant. I'm here to collect important medical information to help prepare for your appointment. Is now a good time to talk?")
        await asyncio.sleep(0.05)

    @session.on("user_message")
    def on_user_message(message):
        logger.info(f"User said: {message.text}")
        # User replied; allow next question
        agent.awaiting_user_reply = False
        
        # Meta-questions: always stay in intake scope
        meta = message.text.strip().lower()
        # End-intent detection at any time
        end_set = {"end call", "hang up", "goodbye", "end the call", "finish", "we are done", "that's all", "no that's all", "no its all", "no it's all"}
        if any(phrase in meta for phrase in end_set):
            async def _respect_end():
                await session.say("Thank you. I'll end the call now.")
                await session.call_tool("end_call")
            asyncio.create_task(_respect_end())
            return

        # If awaiting confirmation after summary, handle yes/no or repeat
        if getattr(agent, 'awaiting_confirmation', False):
            yes_set = {"yes", "yep", "yeah", "correct", "that's correct", "its correct", "it's correct", "ok", "okay", "k"}
            no_set = {"no", "nope", "not correct", "needs changes", "change", "adjust", "incorrect"}
            repeat_set = {"repeat", "say again", "again", "could you repeat", "repeat it", "repeat summary"}
            text_low = meta
            if any(w in text_low for w in yes_set):
                agent.user_confirmed = True
                agent.awaiting_confirmation = False
                async def _end_after_confirm():
                    await session.say("Thank you. I'll end the call now.")
                    # Call the tool so it receives the proper RunContext
                    await session.call_tool("end_call")
                asyncio.create_task(_end_after_confirm())
                return
            if any(w in text_low for w in no_set):
                agent.user_confirmed = False
                agent.awaiting_confirmation = False
                async def _adjust():
                    await session.say("No problem. Please tell me the corrections now.")
                    agent.summary_done = False
                asyncio.create_task(_adjust())
                return
            if any(w in text_low for w in repeat_set):
                async def _repeat_summary():
                    try:
                        await session.say((agent.call_summary or {}).get("medical_report", "I'll repeat the summary now."))
                        await session.say("Is this information correct?")
                    except Exception:
                        pass
                asyncio.create_task(_repeat_summary())
                return

        # If confirmation was already handled, don't process with LLM
        if getattr(agent, 'confirmation_handled', False):
            agent.confirmation_handled = False  # Reset for next message
            return

        # If awaiting confirmation, don't run simple_sequence to prevent LLM interference
        if getattr(agent, 'awaiting_confirmation', False):
                return

        # Capture name if provided and not set
        if not agent.patient_info.get("name"):
            potential = (message.text or "").strip()
            if 2 <= len(potential.split()) <= 5 and any(ch.isalpha() for ch in potential):
                agent.patient_info["name"] = potential
                logger.info(f"Captured name: {potential}")
        
        # Capture appointment date if provided and not set
        if not agent.patient_info.get("appointment_date"):
            potential = (message.text or "").strip()
            # Simple date detection - look for common date patterns
            if any(word in potential.lower() for word in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]) or any(char.isdigit() for char in potential):
                agent.patient_info["appointment_date"] = potential
                logger.info(f"Captured appointment date: {potential}")

        
        # SIMPLE SEQUENCE - no LLM interference
        # This path enforces single-question turns and uses either note-derived
        # questions or asks the LLM to propose exactly one safe medical question.
        async def simple_sequence():
            # Guard against asking more than one question before a reply
            if agent.awaiting_user_reply:
                return
            # Name and appointment are provided via UI; skip identity prompts
            
            # Ask for main concern if not done
            if not agent.id_chief_done:
                agent.id_chief_done = True
                logger.info("Asking for main concern")
                await session.say("What is the main reason for your visit today?")
                agent.awaiting_user_reply = True
                return
            
            # Step 4: Continue with medical questions
            if doctor_note_mode and note_questions:
                if note_q_idx < len(note_questions):
                    q = note_questions[note_q_idx]
                    note_q_idx += 1
                    
                    agent.last_question = q
                    agent.increment_question_count()
                    await asyncio.sleep(0.05)
                    # Ensure single-part phrasing
                    await session.say(q)
                    agent.awaiting_user_reply = True
                    return
                # Finished questions - summarize
                if agent._count_recorded_items() >= 3:
                    await session.call_tool("summarize_and_confirm")
                else:
                    await session.say("Before I summarize, is there anything else important you want your provider to know?")
                agent.awaiting_user_reply = True
                return
            else:
                # Standard mode - ask one medical question
                await session.generate_reply(instructions=(
                    "Ask exactly ONE medically appropriate question for a pre-visit intake. "
                    "Do not combine topics. Keep ≤ 15 words. "
                    "ABSOLUTELY DO NOT say or display: 'record patient info', 'InfoType', 'Category', 'subcategory', 'Value', 'tool', 'function', 'call tool', 'function call'. "
                    "If you need to capture data, silently call record_patient_info and then ask the next question."
                ))
                agent.increment_question_count()
                agent.awaiting_user_reply = True
                return

        asyncio.create_task(simple_sequence())
    
    # Add error handling for rate limits and other errors
    @session.on("error")
    def on_error(error):
        if "429" in str(error) or "quota" in str(error).lower():
            logger.warning("Rate limit hit, waiting before retry...")
            # The agent will automatically retry after a delay
        elif "function calling is not enabled" in str(error).lower():
            logger.error("Model doesn't support function calling - this will break the agent")
        else:
            logger.error(f"Session error: {error}")
    
    # Add connection status monitoring
    @session.on("connected")
    def on_connected():
        logger.info("Agent session connected successfully")
    
    @session.on("disconnected")
    def on_disconnected():
        logger.info("Agent session disconnected")

    # Start the session first before dialing so we don't miss the first utterance
    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(
                # enable Krisp background voice and noise removal
                noise_cancellation=noise_cancellation.BVCTelephony(),
            ),
        )
    )
    # `create_sip_participant` starts dialing the user with retry on timeout
    dial_success = False
    # Allow per-call retry override via metadata
    try:
        per_call_retry = int(str((dial_info or {}).get("retry_count", (dial_info or {}).get("retries", outbound_retry_count))))
    except Exception:
        per_call_retry = outbound_retry_count
    try:
        per_call_delay = int(str((dial_info or {}).get("retry_delay", outbound_retry_delay_s)))
    except Exception:
        per_call_delay = outbound_retry_delay_s
    total_attempts = max(1, per_call_retry + 1)
    for attempt_idx in range(1, total_attempts + 1):
        try:
            logger.info(
                f"Dial attempt {attempt_idx}/{total_attempts} to {phone_number} with timeout {outbound_call_timeout_s}s"
            )
            await asyncio.wait_for(
                ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                # function blocks until user answers the call, or if the call fails
                wait_until_answered=True,
            )
                ),
                timeout=outbound_call_timeout_s,
            )
            dial_success = True
            break
        except asyncio.TimeoutError:
            logger.warning(
                f"Attempt {attempt_idx} timed out after {outbound_call_timeout_s}s (no answer)"
            )
            if attempt_idx < total_attempts:
                logger.info(f"Retrying in {per_call_delay}s...")
                await asyncio.sleep(per_call_delay)
                continue
            # Final failure after retries: persist a JSON note for traceability
            save_no_answer_note(phone_number, reason="timeout", dial_info=dial_info)
            ctx.shutdown()
            return
        except api.TwirpError as e:
            logger.error(
                f"error creating SIP participant: {e.message}, "
                f"SIP status: {e.metadata.get('sip_status_code')} "
                f"{e.metadata.get('sip_status')}"
            )
            reason = f"SIP error {e.metadata.get('sip_status_code')} {e.metadata.get('sip_status')}"
            save_no_answer_note(phone_number, reason=reason, dial_info=dial_info)
            ctx.shutdown()
            return

    if not dial_success:
        return

    # Wait for the agent session start and participant join
    await session_started
    participant = await ctx.wait_for_participant(identity=participant_identity)
    logger.info(f"participant joined: {participant.identity}")

    agent.set_participant(participant)

    # Egress/recording removed
    egress_id = None

    # Set UI-provided identifiers
    if ui_name:
        agent.patient_info["name"] = ui_name
    if ui_appt:
        agent.patient_info["appointment_date"] = ui_appt

    if doctor_note_mode:
        logger.info("Starting doctor-note-driven interview")
        # Build question plan locally without invoking LLM audio/tools
        agent.planning_mode = True
        try:
            note_questions = _questions_from_note(doctor_note)
        finally:
            agent.planning_mode = False
        if note_questions:
            # Wait for first user utterance, or fallback after 2s
            async def _fallback_start():
                await asyncio.sleep(2.0)
                await _start_note_questions_if_needed()
            fallback_task = asyncio.create_task(_fallback_start())
        else:
            # Fallback to normal opening if planning failed
            logger.info("Doctor note plan empty; falling back to normal intro")
            await session.say("Hello, this is your pre-visit intake assistant. I'm here to collect important medical information to help prepare for your appointment. Is now a good time to talk?")
    else:
        # Start the conversation (normal)
        logger.info("Starting professional medical intake interview")
        if ui_name:
            await session.say(f"Hello {ui_name}, this is your pre-visit intake assistant. I'm here to collect important medical information to help prepare for your appointment. Is now a good time to talk?")
        else:
            await session.say("Hello, this is your pre-visit intake assistant. I'm here to collect important medical information to help prepare for your appointment. Is now a good time to talk?")
    # Register a best-effort stop on disconnect
    @session.on("disconnected")
    def _on_disc():
        # No egress to stop anymore
        return


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )