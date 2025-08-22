from __future__ import annotations

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
import pathlib
current_dir = pathlib.Path(__file__).parent
env_path = current_dir / ".env.local"

logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=str(env_path))

# Load all required environment variables
outbound_trunk_id = os.getenv("SIP_OUTBOUND_TRUNK_ID")
deepgram_api_key = os.getenv("GOOGLE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Validate required environment variables
if not deepgram_api_key:
    logger.error("DEEPGRAM_API_KEY is required")
    raise ValueError("DEEPGRAM_API_KEY environment variable is required")

if not google_api_key:
    logger.error("GOOGLE_API_KEY is required")
    raise ValueError("GOOGLE_API_KEY environment variable is required")

class OutboundCaller(Agent):
    def __init__(
        self,
        *,
        name: str,
        appointment_time: str,
        dial_info: dict[str, Any],
    ):
        super().__init__(
            instructions=f"""
            You are a professional medical intake specialist conducting pre-visit screening calls. 
            Follow medical interview protocols and generate comprehensive clinical documentation.
            
            MEDICAL INTERVIEW PROTOCOL - STRUCTURED REPORTING (Maximum 20 questions):
            
            Your responses must follow this EXACT format for the final summary:
            
            ### Primary concern:
            [Collect and list the primary concern/chief complaint the patient is having]
            
            ### History of Present Illness (HPI):
            [Probe deeply to collect comprehensive HPI. Ask about: when it started, progression, triggers, relieving factors, severity, timing, duration, associated symptoms]
            
            ### Relevant Medical History (from EHR):
            [Extract relevant past medical conditions, surgeries, or family history that relates to current complaint]
            
            ### Medications (from EHR and interview):
            [Document current medications, dosages, allergies, and any new medications mentioned]
            
            INTERVIEW STRATEGY:
            - Ask focused questions to fill each section completely
            - Use record_patient_info tool for each piece of information
            - Prioritize information that fits these 4 sections
            - Maximum 20 questions to complete all sections
            - Be thorough but respectful of patient's time
            
            CRITICAL RULES:
            - Ask only ONE question at a time
            - Use medical terminology when appropriate
            - Record ALL information using record_patient_info tool with proper subcategories
            - Focus on gathering information for the 4 structured sections
            - End with summarize_and_confirm when complete
            
            DATA RECORDING INSTRUCTIONS:
            - Use record_patient_info with info_type="chief_complaint" for main concern
            - Use record_patient_info with info_type="hpi" and subcategory="onset" for when it started
            - Use record_patient_info with info_type="hpi" and subcategory="quality" for pain description
            - Use record_patient_info with info_type="hpi" and subcategory="severity" for pain scale
            - Use record_patient_info with info_type="hpi" and subcategory="timing" for when it occurs
            - Use record_patient_info with info_type="hpi" and subcategory="radiation" for pain spread
            - Use record_patient_info with info_type="medications" for current medications
            - Use record_patient_info with info_type="allergies" for allergies
            - Use record_patient_info with info_type="pmh" for past medical history
            - Use record_patient_info with info_type="family" for family history
            
            TOOLS TO USE:
            - record_patient_info: Record each piece of information immediately with proper subcategories
            - summarize_and_confirm: Generate final summary and confirmation
            - end_call: End call after confirmation
            - transfer_call: If patient requests human agent
            
            The patient's name is {name}. Their appointment is on {appointment_time}.
            """
        )
        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        
        # Enhanced patient information structure for medical reports
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
        
        # Track interview progress
        self.interview_phase = "identification"
        self.question_count = 0
        self.max_questions = 20
        
        # Store call summary
        self.call_summary = None

    def set_participant(self, participant: rtc.RemoteParticipant):
        self.participant = participant

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
        logger.info(f"ending the call for {self.participant.identity}")

        # Generate final call summary
        final_summary = {
            "patient_info": self.patient_info,
            "call_duration": "completed",
            "status": "call_ended",
            "timestamp": datetime.now().isoformat(),
            "question_count": self.question_count
        }
        
        logger.info(f"Final call summary: {final_summary}")
        
        # Save professional medical report to file
        await self.save_medical_report(final_summary)
        
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
                "question_count": self.question_count
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
            text_report = f"""
{medical_report}

CALL INFORMATION:
Phone Number: {phone_number}
Call Date: {call_summary["timestamp"]}
Call Duration: {call_summary["call_duration"]}
Call Status: {call_summary["status"]}
Questions Asked: {call_summary["question_count"]}
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
        """Generate a professional medical report in paragraph format"""
        
        # Build HPI paragraph
        hpi_parts = []
        if self.patient_info['history_of_present_illness']['onset']:
            hpi_parts.append(f"The patient reports that symptoms {self.patient_info['history_of_present_illness']['onset']}")
        if self.patient_info['history_of_present_illness']['quality']:
            hpi_parts.append(f"The patient describes the pain as {self.patient_info['history_of_present_illness']['quality']}")
        if self.patient_info['history_of_present_illness']['severity']:
            hpi_parts.append(f"Pain severity is rated {self.patient_info['history_of_present_illness']['severity']}")
        if self.patient_info['history_of_present_illness']['timing']:
            hpi_parts.append(f"The pain {self.patient_info['history_of_present_illness']['timing']}")
        if self.patient_info['history_of_present_illness']['radiation']:
            hpi_parts.append(f"Pain radiates {self.patient_info['history_of_present_illness']['radiation']}")
        
        hpi_text = '. '.join(hpi_parts) if hpi_parts else 'Limited information available about the history of present illness.'
        
        # Build medical history paragraph
        medical_history_parts = []
        if self.patient_info['past_medical_history']:
            medical_history_parts.append(f"Past medical history includes {self.patient_info['past_medical_history']}")
        if self.patient_info['family_history']:
            medical_history_parts.append(f"Family history reveals {self.patient_info['family_history']}")
        
        medical_history_text = '. '.join(medical_history_parts) if medical_history_parts else 'No significant past medical history reported.'
        
        # Build medications paragraph
        medications_parts = []
        if self.patient_info['medications']:
            medications_parts.append(f"Current medications include {self.patient_info['medications']}")
        if self.patient_info['allergies']:
            medications_parts.append(f"The patient reports {self.patient_info['allergies']}")
        
        medications_text = '. '.join(medications_parts) if medications_parts else 'No medications reported.'
        
        report = f"""
PRE-VISIT MEDICAL INTAKE REPORT

Patient Information:
Patient Name: {self.patient_info['name'] or 'Not provided'}
Appointment Date: {self.patient_info['appointment_date'] or 'Not provided'}
Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Primary Concern:
{self.patient_info['chief_complaint'] or 'Not specified'}

History of Present Illness (HPI):
{hpi_text}

Relevant Medical History:
{medical_history_text}

Current Medications and Allergies:
{medications_text}

Interview Summary:
This pre-visit screening was conducted to gather essential medical information. The patient provided details about their current symptoms, medical history, and medications to help prepare for their upcoming appointment.

Report Details:
Questions Asked: {self.question_count}
Interview Status: {self.interview_phase}
        """
        return report.strip()

    @function_tool()
    async def record_patient_info(self, ctx: RunContext, info_type: str, value: str, subcategory: str = None):
        """Record specific patient information during the medical interview
        
        Args:
            info_type: Type of information (name, appointment_date, chief_complaint, hpi, ros, medications, allergies, pmh, social, family, additional)
            value: The information provided by the patient
            subcategory: For structured data like HPI or ROS (onset, provocation, quality, etc.)
        """
        try:
            if info_type == "name":
                self.patient_info["name"] = value
            elif info_type == "appointment_date":
                self.patient_info["appointment_date"] = value
            elif info_type == "chief_complaint":
                self.patient_info["chief_complaint"] = value
            elif info_type == "hpi" and subcategory:
                if subcategory in self.patient_info["history_of_present_illness"]:
                    self.patient_info["history_of_present_illness"][subcategory] = value
            elif info_type == "ros" and subcategory:
                if subcategory in self.patient_info["review_of_systems"]:
                    self.patient_info["review_of_systems"][subcategory] = value
            elif info_type == "pertinent_negative":
                self.patient_info["review_of_systems"]["pertinent_negatives"].append(value)
            elif info_type == "medications":
                self.patient_info["medications"] = value
            elif info_type == "allergies":
                self.patient_info["allergies"] = value
            elif info_type == "pmh":
                self.patient_info["past_medical_history"] = value
            elif info_type == "social":
                self.patient_info["social_history"] = value
            elif info_type == "family":
                self.patient_info["family_history"] = value
            elif info_type == "additional":
                self.patient_info["additional_notes"] = value
            else:
                logger.warning(f"Unknown info_type: {info_type}")
                return {"status": "error", "message": f"Unknown info_type: {info_type}"}
            
            logger.info(f"Recorded {info_type}: {value} for {self.participant.identity}")
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
        
        # Generate professional medical report
        medical_report = self.generate_medical_report()
        
        # Create patient-friendly summary using structured format
        patient_summary = f"""
        Let me summarize what we've collected for your medical visit:
        
        **Primary Concern:** {self.patient_info['chief_complaint'] or 'Not provided'}
        
        **History of Illness:** {self.patient_info['history_of_present_illness']['onset'] or 'Not specified'}{', ' + self.patient_info['history_of_present_illness']['severity'] if self.patient_info['history_of_present_illness']['severity'] else ''}{', ' + self.patient_info['history_of_present_illness']['duration'] if self.patient_info['history_of_present_illness']['duration'] else ''}
        
        **Medical History:** {self.patient_info['past_medical_history'] or 'None reported'}
        **Current Medications:** {self.patient_info['medications'] or 'None reported'}
        **Allergies:** {self.patient_info['allergies'] or 'None reported'}
        
        **Questions Asked:** {self.question_count} out of {self.max_questions}
        
        Is this information complete and accurate? If yes, I'll end the call. If anything needs correction, please let me know.
        """
        
        logger.info(f"Summarizing medical interview for {self.participant.identity}: {self.question_count} questions asked")
        
        # Store the summary for later retrieval
        self.call_summary = {
            "patient_info": self.patient_info.copy(),
            "call_duration": "in_progress",
            "status": "summary_generated",
            "timestamp": datetime.now().isoformat(),
            "question_count": self.question_count,
            "medical_report": medical_report
        }
        
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
        logger.info(f"detected answering machine for {self.participant.identity}")
        await self.hangup()

    def update_interview_phase(self, new_phase: str):
        """Update the current interview phase"""
        self.interview_phase = new_phase
        logger.info(f"Interview phase updated to: {new_phase}")

    def increment_question_count(self):
        """Increment the question counter"""
        self.question_count += 1
        if self.question_count >= self.max_questions:
            logger.info(f"Maximum questions ({self.max_questions}) reached")
        return self.question_count

async def entrypoint(ctx: JobContext):
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

    # Initialize the professional medical intake agent
    agent = OutboundCaller(
        name="Patient",  # Generic name since we'll get it from the patient
        appointment_time="upcoming appointment",
        dial_info=dial_info,
    )

    # Optimized session configuration for minimal delays
    session = AgentSession(
        vad=silero.VAD.load(
            activation_threshold=0.25,  # Even lower threshold for faster detection
            min_speech_duration=0.03,  # Very short minimum speech duration (30ms)
            min_silence_duration=0.15,  # Shorter silence duration (150ms)
            prefix_padding_duration=0.05,  # Minimal prefix padding (50ms)
            max_buffered_speech=20.0,  # Reduced buffer for faster processing
            force_cpu=False,  # Use GPU if available for faster processing
        ),
        stt=deepgram.STT(
            model="nova-3",
            language="en-US",
            endpointing_ms=15,  # Faster endpointing
        ),
        tts=deepgram.TTS(
            model="aura-asteria-en",
        ),
        llm=google.LLM(
            model="gemini-1.5-flash",  # Supports function calling with higher quotas
            temperature=0.7,  # Slightly lower for more consistent medical interviewing
            api_key=google_api_key,
        ),
        # Allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=True,
    )

    # Add event handlers for better conversation flow
    @session.on("user_message")
    def on_user_message(message):
        logger.info(f"User said: {message.text}")
        # Increment question count for each user response
        agent.increment_question_count()
        
        # Generate a response following medical interview protocol
        asyncio.create_task(
            session.generate_reply(
                instructions=f"""
                Respond to: '{message.text}' following medical interview protocol.
                
                Current phase: {agent.interview_phase}
                Questions asked: {agent.question_count}/{agent.max_questions}
                
                - Ask only ONE focused medical question
                - Use record_patient_info tool to capture information
                - Progress through interview phases systematically
                - Keep responses professional and under 15 seconds
                - If approaching 20 questions, prepare to summarize
                """
            )
        )
    
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

    # start the session first before dialing, to ensure that when the user picks up
    # the agent does not miss anything the user says
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
    # `create_sip_participant` starts dialing the user
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                # function blocks until user answers the call, or if the call fails
                wait_until_answered=True,
            )
        )

        # wait for the agent session start and participant join
        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")

        agent.set_participant(participant)

        # Start the conversation with professional medical intake greeting
        logger.info("Starting professional medical intake interview")
        await session.generate_reply(
            instructions="""
            Say briefly and professionally: 
            "Hello, this is your pre-visit medical intake call. I'm here to collect important medical information to help prepare for your appointment. 
            To begin, can you please confirm your full name and the date of your upcoming appointment?"
            
            Keep it professional, friendly, and under 20 seconds. This is the first question of the medical interview.
            """
        )

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )