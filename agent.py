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
google_api_key = os.getenv("GOOGLE_API_KEY")

# Validate required environment variables
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
            You are a pre-visit hospital call agent. Keep calls thorough but focused - aim for 3-4 minutes maximum.
            
            CRITICAL: You have conversation memory. Use the record_patient_info tool IMMEDIATELY when patient provides information.
            Don't ask for information they've already given you.
            
            Collect these essential items:
            1. Patient name and appointment date
            2. Main reason for visit (symptoms)
            3. Current medications
            4. Emergency contact name and phone
            
            ADDITIONAL SYMPTOMS & INFORMATION:
            - After getting the main reason for visit, ask: "Are there any other symptoms or health concerns you'd like to mention?"
            - Listen carefully and record any additional symptoms using record_patient_info with info_type "additional_symptoms"
            - Ask: "Is there anything else you think would be helpful for the doctor to know before your visit?"
            - Record any additional information using record_patient_info with info_type "additional_info"
            
            FORBIDDEN TOPICS - DO NOT ASK ABOUT:
            - Family medical history
            - Insurance information
            - Previous surgeries
            - Special accommodations
            - Any other information not in the essential items
            
            WORKFLOW:
            - Greet briefly
            - Ask for name and appointment date
            - Ask for main symptoms/reason for visit
            - Ask for additional symptoms/concerns
            - Ask for current medications
            - Ask for emergency contact
            - Ask if there's anything else to share
            - Use summarize_and_confirm tool
            - End call after confirmation
            
            Be professional but conversational. Use record_patient_info tool after each answer.
            If the patient wants to speak to a human agent, use the transfer_call tool.
            
            The patient's name is {name}. Their appointment is on {appointment_time}.
            """
        )
        # keep reference to the participant for transfers
        self.participant: rtc.RemoteParticipant | None = None
        self.dial_info = dial_info
        
        # Store patient information during the call
        self.patient_info = {
            "name": "",
            "appointment_date": "",
            "reason_for_visit": "",
            "medications": "",
            "emergency_contact": "",
            "additional_symptoms": "",
            "additional_info": ""
        }
        
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
            await job_ctx.api.sip.transfer_sip_participant(
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
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Final call summary: {final_summary}")
        
        # Save call notes to file
        await self.save_call_notes(final_summary)
        
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
                "timestamp": datetime.now().isoformat()
            }

    async def save_call_notes(self, call_summary):
        """Save call notes to a JSON file"""
        try:
            # Create call_notes directory if it doesn't exist
            import pathlib
            notes_dir = pathlib.Path(__file__).parent / "call_notes"
            notes_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp and phone number
            phone_number = self.participant.identity if self.participant else "unknown"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"call_notes_{phone_number}_{timestamp}.json"
            filepath = notes_dir / filename
            
            # Prepare call notes data
            call_notes = {
                "phone_number": phone_number,
                "timestamp": call_summary["timestamp"],
                "call_duration": call_summary["call_duration"],
                "status": call_summary["status"],
                "patient_info": call_summary["patient_info"],
                "agent_notes": f"Call completed successfully. Patient: {call_summary['patient_info'].get('name', 'Unknown')}",
                "priority": "normal"
            }
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(call_notes, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Call notes saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving call notes: {e}")
            return None

    @function_tool()
    async def record_patient_info(self, ctx: RunContext, info_type: str, value: str):
        """Record specific patient information during the call
        
        Args:
            info_type: Type of information (name, appointment_date, reason_for_visit, medications, emergency_contact, additional_symptoms, additional_info)
            value: The information provided by the patient
        """
        if info_type in self.patient_info:
            self.patient_info[info_type] = value
            logger.info(f"Recorded {info_type}: {value} for {self.participant.identity}")
            return {
                "status": "info_recorded",
                "info_type": info_type,
                "value": value,
                "all_info": self.patient_info
            }
        else:
            logger.warning(f"Unknown info_type: {info_type}")
            return {"status": "error", "message": f"Unknown info_type: {info_type}"}

    @function_tool()
    async def summarize_and_confirm(self, ctx: RunContext):
        """Summarize all collected information and ask patient to confirm"""
        summary = f"""
        Let me summarize what we've collected:
        
        Name: {self.patient_info['name'] or 'Not provided'}
        Appointment Date: {self.patient_info['appointment_date'] or 'Not provided'}
        Reason for Visit: {self.patient_info['reason_for_visit'] or 'Not provided'}
        Additional Symptoms: {self.patient_info['additional_symptoms'] or 'None mentioned'}
        Current Medications: {self.patient_info['medications'] or 'Not provided'}
        Emergency Contact: {self.patient_info['emergency_contact'] or 'Not provided'}
        Additional Information: {self.patient_info['additional_info'] or 'None provided'}
        
        Is this information correct? If yes, I'll end the call. If no, please let me know what needs to be corrected.
        """
        
        logger.info(f"Summarizing call for {self.participant.identity}: {self.patient_info}")
        
        # Store the summary for later retrieval
        self.call_summary = {
            "patient_info": self.patient_info.copy(),
            "call_duration": "in_progress",
            "status": "summary_generated",
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "summary_ready",
            "summary": summary,
            "patient_info": self.patient_info,
            "call_summary": self.call_summary
        }

    @function_tool()
    async def save_notes(self, ctx: RunContext):
        """Save current call notes to file"""
        try:
            current_summary = {
                "patient_info": self.patient_info.copy(),
                "call_duration": "in_progress",
                "status": "notes_saved",
                "timestamp": datetime.now().isoformat()
            }
            
            filepath = await self.save_call_notes(current_summary)
            if filepath:
                return {
                    "status": "notes_saved",
                    "filepath": filepath,
                    "message": "Call notes have been saved"
                }
            else:
                return {
                    "status": "error",
                    "message": "Failed to save call notes"
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

    # Initialize the pre-visit screening agent
    agent = OutboundCaller(
        name="Patient",  # Generic name since we'll get it from the patient
        appointment_time="upcoming appointment",
        dial_info=dial_info,
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=google.STT(languages=["en-US"]),
        tts=google.TTS(language="en-US"),
        llm=google.LLM(
            model="gemini-2.0-flash-exp",
            temperature=0.8,
            api_key=google_api_key,
        ),
        # Allow the LLM to generate a response while waiting for the end of turn
        preemptive_generation=True,
    )

    # Add event handlers for better conversation flow
    @session.on("user_message")
    def on_user_message(message):
        logger.info(f"User said: {message.text}")
        # Generate a response to what the user said
        asyncio.create_task(
            session.generate_reply(
                instructions=f"Respond briefly to: '{message.text}'. Ask only the next essential question. Keep responses under 15 seconds."
            )
        )

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

        # Start the conversation with an initial greeting
        logger.info("Starting conversation with initial greeting")
        await session.generate_reply(
            instructions="Say briefly: 'Hi, this is your pre-visit screening call. I'm here to collect some information to help prepare for your appointment. Can you confirm your name?' Keep it under 15 seconds and be friendly."
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