<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Medical Intake AI Agent System

A professional AI-powered medical intake system that conducts pre-visit screening calls and generates comprehensive medical reports. Built with LiveKit Agents, Deepgram STT/TTS, and Google Gemini LLM.

## 🏥 Features

### **Professional Medical Interview Protocol**
- **Structured 4-Section Reports**: Primary concern, HPI, medical history, medications
- **Maximum 20 Questions**: Efficient, focused medical interviews
- **OPQRST Framework**: Systematic symptom assessment
- **Medical Terminology**: Professional healthcare language
- **Pertinent Negatives**: Captures what patients DON'T have

### **Advanced AI Capabilities**
- **Real-time Speech Recognition**: Deepgram STT with Nova-3 model
- **Natural Voice Generation**: Deepgram TTS with Aura-Asteria model
- **Medical Reasoning**: Google Gemini 1.5 Flash for clinical logic
- **Voice Activity Detection**: Optimized Silero VAD for minimal delays
- **Background Noise Cancellation**: Krisp telephony-grade audio processing

### **Professional Report Generation**
- **Clean TXT Format**: No markdown, healthcare provider friendly
- **Structured Data**: Organized by medical categories
- **Call Details**: Complete call metadata and timestamps
- **Patient Summaries**: Concise information for quick review
- **EHR Ready**: Easy integration with electronic health records

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- LiveKit account and credentials
- Deepgram API key
- Google Gemini API key
- SIP trunk for outbound calling

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Musafarz360/outbound-caller-python.git
   cd outbound-caller-python
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your API keys
   ```

4. **Configure credentials**
   ```bash
   cp google-credentials.example.json google-credentials.json
   # Add your Google Cloud service account credentials
   ```

### **Environment Variables**
```env
# LiveKit Configuration
LIVEKIT_URL=wss://your-livekit-instance.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# SIP Configuration
SIP_OUTBOUND_TRUNK_ID=your_sip_trunk_id

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=google-credentials.json
GOOGLE_API_KEY=your_gemini_api_key

# Deepgram
DEEPGRAM_API_KEY=your_deepgram_api_key
```

## 📱 Usage

### **Starting the System**

1. **Start the AI Agent**
   ```bash
   python agent.py dev
   ```

2. **Start the Web Interface**
   ```bash
   python main.py
   ```

3. **Access the Web UI**
   - Open: `http://localhost:5000`
   - Make immediate calls or schedule future calls

### **Making Medical Intake Calls**

1. **Immediate Call**
   - Enter patient phone number
   - Add patient name and appointment date
   - Set priority level
   - Add notes if needed
   - Click "Start Call"

2. **Scheduled Call**
   - Switch to "Schedule Call" tab
   - Set date and time
   - Fill in patient details
   - System will automatically call at scheduled time

### **During the Call**

The AI agent will:
- **Greet professionally** and confirm patient identity
- **Collect chief complaint** and main symptoms
- **Use OPQRST framework** for systematic assessment
- **Record all information** using structured data collection
- **Ask focused questions** (maximum 20)
- **Generate comprehensive summary** for confirmation
- **End call professionally** after confirmation

## 📊 Generated Reports

### **Report Structure**
```
PRE-VISIT MEDICAL INTAKE REPORT

Patient Information:
Patient Name: [Name]
Appointment Date: [Date]
Report Date: [Timestamp]

Primary Concern:
[Chief complaint]

History of Present Illness (HPI):
[Comprehensive symptom details]

Relevant Medical History:
[Past conditions and family history]

Current Medications and Allergies:
[Current meds and allergies]

Interview Summary:
[Purpose and context]

Report Details:
Questions Asked: [Number]
Interview Status: [Phase]
```

### **Report Location**
- **Directory**: `call_notes/`
- **Format**: `medical_report_[phone]_[timestamp].txt`
- **Content**: Professional medical reports ready for healthcare providers

## 🔧 Configuration

### **VAD Optimization**
```python
vad=silero.VAD.load(
    activation_threshold=0.25,      # Faster detection
    min_speech_duration=0.03,      # 30ms minimum
    min_silence_duration=0.15,     # 150ms silence
    prefix_padding_duration=0.05,  # 50ms padding
    max_buffered_speech=20.0,      # Reduced buffer
    force_cpu=False,                # GPU acceleration
)
```

### **STT Configuration**
```python
stt=deepgram.STT(
    model="nova-3",           # High accuracy
    language="en-US",         # English
    endpointing_ms=15,        # Fast endpointing
)
```

### **TTS Configuration**
```python
tts=deepgram.TTS(
    model="aura-asteria-en",  # Natural voice
)
```

## 🏗️ Architecture

### **System Components**
- **LiveKit Agents**: Real-time communication framework
- **Deepgram**: Speech-to-text and text-to-speech
- **Google Gemini**: Medical reasoning and interview logic
- **Silero VAD**: Voice activity detection
- **Flask Web UI**: Call management interface
- **SIP Integration**: Outbound calling capabilities

### **Data Flow**
1. **Web UI** → **Flask Backend** → **LiveKit Room**
2. **SIP Call** → **Patient Phone** → **Agent Session**
3. **Voice Input** → **Deepgram STT** → **Gemini LLM**
4. **LLM Response** → **Deepgram TTS** → **Patient**
5. **Data Collection** → **Structured Report** → **TXT File**

## 📋 Medical Interview Protocol

### **Phase 1: Identification & Chief Complaint**
- Confirm patient identity
- Establish main concern

### **Phase 2: History of Present Illness (HPI)**
- **O**: Onset - When did it start?
- **P**: Provocation - What makes it worse/better?
- **Q**: Quality - How would you describe it?
- **R**: Radiation - Does it spread?
- **S**: Severity - Rate 1-10
- **T**: Time - How long does it last?

### **Phase 3: Medical History & Medications**
- Current medications and dosages
- Relevant past conditions
- Allergies and reactions

### **Phase 4: Family & Social History**
- Family medical history
- Recent exposures or travel

## 🚨 Troubleshooting

### **Common Issues**

1. **API Rate Limits**
   - Error: "429 Too Many Requests"
   - Solution: Wait for quota reset or upgrade API tier

2. **Function Calling Errors**
   - Error: "Function calling is not enabled"
   - Solution: Use `gemini-1.5-flash` model

3. **VAD Delays**
   - Issue: Slow speech detection
   - Solution: Adjust VAD parameters in configuration

4. **Connection Issues**
   - Error: "Failed to connect to Deepgram"
   - Solution: Check internet connection and API keys

### **Performance Optimization**
- **GPU Acceleration**: Set `force_cpu=False` for faster VAD
- **VAD Tuning**: Adjust thresholds for your environment
- **Model Selection**: Use appropriate Gemini model for your quota needs

## 🔒 Security & Privacy

### **Data Protection**
- **Local Storage**: Reports stored locally, not in cloud
- **API Keys**: Stored in `.env.local` (not committed to Git)
- **Patient Data**: Encrypted in transit via LiveKit
- **Compliance**: HIPAA-aware data handling

### **Best Practices**
- Never commit `.env.local` or `google-credentials.json`
- Use strong API keys and rotate regularly
- Monitor API usage and quotas
- Secure your LiveKit instance

## 📈 Future Enhancements

### **Planned Features**
- **Multi-language Support**: Spanish, French, etc.
- **EHR Integration**: Direct FHIR API connections
- **Advanced Analytics**: Call quality metrics
- **Mobile App**: iOS/Android companion apps
- **Voice Biometrics**: Patient voice recognition

### **Customization Options**
- **Interview Templates**: Specialty-specific protocols
- **Custom VAD Models**: Domain-specific voice detection
- **Report Formats**: PDF, Word, or custom formats
- **Integration APIs**: Webhook support for external systems

## 🤝 Contributing

### **Development Setup**
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

### **Testing**
```bash
# Run agent tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_agent.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### **Documentation**
- [LiveKit Agents Documentation](https://docs.livekit.io/agents)
- [Deepgram API Reference](https://developers.deepgram.com/)
- [Google Gemini API Guide](https://ai.google.dev/docs)

### **Community**
- **Issues**: [GitHub Issues](https://github.com/Musafarz360/outbound-caller-python/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Musafarz360/outbound-caller-python/discussions)

### **Professional Support**
For enterprise deployments and professional support, contact the development team.

---

**Built with ❤️ for healthcare professionals**

*This system is designed to assist healthcare providers but should not replace professional medical judgment. All information should be verified during actual medical visits.*
