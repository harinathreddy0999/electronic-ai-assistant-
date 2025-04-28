from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import os
import google.generativeai as genai
import googlemaps
from dotenv import load_dotenv
import uuid
import asyncio # Added for async operations
import logging # Added for better logging
import cv2 # Added OpenCV
import numpy as np # Added NumPy for array manipulation
import re # Import regex for keyword spotting
from sqlalchemy import or_ # For database querying
import json # For handling JSON data from DB
import onnxruntime as ort # Added ONNX Runtime import

# Import aiortc components
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.contrib.media import MediaRelay, MediaPlayer, MediaRecorder # Example utilities
# Import helper for converting aiortc frames to NumPy arrays
from av import VideoFrame

# Import db instance and models
from models import db, Session, Message, Device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Database Configuration ---
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.warning("DATABASE_URL environment variable not set. Database features disabled.")
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# --- CORS Configuration ---
# Apply CORS to the Flask app AND SocketIO
# Allow credentials if needed later for authentication
# Added the current frontend origin (5176) to the allowed list
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5176", "http://127.0.0.1:5176"]
cors = CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)

# --- SocketIO Configuration ---
socketio = SocketIO(app, cors_allowed_origins=ALLOWED_ORIGINS, async_mode='eventlet')

# --- External API Configuration ---
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# Configure Gemini
model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # Use a currently available model name like gemini-1.5-flash-latest
        model = genai.GenerativeModel('gemini-1.5-flash-latest') 
        logger.info("Gemini model initialized (gemini-1.5-flash-latest).")
    except Exception as e:
        logger.error(f"Error configuring Gemini: {e}")
else:
     logger.warning("GOOGLE_API_KEY environment variable not set.")

# Configure Google Maps Client
gmaps = None
if GOOGLE_MAPS_API_KEY:
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    except Exception as e:
        logger.error(f"Error configuring Google Maps Client: {e}")
else:
    logger.warning("GOOGLE_MAPS_API_KEY environment variable not set.")

# --- WebRTC Backend State ---
pcs_by_session = {} # Maps session_id -> {sid -> RTCPeerConnection}
latest_cv_results = {} # Maps session_id -> {sid -> list[detection_results]} 

# --- CV Model Globals & Setup ---
# Define paths for the model files (relative to backend directory)
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models', 'cv')
PROTOTXT_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.prototxt.txt")
MODEL_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_deploy.caffemodel")

# Load the DNN model (load once globally or per track instance? Global is more efficient)
cv_net = None
if os.path.exists(PROTOTXT_PATH) and os.path.exists(MODEL_PATH):
    try:
        logger.info(f"Loading CV model from: {MODEL_DIR}")
        cv_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
        logger.info("MobileNet SSD model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading CV model: {e}", exc_info=True)
        cv_net = None
else:
    logger.warning(f"CV model files not found in {MODEL_DIR}. Object detection disabled.")
    logger.warning(f"Looked for: {PROTOTXT_PATH} and {MODEL_PATH}")

# COCO Class Labels MobileNet SSD was trained on
# (Check your specific model's documentation for correct labels/order)
COCO_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor", "keyboard", "mouse", "laptop", 
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator"]
# Electronics relevant subset (indices based on above list - ADJUST IF NEEDED)
ELECTRONICS_CLASSES = [COCO_CLASSES.index(cls) for cls in ["tvmonitor", "keyboard", "mouse", "laptop", "cell phone"] if cls in COCO_CLASSES]
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence to consider detection
# --- End CV Model Setup ---

# --- Custom Electronics Model (Placeholder) --- 
CUSTOM_MODEL_PATH = os.path.join(MODEL_DIR, "custom_electronics_model.onnx")
custom_model_session = None
custom_model_input_name = None
custom_model_output_names = None
CUSTOM_MODEL_CLASSES = [] # Placeholder - Load from a labels file usually

if os.path.exists(CUSTOM_MODEL_PATH):
    try:
        logger.info(f"Loading custom CV model from: {CUSTOM_MODEL_PATH}")
        # Example using ONNX Runtime
        custom_model_session = ort.InferenceSession(CUSTOM_MODEL_PATH, providers=['CPUExecutionProvider'])
        # Get input/output names (common practice for ONNX)
        custom_model_input_name = custom_model_session.get_inputs()[0].name
        custom_model_output_names = [output.name for output in custom_model_session.get_outputs()]
        logger.info(f"Custom model loaded. Input: {custom_model_input_name}, Outputs: {custom_model_output_names}")
        # TODO: Load CUSTOM_MODEL_CLASSES from a corresponding label file
        # Example: CUSTOM_MODEL_CLASSES = [line.strip() for line in open(os.path.join(MODEL_DIR, "custom_labels.txt")).readlines()]
        CUSTOM_MODEL_CLASSES = ["background", "macbook_pro_14", "iphone_13", "anker_charger", "mx_master_3s", "frayed_cable", "cracked_screen"] # Example placeholder classes
        logger.info(f"Loaded {len(CUSTOM_MODEL_CLASSES)} custom model class labels.")

    except Exception as e:
        logger.error(f"Error loading custom CV model: {e}", exc_info=True)
        custom_model_session = None
else:
    logger.info(f"Custom CV model file not found at {CUSTOM_MODEL_PATH}. Will use fallback if available.")
# --------------------------------------------

# --- Updated Video Track Processor with OpenCV & SocketIO Emit ---
class VideoProcessorTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, track, session_id, client_sid):
        super().__init__()
        self.track = track
        self.session_id = str(session_id) # Ensure session_id is string for dict key
        self.client_sid = client_sid
        self.frame_count = 0
        # Prioritize custom model if loaded
        self.custom_model = custom_model_session 
        self.net = cv_net # Fallback generic model
        logger.info(f"VideoProcessorTrack initialized for {client_sid}. Custom model loaded: {bool(self.custom_model)}")
        if self.session_id not in latest_cv_results:
             latest_cv_results[self.session_id] = {}
        latest_cv_results[self.session_id][self.client_sid] = []

    async def recv(self):
        frame = await self.track.recv()
        self.frame_count += 1
        
        detected_objects_for_frame = []
        model_used = None # Track which model provided results

        try:
            img_bgr = frame.to_ndarray(format="bgr24")
            (h, w) = img_bgr.shape[:2]

            # --- Try Custom Model First --- 
            if self.custom_model and custom_model_input_name:
                model_used = "custom"
                try:
                    # TODO: Preprocessing specific to the custom model
                    # Example: Resize, normalize, change channel order? Depends on training.
                    # Assuming input needs resizing to (640, 640) and normalization [0,1], CHW format
                    target_size = (640, 640)
                    img_resized = cv2.resize(img_bgr, target_size)
                    input_blob = cv2.dnn.blobFromImage(img_resized, scalefactor=1/255.0, size=target_size, swapRB=True, crop=False)
                    
                    # Run inference (ONNX Runtime example)
                    outputs = self.custom_model.run(custom_model_output_names, {custom_model_input_name: input_blob})
                    
                    # TODO: Post-processing specific to the custom model output format
                    # This depends heavily on the model architecture (YOLO, SSD, Faster R-CNN etc.)
                    # Example placeholder assuming output format similar to MobileNet SSD (but with custom classes)
                    # outputs[0] might contain boxes, outputs[1] scores, outputs[2] classes? Or combined?
                    # This part NEEDS to be adapted based on the actual chosen/trained model.
                    logger.debug(f"[CV-Custom-{self.client_sid}] Raw output shapes: {[o.shape for o in outputs]}")
                    # Placeholder post-processing:
                    # Assuming output[0] has shape [1, N, 7] where 7 = [batch_id, class_id, score, x1, y1, x2, y2]
                    if isinstance(outputs, list) and len(outputs) > 0:
                        detections = outputs[0][0] # Assuming first output, first batch
                        for detection in detections:
                            score = detection[2]
                            if score > CONFIDENCE_THRESHOLD: 
                                class_id = int(detection[1])
                                if class_id < len(CUSTOM_MODEL_CLASSES):
                                    label = CUSTOM_MODEL_CLASSES[class_id]
                                    # Rescale box coords if necessary (depends on model output)
                                    box = detection[3:7] * np.array([w, h, w, h]) # Example rescale
                                    detected_objects_for_frame.append({"label": label, "confidence": float(score)})
                                else:
                                     logger.warning(f"Custom model detected class ID {class_id} out of bounds.")

                except Exception as custom_e:
                    logger.error(f"[CV-Custom-{self.client_sid}] Error during custom model processing: {custom_e}", exc_info=False)
                    # Fallback to MobileNet SSD if custom model fails
                    model_used = None 
            # 
            # --- Fallback to MobileNet SSD --- 
            if not detected_objects_for_frame and self.net: # Only run if custom failed or wasn't used
                model_used = "mobilenet_ssd"
                try:
                    blob = cv2.dnn.blobFromImage(cv2.resize(img_bgr, (300, 300)), 0.007843, (300, 300), 127.5)
                    self.net.setInput(blob)
                    detections = self.net.forward()
                    for i in np.arange(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > CONFIDENCE_THRESHOLD:
                            idx = int(detections[0, 0, i, 1])
                            if idx < len(COCO_CLASSES):
                                class_name = COCO_CLASSES[idx]
                                if idx in ELECTRONICS_CLASSES:
                                     detected_objects_for_frame.append({"label": class_name, "confidence": float(confidence)})
                except Exception as mobilenet_e:
                    logger.error(f"[CV-MobileNet-{self.client_sid}] Error processing frame: {mobilenet_e}", exc_info=False)
                    model_used = None
            # ------------------------------------
            
            # Update shared state and emit results if found by either model
            if detected_objects_for_frame:
                if self.session_id in latest_cv_results and self.client_sid in latest_cv_results[self.session_id]:
                    latest_cv_results[self.session_id][self.client_sid] = detected_objects_for_frame
                # Throttle emitting?
                if self.frame_count % 5 == 0: 
                    logger.debug(f"[CV-{model_used}-{self.client_sid}] Emitting: {detected_objects_for_frame}")
                    socketio.emit('cv_results', {'detections': detected_objects_for_frame}, room=self.client_sid)
            else:
                 # Clear latest results if nothing detected this frame
                 if self.session_id in latest_cv_results and self.client_sid in latest_cv_results[self.session_id]:
                      latest_cv_results[self.session_id][self.client_sid] = []

        except Exception as e:
            if self.frame_count % 60 == 0: logger.error(f"[CV-{self.client_sid}] Outer frame processing error: {e}", exc_info=False)

        return frame

    # Add cleanup for the shared state when track stops (if needed)
    async def stop(self):
        await super().stop()
        if self.session_id in latest_cv_results and self.client_sid in latest_cv_results[self.session_id]:
            del latest_cv_results[self.session_id][self.client_sid]
            if not latest_cv_results[self.session_id]:
                 del latest_cv_results[self.session_id]
        logger.info(f"Cleaned up CV results state for {self.client_sid}")

# -----------------------------------------------------------

# --- Helper Functions (DB) ---
def get_session_history_for_gemini(session_id_uuid):
    """Retrieves formatted message history for the Gemini API from DB."""
    session = db.session.get(Session, session_id_uuid) # Use db.session.get for PK lookup
    if not session:
        return []

    history = []
    # Order messages by timestamp to maintain conversation order
    messages_from_db = sorted(session.messages, key=lambda m: m.timestamp)

    for msg in messages_from_db:
        role = 'user' if msg.sender == 'user' else 'model'
        history.append({'role': role, 'parts': [str(msg.text)]})
    return history

def check_user_confirmation(text):
    """Simple check for positive confirmation keywords."""
    text_lower = text.lower()
    confirmation_words = ['yes', 'okay', 'ok', 'done', 'did that', 'yep', 'yeah', 'sure', 'alright']
    # Negative/problem words to potentially stop guide
    problem_words = ['no', "didn't work", 'still broken', 'problem', 'issue', 'error'] 
    
    is_confirm = any(word in text_lower for word in confirmation_words)
    is_problem = any(word in text_lower for word in problem_words)

    if is_problem: return False # User indicated problem
    if is_confirm: return True  # User confirmed positively
    return None # Ambiguous or unrelated response

# --- API Routes (DB) ---
@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Initiates a new troubleshooting session in the database."""
    if not DATABASE_URL:
        return jsonify({"error": "Database not configured"}), 500

    try:
        new_session = Session() # Defaults handle session_id and timestamps
        # TODO: Add user_id if implementing user accounts
        db.session.add(new_session)
        db.session.commit()
        logger.info(f"Session created in DB: {new_session.session_id}")
        return jsonify({"session_id": str(new_session.session_id)}), 201 # Return ID as string
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating session in DB: {e}", exc_info=True)
        return jsonify({"error": "Failed to create session"}), 500

@app.route('/api/sessions/<session_id>/messages', methods=['POST'])
def handle_message(session_id):
    """Handles incoming messages, uses structured LLM calls for diagnostics & guide initiation."""
    # --- Initial Checks & Setup --- 
    if not DATABASE_URL: return jsonify({"error": "DB not configured"}), 500
    if not model: return jsonify({"error": "AI not configured"}), 500
    try: session_id_uuid = uuid.UUID(session_id)
    except ValueError: return jsonify({"error": "Invalid session ID"}), 400
    session = db.session.get(Session, session_id_uuid)
    if not session: return jsonify({"error": "Session not found"}), 404
    if session.status != 'active': return jsonify({"error": "Session not active"}), 400
    data = request.get_json()
    if not data or 'text' not in data: return jsonify({"error": "Missing text"}), 400
    user_text = data['text']
    logger.info(f"Session {session_id} received: {user_text}")

    # --- Stateful Guide Execution Logic --- 
    if session.active_guide_name and session.active_guide_steps and session.current_guide_step:
        logger.info(f"Session {session_id} processing step {session.current_guide_step} of guide '{session.active_guide_name}'")
        
        # Safely parse steps (expecting a JSON list string)
        steps_list = []
        try:
            parsed_steps = json.loads(session.active_guide_steps) if isinstance(session.active_guide_steps, str) else session.active_guide_steps
            if isinstance(parsed_steps, list):
                steps_list = [str(step) for step in parsed_steps] # Ensure steps are strings
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error parsing guide steps for session {session_id}: {e}")
            # Clear bad guide state and let normal flow handle it
            session.active_guide_name = None
            session.active_guide_steps = None
            session.current_guide_step = None
            db.session.add(session)
            # Fall through to normal message handling
        
        if steps_list: # Only proceed if steps were parsed correctly
            total_steps = len(steps_list)
            current_step_index = session.current_guide_step # 1-based
            confirmation = check_user_confirmation(user_text)
            agent_reply = ""
            should_clear_guide = False

            if confirmation is True and current_step_index < total_steps:
                session.current_guide_step += 1
                next_step_text = steps_list[session.current_guide_step - 1]
                agent_reply = f"Great! Step {session.current_guide_step} of {total_steps}: {next_step_text}" 
            elif confirmation is True and current_step_index >= total_steps:
                agent_reply = f"Okay, that was the last step for the '{session.active_guide_name}' guide. Did that resolve the issue, or are you still experiencing problems?"
                should_clear_guide = True 
            elif confirmation is False:
                agent_reply = "Okay, it sounds like that didn't work or there was a problem. I'll stop the current guide. Can you describe what happened or what you see now?" 
                should_clear_guide = True 
            else: 
                current_step_text = steps_list[current_step_index - 1]
                agent_reply = f"Sorry, I need a clearer confirmation for Step {current_step_index} ('{current_step_text[:50]}...'). Did you complete it successfully? (Yes/No)" 
            
            # Save state and messages
            user_message = Message(session_id=session.session_id, sender='user', text=user_text)
            db.session.add(user_message)
            if should_clear_guide:
                logger.info(f"Clearing active guide '{session.active_guide_name}' for session {session_id}")
                session.active_guide_name = None
                session.active_guide_steps = None
                session.current_guide_step = None
            db.session.add(session)
            agent_message = Message(session_id=session.session_id, sender='agent', text=agent_reply)
            db.session.add(agent_message)
            try:
                db.session.commit()
                return jsonify({"reply_text": agent_reply, "diagnostic_state": {'in_guide': not should_clear_guide}})
            except Exception as e:
                 db.session.rollback(); logger.error(f"DB Error (Guide): {e}"); return jsonify({"error": "DB error"}), 500

    # --- If NOT in an active guide, proceed with diagnostics --- 
    logger.info(f"Session {session_id} running diagnostics...")
    
    # --- LLM Call 1: Entity Extraction --- 
    extracted_entities = {"device_brand": None, "device_model": None, "device_type": None, "symptoms": []}
    entity_extraction_prompt = (
        f"Analyze the following user query to identify the electronic device and symptoms mentioned. "
        f"Provide the output as a JSON object with keys: 'device_brand', 'device_model', 'device_type', 'symptoms' (list of strings). "
        f"If a value isn't clearly mentioned, use null or an empty list. Focus only on information in the query.\n\n"
        f"User Query: \"{user_text}\""
        f"\n\nJSON Output:\""
    )
    try:
        logger.info("--- Sending to Gemini for Entity Extraction ---")
        # Use generate_content for single-turn extraction
        extraction_response = model.generate_content(entity_extraction_prompt)
        # Attempt to parse the JSON response from the LLM
        response_text = extraction_response.text.strip().replace('\n', '')
        # Basic cleanup: remove potential markdown backticks
        if response_text.startswith("```json"):
             response_text = response_text[7:]
        if response_text.startswith("```"):
             response_text = response_text[3:]
        if response_text.endswith("```"):
             response_text = response_text[:-3]
        
        extracted_entities = json.loads(response_text)
        logger.info(f"Extracted Entities: {extracted_entities}")
        # Basic validation
        if not isinstance(extracted_entities.get('symptoms'), list):
            extracted_entities['symptoms'] = [] # Ensure symptoms is a list

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from entity extraction LLM: {e}. Response: {extraction_response.text}")
        # Proceed without extracted entities if parsing fails
    except Exception as e:
        logger.error(f"Error during entity extraction LLM call: {e}", exc_info=True)
        # Proceed without extracted entities
    # --- End Entity Extraction ---

    # --- Identify Device (using Session, CV, Entities, Text) --- 
    identified_device = None
    identified_device_source = None 
    # 1. Check session 
    if session.identified_device_id: 
        identified_device = db.session.get(Device, session.identified_device_id)
        if identified_device: identified_device_source = 'db'

    # 2. Check CV results (if device not already known)
    cv_context = ""
    cv_detected_labels = []
    if not identified_device:
        client_sid = next(iter(pcs_by_session.get(str(session.session_id), {})), None) # Get first sid for session
        if client_sid and str(session.session_id) in latest_cv_results and client_sid in latest_cv_results[str(session.session_id)]:
            current_cv_detections = latest_cv_results[str(session.session_id)][client_sid]
            if current_cv_detections: 
                cv_detected_labels = list(set([d['label'] for d in current_cv_detections]))
                cv_context = f"\n[System Info: Video shows: {', '.join(cv_detected_labels)}]"
                # Try matching CV label to DB type
                for label in cv_detected_labels: 
                    db_type_guess = label if label in ['laptop', 'mouse'] else ('smartphone' if label == 'cell phone' else None)
                    if db_type_guess:
                        device_from_cv = db.session.query(Device).filter(Device.type.ilike(f'%{db_type_guess}%')).first()
                        if device_from_cv: identified_device = device_from_cv; identified_device_source = 'cv'; break 

    # 3. Use LLM Extracted Entities (if device not already known)
    if not identified_device and (extracted_entities.get('device_type') or extracted_entities.get('device_model') or extracted_entities.get('device_brand')):
        logger.info("Attempting device lookup using LLM extracted entities.")
        query = db.session.query(Device)
        if extracted_entities.get('device_type'): query = query.filter(Device.type.ilike(f"%{extracted_entities['device_type']}%"))
        if extracted_entities.get('device_model'): query = query.filter(Device.model.ilike(f"%{extracted_entities['device_model']}%"))
        if extracted_entities.get('device_brand'): query = query.filter(Device.brand.ilike(f"%{extracted_entities['device_brand']}%"))
        device_from_entities = query.first()
        if device_from_entities:
             identified_device = device_from_entities
             identified_device_source = 'llm_entities'
             logger.info(f"Device identified via LLM entities: {identified_device.brand} {identified_device.model}")

    # Link session if device newly identified
    if identified_device and identified_device_source != 'db':
        session.identified_device_id = identified_device.device_id
        db.session.add(session) 
        logger.info(f"Linked session {session_id} to device {identified_device.device_id} (Source: {identified_device_source})")

    # Use LLM extracted symptoms
    extracted_symptoms = extracted_entities.get('symptoms', [])
    logger.info(f"Using LLM extracted symptoms: {extracted_symptoms}")
    # -----------------------------------
    
    # --- Build Knowledge Base Context --- 
    knowledge_context = ""
    potential_guide_key = None
    device_guides = {}
    if identified_device:
        device = identified_device
        knowledge_context += f"\n\n[System Knowledge: {device.brand} {device.model} ({device.type})]\n"
        # Add relevant common issues based on extracted symptoms
        if device.common_issues and isinstance(device.common_issues, list) and extracted_symptoms:
            matched_issue_count = 0
            relevant_issues_text = ""
            for issue in device.common_issues:
                issue_symptom = issue.get('symptom', '')
                # Improved check: see if any extracted symptom is substring of KB symptom or vice versa
                if issue_symptom and any(extracted.lower() in issue_symptom.lower() or issue_symptom.lower() in extracted.lower() for extracted in extracted_symptoms):
                    relevant_issues_text += f"- Issue: {issue_symptom}\n"
                    if issue.get('potential_causes'): relevant_issues_text += f"  Potential Causes: {', '.join(issue['potential_causes'][:2])}...\n"
                    if issue.get('diagnostic_questions'): relevant_issues_text += f"  Suggest asking: '{issue['diagnostic_questions'][0]}'\n"
                    matched_issue_count += 1
            if matched_issue_count > 0: knowledge_context += "Relevant Common Issues Based on Query:\n" + relevant_issues_text
            # else: # Only add general symptoms if no specific match? 
            #     issues_summary = ", ".join([i.get('symptom', '?') for i in device.common_issues[:2]])
            #     if issues_summary: knowledge_context += f"General Symptoms: {issues_summary}...\n"

        # Get available guides and check if user text *directly* mentioned one
        if device.troubleshooting_guides and isinstance(device.troubleshooting_guides, dict):
            device_guides = device.troubleshooting_guides
            guides_summary = ", ".join(device_guides.keys())
            if guides_summary: knowledge_context += f"Available Guides: {guides_summary}\n"
            for key in device_guides.keys():
                if re.search(r'\b' + key.replace('_', ' ') + r'\b', user_text, re.IGNORECASE): # Match whole word/phrase
                    potential_guide_key = key
                    knowledge_context += f"(User mentioned guide: '{key}')\n"
                    break
        if len(knowledge_context) < 100: knowledge_context += "(No specific KB info found)\n"
    # --- End KB Context Build ---

    # --- Prepare Prompt for Main Gemini Call (Structured Output Request) --- 
    if not model: return jsonify({"error": "AI not configured"}), 500
    
    # Refined system instruction - emphasize using extracted symptoms with KB
    system_instruction_parts = [
        "You are an expert Electronics Technician AI Assistant.",
        "Goal: Diagnose issue based ONLY on context (Video, KB, History).",
        f"Context includes: {cv_context if cv_context else '(No video analysis available)'}.",
        f"Identified Device: {(identified_device.brand + ' ' + identified_device.model) if identified_device else 'Unknown'}.",
        f"Knowledge Base for identified device: {knowledge_context if knowledge_context else '(No specific KB info found)'}.",
        f"Extracted User Symptoms: {', '.join(extracted_symptoms) if extracted_symptoms else 'None specified'}.", # Explicitly mention extracted symptoms
        "Analyze the user query AND the extracted symptoms.", # Instruct LLM to use the symptoms
        "If the KB lists 'Relevant Common Issues' based on the symptoms, refer to them and suggest diagnostic questions from the KB.",
        "If the KB lists 'Available Guides' and one seems relevant to the symptoms/query, suggest starting that guide using its EXACT name (e.g., 'smc_reset').",
        "If unsure, ask clarifying questions (e.g., ask for device model/generation if needed). Do NOT invent info.",
        "Output your decision and response as a JSON object with keys: 'action', 'details', 'agent_reply'."
        # ... (Examples remain the same)
    ]
    system_instruction = " ".join(system_instruction_parts)
    full_prompt_for_turn = system_instruction + "\n\nUser Query: " + user_text
    
    agent_reply = "Sorry, I encountered an issue processing that." # Default error reply
    final_diagnostic_state = {'in_guide': False}

    # Save user message (defer commit)
    try: user_message = Message(session_id=session.session_id, sender='user', text=user_text); db.session.add(user_message)
    except Exception as e: db.session.rollback(); logger.error(f"Error adding user msg: {e}"); return jsonify({"error": "DB error"}), 500

    # Call Gemini API for Diagnosis / Next Action
    try:
        chat_history = get_session_history_for_gemini(session.session_id)
        logger.info(f"--- Sending to Gemini (Main Query - Expecting JSON) ---") 
        chat = model.start_chat(history=chat_history)
        response = chat.send_message(full_prompt_for_turn) 
        llm_response_text = response.text.strip()
        logger.info(f"LLM Raw Response: {llm_response_text}")

        # --- Parse Structured Response --- 
        llm_action = 'error'
        llm_details = None
        try:
            # Basic cleanup for markdown code blocks
            if llm_response_text.startswith("```json"): llm_response_text = llm_response_text[7:]
            if llm_response_text.startswith("```"): llm_response_text = llm_response_text[3:]
            if llm_response_text.endswith("```"): llm_response_text = llm_response_text[:-3]
            
            parsed_llm_response = json.loads(llm_response_text)
            llm_action = parsed_llm_response.get('action', 'error')
            llm_details = parsed_llm_response.get('details')
            agent_reply = parsed_llm_response.get('agent_reply', agent_reply) # Use LLM reply if valid
            logger.info(f"Parsed LLM Action: {llm_action}, Details: {llm_details}")

        except json.JSONDecodeError as json_e:
            logger.error(f"Failed to parse JSON response from LLM: {json_e}. Raw: {llm_response_text}")
            # Use the raw text as the reply if JSON parsing fails but text exists
            if llm_response_text: agent_reply = llm_response_text 
            llm_action = 'error' # Treat as error if parsing failed
        # --- End Parse Structured Response --- 

        # --- Process Action --- 
        if llm_action == 'suggest_guide' and llm_details and identified_device and device_guides.get(llm_details):
            guide_key = llm_details
            steps_text = device_guides[guide_key]
            steps_list = []
            # ... (Parse steps_list from steps_text as before)
            if isinstance(steps_text, str): # Parse steps
                 potential_steps = re.split(r'\n?\d+\. ?', steps_text)
                 steps_list = [s.strip() for s in potential_steps if s.strip()]
                 if not steps_list: steps_list = [s.strip() for s in steps_text.split('\n') if s.strip()]
            elif isinstance(steps_text, list): steps_list = steps_text

            if steps_list:
                session.active_guide_name = guide_key
                session.active_guide_steps = json.dumps(steps_list)
                session.current_guide_step = 1
                db.session.add(session)
                # Override agent reply with first step + confirmation from LLM's original reply
                first_step_reply = f"Step 1: {steps_list[0]}"
                # Combine LLM's conversational suggestion with the first step
                agent_reply = f"{agent_reply.strip()} {first_step_reply}" 
                final_diagnostic_state['in_guide'] = True
                logger.info(f"Initiating guide '{guide_key}' based on LLM action.")
            else:
                logger.warning(f"LLM suggested guide '{guide_key}' but it has no steps.")
                # Keep the original agent_reply from the LLM in this case
        elif llm_action == 'request_location':
             # Just use the LLM's agent_reply, frontend needs to interpret this
             logger.info("LLM requested location.")
             # No specific backend state change needed here yet
        elif llm_action in ['ask_clarification', 'provide_information']:
             # Use the agent_reply directly from the LLM
             logger.info(f"LLM action: {llm_action}")
        else: # Includes 'error' or unknown actions
             logger.warning(f"LLM action was '{llm_action}' or parse failed. Using LLM text as basic reply.")
             # Keep the potentially modified agent_reply 
        # --- End Process Action ---

    except Exception as e:
        db.session.rollback() # Rollback user message save if LLM call failed badly
        logger.error(f"Error during Gemini processing: {e}", exc_info=True)
        error_detail = str(e); error_msg = "Failed AI processing"
        if "response" in locals() and hasattr(response, 'prompt_feedback'):
             error_detail = f"Gemini Error: {response.prompt_feedback}"; error_msg = "AI backend processing failed"
        agent_reply = f"Sorry, I encountered an error processing that: {error_msg}."
        # Save user msg + this error reply
        try: 
            agent_message = Message(session_id=session.session_id, sender='agent', text=agent_reply)
            db.session.add_all([user_message, agent_message, session]) # Re-add user message too
            db.session.commit()
        except Exception as db_e: db.session.rollback(); logger.error(f"DB error saving LLM error: {db_e}")
        return jsonify({"error": error_msg, "details": error_detail}), 500

    # --- Save messages and commit --- 
    try:
        agent_message = Message(session_id=session.session_id, sender='agent', text=agent_reply)
        db.session.add(agent_message)
        db.session.add(session) # Add session to ensure updates (like device ID) are staged
        db.session.commit()
        # Update diagnostic state based on final outcome
        final_diagnostic_state['identified_device'] = { 
            'brand': identified_device.brand, 'model': identified_device.model, 
            'type': identified_device.type, 'source': identified_device_source
        } if identified_device else None
        return jsonify({"reply_text": agent_reply, "diagnostic_state": final_diagnostic_state})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving final messages/session: {e}", exc_info=True)
        return jsonify({"error": "Failed to save final messages"}), 500

@app.route('/api/sessions/<session_id>/location_query', methods=['GET'])
def get_location_info(session_id):
    """Provides local repair/retailer info using Google Places API, logs query to DB."""
    if not DATABASE_URL:
        return jsonify({"error": "Database not configured"}), 500
    if not gmaps:
         return jsonify({"error": "Mapping service not configured or failed to initialize."}), 500

    try:
        session_id_uuid = uuid.UUID(session_id) # Convert string ID to UUID
    except ValueError:
        return jsonify({"error": "Invalid session ID format"}), 400

    # Find session in DB (optional, but good practice)
    session = db.session.get(Session, session_id_uuid)
    if not session:
        return jsonify({"error": "Session not found"}), 404

    try:
        latitude = float(request.args.get('latitude'))
        longitude = float(request.args.get('longitude'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing 'latitude' or 'longitude' query parameters"}), 400

    device_type = request.args.get('device_type', '')
    issue_type = request.args.get('issue_type', '')
    query_params = {
        'latitude': latitude,
        'longitude': longitude,
        'device_type': device_type,
        'issue_type': issue_type
    }

    logger.info(f"Location query for session {session_id}: params={query_params}")

    # Construct search query
    search_keywords = "electronics repair OR computer repair OR phone repair OR electronics store"
    if device_type:
        search_keywords = f"{device_type} {issue_type} repair OR {search_keywords}"

    try:
        places_result = gmaps.places_nearby(
            location=(latitude, longitude),
            radius=10000,
            keyword=search_keywords,
            type=['electronics_store', 'hardware_store']
        )

        locations = []
        for place in places_result.get('results', []):
            loc_data = {
                "name": place.get('name'),
                "address": place.get('vicinity'),
                "phone": None,
                "latitude": place['geometry']['location']['lat'],
                "longitude": place['geometry']['location']['lng'],
                "type": ", ".join(place.get('types', [])),
                "google_place_id": place.get('place_id'),
                "maps_url": f"https://www.google.com/maps/search/?api=1&query=Google&query_place_id={place.get('place_id')}"
            }
            locations.append(loc_data)

        # Log the query parameters to the session in DB
        try:
            session.last_location_query = query_params
            db.session.commit()
        except Exception as db_err:
            db.session.rollback()
            logger.warning(f"Warning: Failed to log location query to session {session_id}: {db_err}")
            # Don't fail the whole request if logging fails

        logger.info(f"Found {len(locations)} potential locations.")
        return jsonify({"locations": locations})

    except Exception as e:
        logger.error(f"Error calling Google Maps API: {e}")
        return jsonify({"error": "Failed to get response from mapping service", "details": str(e)}), 500

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_connect():
    """Handles new WebSocket connections."""
    logger.info(f"Client connected: {request.sid}")
    # Optionally require session_id on connect?
    # emit('your_id', {'id': request.sid}) # Send client their socket ID

@socketio.on('disconnect')
def handle_disconnect():
    """Handles WebSocket disconnections."""
    logger.info(f"Client disconnected: {request.sid}")
    # Clean up associated peer connections
    session_id_to_remove = None
    pc_to_remove = None
    sid_to_remove = request.sid

    # Use items() for safe iteration while potentially modifying the dict
    for s_id, pcs_dict in list(pcs_by_session.items()): 
        if sid_to_remove in pcs_dict:
            session_id_to_remove = s_id
            pc_to_remove = pcs_dict[sid_to_remove]
            # Remove the sid entry from the inner dict
            del pcs_by_session[session_id_to_remove][sid_to_remove]
            logger.info(f"Removed PC reference for {sid_to_remove} in session {session_id_to_remove}")
            # If the inner dict is now empty, remove the session entry
            if not pcs_by_session[session_id_to_remove]:
                del pcs_by_session[session_id_to_remove]
                logger.info(f"Removed empty session entry {session_id_to_remove} from PC tracking.")
            break # Found and removed, exit loop
    
    if pc_to_remove:
        logger.info(f"Closing PeerConnection for disconnected client {sid_to_remove}")
        # Schedule the async close operation properly for eventlet/gevent
        # Use socketio.start_background_task if needing true async cleanup
        async def close_pc_async(): 
            await pc_to_remove.close()
        socketio.start_background_task(close_pc_async) # Run close in background
    else:
        logger.warning(f"Could not find PeerConnection for disconnected client {sid_to_remove} to clean up.")

@socketio.on('join_session')
def handle_join_session(data):
    """Client joins a room based on their session_id."""
    session_id = data.get('session_id')
    if session_id:
        join_room(session_id)
        logger.info(f"Client {request.sid} joined room {session_id}")
        # TODO: Maybe fetch session state/messages and send to client?
    else:
        logger.warning(f"Client {request.sid} tried to join without session_id")

# Updated signal handler for WebRTC with aiortc
# Make the handler async to use await with aiortc functions
@socketio.on('signal')
async def handle_signal(data):
    signal_data = data.get('signal')
    session_id = data.get('session_id')
    if not session_id or not signal_data:
        logger.warning(f"Invalid signal message from {request.sid}")
        return
    
    sid = request.sid # Client's socket ID

    # Ensure session entry exists in our tracking dict
    if session_id not in pcs_by_session:
        pcs_by_session[session_id] = {}

    pc = pcs_by_session[session_id].get(sid)

    if signal_data.get('type') == 'offer':
        if pc:
            logger.warning(f"Offer received for client {sid} in session {session_id}, but PC already exists. Closing old one.")
            await pc.close()
            del pcs_by_session[session_id][sid]

        logger.info(f"Received OFFER from {sid} for session {session_id}")
        pc = RTCPeerConnection()
        pcs_by_session[session_id][sid] = pc # Store the new PC, mapping sid to PC

        @pc.on("track")
        async def on_track(track):
            logger.info(f"Track {track.kind} received from {sid}")
            if track.kind == "video":
                # Pass session_id and sid for context if needed
                pc.addTrack(VideoProcessorTrack(track, session_id=session_id, client_sid=sid))
                # Note: Adding the processed track back to the PC sends it to the client.
                # We might not want this; just analyze the incoming track.
                # For analysis only, just process `track` directly without pc.addTrack.
                # Let's adjust: We likely don't need to add the processed track back.
                # Instead, we just consume the original track in our processor.
                # This requires rethinking the processor class slightly, or just
                # running the processing loop directly here.
                # For simplicity now, we keep addTrack but it might send processed video back.
            @track.on("ended")
            async def on_ended():
                logger.info(f"Track {track.kind} from {sid} ended")
                # TODO: Handle track ending (e.g., stop recorder)

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            logger.info(f"ICE connection state for {sid} is {pc.iceConnectionState}")
            if pc.iceConnectionState == "failed":
                await pc.close()
                if sid in pcs_by_session.get(session_id, {}):
                    del pcs_by_session[session_id][sid]
                    logger.info(f"Closed failed PC for {sid}")

        # Set remote description (the offer)
        offer = RTCSessionDescription(sdp=signal_data['sdp'], type=signal_data['type'])
        try:
            await pc.setRemoteDescription(offer)
            logger.info(f"Set remote description (offer) for {sid}")
        except Exception as e:
            logger.error(f"Error setting remote description for {sid}: {e}", exc_info=True)
            return

        # Create answer
        try:
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            logger.info(f"Created and set local description (answer) for {sid}")
        except Exception as e:
            logger.error(f"Error creating/setting answer for {sid}: {e}", exc_info=True)
            await pc.close() # Close PC on error
            if sid in pcs_by_session.get(session_id, {}):
                 del pcs_by_session[session_id][sid]
            return

        # Send answer back to the client
        logger.info(f"Sending ANSWER to {sid}")
        emit('signal', {
            'sid': 'backend', # Identify sender as backend
            'signal': {'type': answer.type, 'sdp': answer.sdp}
        }, room=sid) # Emit directly to the specific client using their sid

    elif signal_data.get('candidate') and pc:
        try:
            candidate_json = signal_data['candidate']
            # Basic check from frontend ICE candidate format
            if candidate_json and isinstance(candidate_json, dict) and 'candidate' in candidate_json:
                 # Construct RTCIceCandidate - needs more robust parsing in production
                candidate = RTCIceCandidate(
                    sdpMid=candidate_json.get('sdpMid'),
                    sdpMLineIndex=candidate_json.get('sdpMLineIndex'),
                    sdp=candidate_json['candidate'],
                )
                logger.info(f"Adding ICE candidate for {sid}: {candidate.sdp[:30]}...")
                await pc.addIceCandidate(candidate)
            else:
                 logger.warning(f"Received malformed ICE candidate from {sid}: {candidate_json}")
        except Exception as e:
            logger.error(f"Error adding ICE candidate for {sid}: {e}", exc_info=True)
    elif not pc:
         logger.warning(f"Received signal for {sid} in session {session_id}, but no active PeerConnection found.")
    else:
        logger.warning(f"Received unknown signal type from {sid}: {signal_data}")

# --- Database Initialization Command ---
@app.cli.command("init-db")
def init_db_command():
    """Create database tables from models."""
    if not DATABASE_URL:
        logger.error("DATABASE_URL not set. Cannot initialize database.")
        return
    try:
        # Create tables within app context
        with app.app_context():
             db.create_all()
        logger.info("Database tables created successfully!")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)

@app.cli.command("seed-db")
def seed_db_command():
    """Adds initial device data to the knowledge base."""
    if not DATABASE_URL:
        logger.error("DATABASE_URL not set. Cannot seed database.")
        return
    
    logger.info("Seeding database with initial device data...")

    # Example Device Data (Add more as needed)
    devices_to_seed = [
        {
            "brand": "Apple", "model": "MacBook Pro 14-inch M1 Pro", "type": "Laptop", "release_year": 2021,
            "specifications": {"cpu": "M1 Pro", "ram_gb": [16, 32], "ports": ["Thunderbolt 4", "HDMI", "SDXC card slot", "MagSafe 3", "Headphone jack"]},
            "common_issues": [
                {"symptom": "Screen flickering", "potential_causes": ["Software bug (macOS)", "Display settings (refresh rate)", "Loose internal cable", "GPU issue"], "diagnostic_questions": ["Does it happen in Safe Mode?", "Does it happen on an external display?", "Did it start after an update?"]},
                {"symptom": "Not charging", "potential_causes": ["Faulty charger/cable", "Dirty MagSafe port", "SMC issue", "Battery health degradation", "Logic board issue"], "troubleshooting_steps": ["Try different outlet", "Try different charger/cable", "Clean port carefully", "Reset SMC"]}
            ],
            "troubleshooting_guides": {"smc_reset": "1. Shut down. 2. Press and hold Control+Option+Shift (left side) + Power button for 10 seconds. 3. Release keys. 4. Press Power button.", "pram_reset": "1. Shut down. 2. Press Power button and immediately press and hold Option+Command+P+R. 3. Hold until you hear the startup sound twice or see Apple logo twice."}
        },
        {
            "brand": "Apple", "model": "iPhone 13 Pro", "type": "Smartphone", "release_year": 2021,
            "specifications": {"chip": "A15 Bionic", "display_size_inches": 6.1, "ports": ["Lightning"]},
            "common_issues": [
                {"symptom": "Battery draining quickly", "potential_causes": ["Background app refresh", "Location services", "iOS update bug", "Battery health degradation"], "troubleshooting_steps": ["Check Battery Usage stats", "Reduce screen brightness", "Turn off background refresh for some apps", "Restart phone", "Update iOS"]},
                {"symptom": "Overheating", "potential_causes": ["Heavy processing (games, video)", "Direct sunlight", "Charging while using heavily", "Software issue"]}
            ]
        },
        {
            "brand": "Anker", "model": "PowerPort III 65W Pod", "type": "Charger", "release_year": None,
            "specifications": {"wattage": 65, "ports": ["USB-C"], "tech": ["PD", "GaN"]},
            "common_issues": [
                {"symptom": "Not charging device", "potential_causes": ["Faulty cable", "Incompatible device", "Charger failure", "Outlet issue"]}
            ]
        },
         {
            "brand": "Logitech", "model": "MX Master 3S", "type": "Mouse", "release_year": 2022,
            "specifications": {"connection": ["Bluetooth", "Logi Bolt USB receiver"], "dpi": 8000, "buttons": 7},
            "common_issues": [
                {"symptom": "Cursor lagging/jumping", "potential_causes": ["Low battery", "Interference", "Surface issue", "Driver/Software issue (Logi Options+)"], "troubleshooting_steps": ["Charge mouse", "Move receiver closer/use extension", "Try different mousepad/surface", "Reconnect Bluetooth", "Update/reinstall Logi Options+"]},
                {"symptom": "Not connecting", "potential_causes": ["Turned off", "Not paired", "Receiver issue", "Bluetooth issue (OS)"], "troubleshooting_steps": ["Check power switch", "Re-pair device via Bluetooth/Receiver", "Try different USB port for receiver", "Restart computer Bluetooth"]}
            ]
        },
        {
            "brand": "Apple", "model": "AirPods (2nd generation)", "type": "Earbuds", "release_year": 2019,
            "specifications": { "chip": "H1", "connection": "Bluetooth 5.0", "case_charging": ["Lightning"] },
            "common_issues": [
                {"symptom": "One AirPod not working", "potential_causes": ["Dirty charging contacts (AirPod or case)", "Needs reset", "Low battery on one AirPod", "Pairing issue", "Hardware failure"], "troubleshooting_steps": ["Clean contacts gently", "Check battery levels", "Put both AirPods in case, close lid, wait 30s, open near iPhone", "Reset AirPods (hold case button)"] },
                {"symptom": "Not connecting", "potential_causes": ["Bluetooth off on device", "Needs reset", "Out of range", "Interference", "Needs charge"], "diagnostic_questions": ["Is Bluetooth enabled on your phone/computer?", "Are the AirPods charged?", "Have you tried resetting them?"] },
                {"symptom": "Audio cutting out", "potential_causes": ["Bluetooth interference", "Out of range", "Software glitch (iOS/macOS)", "Damaged AirPod"], "troubleshooting_steps": ["Move closer to device", "Avoid microwaves/dense Wi-Fi areas", "Restart phone/computer", "Reset AirPods"] },
                {"symptom": "Fell in water", "potential_causes": ["Liquid damage (short circuit)"], "diagnostic_questions": ["How long ago?", "What type of liquid?", "Have you tried drying it?"], "troubleshooting_steps": ["DO NOT charge", "Wipe dry thoroughly", "Leave out to air dry completely (24-48 hours)", "Try silica gel packets", "Contact Apple for repair/replacement (liquid damage often not covered)"] }
            ],
            "troubleshooting_guides": { 
                "reset_airpods": "1. Put AirPods in charging case, close lid. 2. Wait 30 seconds. 3. Open lid. 4. On your iPhone/iPad, go to Settings > Bluetooth, tap 'i' next to AirPods, tap 'Forget This Device'. 5. With case lid open, press and hold setup button on back of case for ~15 seconds until status light flashes amber, then white. 6. Reconnect by placing near device with lid open."
            }
        }
    ]

    try:
        with app.app_context():
            added_count = 0
            skipped_count = 0
            for device_data in devices_to_seed:
                # Check if device already exists
                exists = db.session.query(Device.device_id).filter_by(
                    brand=device_data["brand"],
                    model=device_data["model"],
                    type=device_data["type"]
                ).first() is not None

                if not exists:
                    new_device = Device(**device_data)
                    db.session.add(new_device)
                    added_count += 1
                    logger.info(f"Adding: {device_data['brand']} {device_data['model']}")
                else:
                    skipped_count += 1
                    logger.info(f"Skipping (already exists): {device_data['brand']} {device_data['model']}")
            
            if added_count > 0:
                db.session.commit()
                logger.info(f"Successfully added {added_count} devices.")
            if skipped_count > 0:
                 logger.info(f"Skipped {skipped_count} devices that already exist.")
            if added_count == 0 and skipped_count == 0:
                logger.info("No new devices to add.")

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error seeding database: {e}", exc_info=True)

# --- Main Execution (Using SocketIO) ---
if __name__ == '__main__':
    logger.info("Starting Flask-SocketIO server with aiortc support...")
    # Use socketio.run() instead of app.run()
    # Ensure host='0.0.0.0' to be accessible externally if needed, default is 127.0.0.1
    socketio.run(app, debug=True, port=5001, host='127.0.0.1') 