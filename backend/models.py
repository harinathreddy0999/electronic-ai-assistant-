from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import UUID # Use UUID type for session_id
from sqlalchemy import Text, ForeignKey, DateTime, Integer, String, JSON
import uuid
from datetime import datetime

# Initialize SQLAlchemy without app for now, will be bound in app.py
db = SQLAlchemy()

class Session(db.Model):
    __tablename__ = 'sessions'

    # Use UUID for primary key, matching API design
    session_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.user_id'), nullable=True) # Link to Users table if implemented
    start_time = db.Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = db.Column(DateTime, nullable=True)
    identified_device_id = db.Column(db.Integer, db.ForeignKey('devices.device_id'), nullable=True) # Link to Devices table
    diagnostic_summary = db.Column(Text, nullable=True)
    last_location_query = db.Column(JSON, nullable=True) # Store as JSON
    status = db.Column(String(50), nullable=False, default='active') # e.g., active, ended
    # --- Added for stateful guide tracking --- 
    active_guide_name = db.Column(String(100), nullable=True) # e.g., "smc_reset"
    active_guide_steps = db.Column(JSON, nullable=True)     # Store the actual steps when guide starts
    current_guide_step = db.Column(Integer, nullable=True) # 1-based index of the current step
    # -----------------------------------------
    created_at = db.Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to Messages (one-to-many)
    messages = db.relationship('Message', backref='session', lazy=True, cascade="all, delete-orphan")
    identified_device = db.relationship('Device', backref='sessions', lazy=True) # Added relationship definition

    def __repr__(self):
        return f'<Session {self.session_id} - Guide: {self.active_guide_name} Step: {self.current_guide_step}>'

class Message(db.Model):
    __tablename__ = 'messages'

    message_id = db.Column(Integer, primary_key=True) # Auto-incrementing integer PK
    session_id = db.Column(UUID(as_uuid=True), ForeignKey('sessions.session_id'), nullable=False)
    timestamp = db.Column(DateTime, nullable=False, default=datetime.utcnow)
    sender = db.Column(String(10), nullable=False) # 'user', 'agent', or 'system'
    text = db.Column(Text, nullable=False)
    # Sequence number might be implicitly handled by timestamp ordering or message_id
    # sequence_number = db.Column(Integer, nullable=False)

    def __repr__(self):
        return f'<Message {self.message_id} from {self.sender} in Session {self.session_id}>'

# --- Electronics Knowledge Base Model ---
class Device(db.Model):
    __tablename__ = 'devices'

    device_id = db.Column(Integer, primary_key=True) # Auto-incrementing ID
    brand = db.Column(String(100), nullable=False, index=True)
    model = db.Column(String(150), nullable=False, index=True)
    type = db.Column(String(50), nullable=False, index=True) # e.g., Laptop, Smartphone, Charger
    release_year = db.Column(Integer, nullable=True)
    # Using JSONB for potentially complex, queryable structured data
    specifications = db.Column(JSON, nullable=True) # e.g., { "cpu": "M1", "ram_gb": 8, "ports": ["usb-c", "headphone"] }
    common_issues = db.Column(JSON, nullable=True) # e.g., [ { "symptom": "screen flicker", "causes": [...], "steps": [...] } ]
    troubleshooting_guides = db.Column(JSON, nullable=True) # e.g., { "smc_reset": "...steps...", "pram_reset": "..." }
    visual_identifiers = db.Column(JSON, nullable=True) # e.g., { "port_locations": {...}, "logo_feature": "..." }
    
    created_at = db.Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship back to Sessions (one-to-many)
    # sessions = db.relationship('Session', backref='identified_device', lazy=True) # Defined in Session now

    # Unique constraint on brand/model/type might be useful depending on data granularity
    # db.UniqueConstraint('brand', 'model', 'type', name='uq_device_identifier')

    def __repr__(self):
        return f'<Device {self.device_id}: {self.brand} {self.model} ({self.type})>'

# TODO: Define User model later if needed 