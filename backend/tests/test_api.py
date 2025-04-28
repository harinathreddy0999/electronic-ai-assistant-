import pytest
from backend.app import app as flask_app, db as sqlalchemy_db # Import app and db
from backend.models import Session, Message, Device # Import models
import uuid
import os
import json
from unittest.mock import patch, MagicMock

# --- Test Fixtures --- 

@pytest.fixture(scope='module')
def app():
    """Flask app fixture configured for testing with in-memory SQLite."""
    # Generate a unique URI for each test module run
    db_uri = f"sqlite:///:memory:?uri=true&{uuid.uuid4()}" 
    
    flask_app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": db_uri,
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        # Disable external API calls during most tests unless explicitly un-mocked
        "GOOGLE_API_KEY": None, 
        "GOOGLE_MAPS_API_KEY": None,
    })
    
    # Establish an application context before creating the tables.
    with flask_app.app_context():
        print(f"Creating tables for {db_uri}")
        sqlalchemy_db.create_all()

    yield flask_app

    # Teardown: Implicitly handled by in-memory DB scope, but good practice:
    with flask_app.app_context():
        # print(f"Dropping tables for {db_uri}") # Optional: verify teardown
        # sqlalchemy_db.session.remove()
        # sqlalchemy_db.drop_all()
        pass

@pytest.fixture()
def db(app):
    """Yield the SQLAlchemy db object."""
    # Ensure we are in app context for db operations if needed outside requests
    with app.app_context():
        yield sqlalchemy_db

@pytest.fixture()
def client(app):
    """Test client fixture."""
    return app.test_client()

# --- Helpers to check if REAL APIs are configured (for skipping tests) ---
REAL_GEMINI_API_CONFIGURED = bool(os.getenv('GOOGLE_API_KEY'))
REAL_MAPS_API_CONFIGURED = bool(os.getenv('GOOGLE_MAPS_API_KEY'))
# ---------------------------------------------

# --- API Endpoint Tests --- 

def test_create_session(client, db):
    """Test session creation and DB interaction."""
    response = client.post('/api/sessions')
    assert response.status_code == 201
    json_data = response.get_json()
    session_id_str = json_data['session_id']
    assert 'session_id' in json_data
    session_id_uuid = uuid.UUID(session_id_str)
    
    # Verify session exists in the database
    with db.engine.connect() as connection:
         session_record = connection.execute(db.select(Session).filter_by(session_id=session_id_uuid)).first()
         assert session_record is not None
         assert session_record.session_id == session_id_uuid
         assert session_record.status == 'active'

@patch('backend.app.model') # Mock the Gemini model object
def test_handle_message_db_interaction(mock_gemini_model, client, db):
    """Integration test: message saving and placeholder reply (mocked LLM)."""
    # Mock Gemini response
    mock_chat = MagicMock()
    mock_chat.send_message.return_value = MagicMock(text="Mocked LLM reply")
    mock_gemini_model.start_chat.return_value = mock_chat
    
    # 1. Create Session
    create_resp = client.post('/api/sessions')
    session_id = create_resp.get_json()['session_id']
    session_id_uuid = uuid.UUID(session_id)

    # 2. Send Message
    user_text = "This is a test message"
    message_payload = {"text": user_text}
    msg_resp = client.post(f'/api/sessions/{session_id}/messages', json=message_payload)

    # 3. Assertions
    assert msg_resp.status_code == 200
    assert msg_resp.get_json()['reply_text'] == "Mocked LLM reply"
    mock_gemini_model.start_chat.assert_called_once() # Check if LLM was called
    mock_chat.send_message.assert_called_once() 

    # 4. Verify DB writes
    with db.engine.connect() as connection:
        messages = connection.execute(
            db.select(Message).filter_by(session_id=session_id_uuid).order_by(Message.timestamp)
        ).fetchall()
        assert len(messages) == 2
        assert messages[0].sender == 'user'
        assert messages[0].text == user_text
        assert messages[1].sender == 'agent'
        assert messages[1].text == "Mocked LLM reply"

# Test other handle_message scenarios (invalid session, missing text) - these don't need mocks
def test_handle_message_invalid_session(client):
    """Test sending a message to a non-existent session."""
    invalid_session_id = uuid.uuid4() # Generate a random UUID
    message_payload = {"text": "hello?"}
    response = client.post(f'/api/sessions/{invalid_session_id}/messages', json=message_payload)
    assert response.status_code == 404
    assert 'error' in response.get_json()
    assert 'Session not found' in response.get_json()['error']

def test_handle_message_missing_text(client):
    """Test sending a message without the 'text' field."""
    create_response = client.post('/api/sessions')
    session_id = create_response.get_json()['session_id']
    message_payload = {"not_text": "something else"}
    response = client.post(f'/api/sessions/{session_id}/messages', json=message_payload)
    assert response.status_code == 400
    assert 'error' in response.get_json()
    assert 'Missing \'text\'' in response.get_json()['error']

@patch('backend.app.gmaps') # Mock the Google Maps client object
def test_location_query_db_interaction(mock_gmaps_client, client, db):
    """Integration test: location query logging to DB (mocked Maps API)."""
    # Mock Maps API response
    mock_gmaps_client.places_nearby.return_value = {
        'results': [
            {'name': 'Mock Repair', 'vicinity': '123 Test St', 'geometry': {'location': {'lat': 41.8, 'lng': -88.3}}, 'types': ['electronics_store'], 'place_id': 'mock_place_1'}
        ]
    }

    # 1. Create Session
    create_resp = client.post('/api/sessions')
    session_id = create_resp.get_json()['session_id']
    session_id_uuid = uuid.UUID(session_id)

    # 2. Send Location Query
    lat, lon = 41.805, -88.35
    device, issue = "test_device", "test_issue"
    query_params = {'latitude': lat, 'longitude': lon, 'device_type': device, 'issue_type': issue}
    loc_resp = client.get(f'/api/sessions/{session_id}/location_query', query_string=query_params)

    # 3. Assertions
    assert loc_resp.status_code == 200
    assert len(loc_resp.get_json()['locations']) == 1
    mock_gmaps_client.places_nearby.assert_called_once()

    # 4. Verify DB log write
    with db.engine.connect() as connection:
        session_record = connection.execute(db.select(Session).filter_by(session_id=session_id_uuid)).first()
        assert session_record is not None
        assert session_record.last_location_query is not None
        # Depending on DB driver, might be string or dict
        query_log = json.loads(session_record.last_location_query) if isinstance(session_record.last_location_query, str) else session_record.last_location_query
        assert query_log['latitude'] == lat
        assert query_log['longitude'] == lon
        assert query_log['device_type'] == device
        assert query_log['issue_type'] == issue

# Test other location_query scenarios (invalid session, missing/invalid params) - these don't need mocks
def test_location_query_invalid_session(client):
    """Test GET /location_query with invalid session ID."""
    invalid_session_id = uuid.uuid4()
    query_params = {'latitude': 40.7, 'longitude': -74.0} # NYC
    response = client.get(f'/api/sessions/{invalid_session_id}/location_query', query_string=query_params)
    assert response.status_code == 404
    assert 'Session not found' in response.get_json().get('error', '')

def test_location_query_missing_lat_lon(client):
    """Test GET /location_query with missing latitude."""
    create_response = client.post('/api/sessions')
    session_id = create_response.get_json()['session_id']
    query_params = {'longitude': -88.3}
    response = client.get(f'/api/sessions/{session_id}/location_query', query_string=query_params)
    assert response.status_code == 400
    assert 'Invalid or missing' in response.get_json().get('error', '')
    
    query_params = {'latitude': 41.8}
    response = client.get(f'/api/sessions/{session_id}/location_query', query_string=query_params)
    assert response.status_code == 400
    assert 'Invalid or missing' in response.get_json().get('error', '')

def test_location_query_invalid_lat_lon(client):
    """Test GET /location_query with invalid latitude/longitude format."""
    create_response = client.post('/api/sessions')
    session_id = create_response.get_json()['session_id']
    query_params = {'latitude': 'not-a-number', 'longitude': -88.3}
    response = client.get(f'/api/sessions/{session_id}/location_query', query_string=query_params)
    assert response.status_code == 400
    assert 'Invalid or missing' in response.get_json().get('error', '')

# --- Stateful Guide Logic Tests (using mocks) --- 
@patch('backend.app.db.session')
def test_guide_confirmation_next_step(mock_db_session, client):
    """Test confirming a step moves to the next step."""
    session_id = uuid.uuid4()
    guide_name = "test_guide"
    steps = ["Step One", "Step Two", "Step Three"]
    current_step = 1

    # Mock the session object returned by db.session.get
    mock_session = MagicMock(spec=Session)
    mock_session.session_id = session_id
    mock_session.status = 'active'
    mock_session.active_guide_name = guide_name
    mock_session.active_guide_steps = json.dumps(steps)
    mock_session.current_guide_step = current_step
    mock_db_session.get.return_value = mock_session

    # Simulate user confirming step 1
    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": "yes, done"})

    assert response.status_code == 200
    reply = response.get_json()['reply_text']
    assert f"Step {current_step + 1}" in reply
    assert steps[1] in reply # Check content of next step
    
    # Verify session state was updated (mocked)
    mock_db_session.add.assert_called_with(mock_session)
    assert mock_session.current_guide_step == current_step + 1 # Verify step increment
    mock_db_session.commit.assert_called_once()

@patch('backend.app.db.session')
def test_guide_confirmation_last_step(mock_db_session, client):
    """Test confirming the last step finishes the guide."""
    session_id = uuid.uuid4()
    guide_name = "test_guide"
    steps = ["Step One", "Step Two"]
    current_step = 2 # Start on the last step

    mock_session = MagicMock(spec=Session)
    mock_session.session_id = session_id
    mock_session.status = 'active'
    mock_session.active_guide_name = guide_name
    mock_session.active_guide_steps = json.dumps(steps)
    mock_session.current_guide_step = current_step
    mock_db_session.get.return_value = mock_session

    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": "ok done"})

    assert response.status_code == 200
    reply = response.get_json()['reply_text']
    assert "last step" in reply
    assert "Did that resolve the issue?" in reply

    # Verify guide state was cleared
    mock_db_session.add.assert_called_with(mock_session)
    assert mock_session.active_guide_name is None
    assert mock_session.active_guide_steps is None
    assert mock_session.current_guide_step is None
    mock_db_session.commit.assert_called_once()

@patch('backend.app.db.session')
def test_guide_problem_report(mock_db_session, client):
    """Test reporting a problem stops the guide."""
    session_id = uuid.uuid4()
    guide_name = "test_guide"
    steps = ["Step One", "Step Two"]
    current_step = 1

    mock_session = MagicMock(spec=Session)
    mock_session.session_id = session_id
    mock_session.status = 'active'
    mock_session.active_guide_name = guide_name
    mock_session.active_guide_steps = json.dumps(steps)
    mock_session.current_guide_step = current_step
    mock_db_session.get.return_value = mock_session

    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": "no that didnt work"})

    assert response.status_code == 200
    reply = response.get_json()['reply_text']
    assert "didn't work" in reply # Check for appropriate agent response
    assert "stop the current guide" in reply

    # Verify guide state was cleared
    mock_db_session.add.assert_called_with(mock_session)
    assert mock_session.active_guide_name is None
    assert mock_session.active_guide_steps is None
    assert mock_session.current_guide_step is None
    mock_db_session.commit.assert_called_once()

@patch('backend.app.db.session')
def test_guide_ambiguous_response(mock_db_session, client):
    """Test ambiguous response asks for clarification."""
    session_id = uuid.uuid4()
    guide_name = "test_guide"
    steps = ["Do the first thing", "Do the second thing"]
    current_step = 1

    mock_session = MagicMock(spec=Session)
    mock_session.session_id = session_id
    mock_session.status = 'active'
    mock_session.active_guide_name = guide_name
    mock_session.active_guide_steps = json.dumps(steps)
    mock_session.current_guide_step = current_step
    mock_db_session.get.return_value = mock_session

    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": "what about the screen?"})

    assert response.status_code == 200
    reply = response.get_json()['reply_text']
    assert "Sorry, I wasn't sure" in reply 
    assert f"Step {current_step}" in reply
    assert "Did you complete it successfully?" in reply

    # Verify guide state remains unchanged
    mock_db_session.add.assert_called_with(mock_session)
    assert mock_session.active_guide_name == guide_name
    assert mock_session.current_guide_step == current_step
    mock_db_session.commit.assert_called_once()

# --- Stateful Guide Logic Tests (Initiation) ---
@patch('backend.app.db.session') # Mock database session
@patch('backend.app.model')      # Mock the Gemini model object
def test_guide_initiation_via_llm(mock_gemini_model, mock_db_session, client, app):
    """Test guide initiation when LLM suggests a known guide."""
    session_id = uuid.uuid4()
    guide_name = "smc_reset"
    guide_steps_text = "1. First Step Content.\n2. Second Step Content."
    parsed_steps = ["First Step Content.", "Second Step Content."]

    # Mock session returned for message handler
    mock_session = MagicMock(spec=Session)
    mock_session.session_id = session_id
    mock_session.status = 'active'
    mock_session.identified_device_id = 1 # Assume device already identified
    mock_session.active_guide_name = None # Ensure no guide active initially
    mock_session.active_guide_steps = None
    mock_session.current_guide_step = None
    
    # Mock device returned when session is loaded
    mock_device = MagicMock(spec=Device)
    mock_device.device_id = 1
    mock_device.brand = "Apple"
    mock_device.model = "MacBook Pro"
    mock_device.type = "Laptop"
    mock_device.common_issues = []
    mock_device.troubleshooting_guides = {guide_name: guide_steps_text} # Define the guide
    
    # Configure mock db.session.get to return session or device based on args
    def mock_get(model_cls, pk):
        if model_cls == Session and pk == session_id:
            return mock_session
        if model_cls == Device and pk == 1:
             return mock_device
        return None
    mock_db_session.get.side_effect = mock_get

    # Mock Gemini response to suggest the guide
    llm_reply_suggesting_guide = f"Based on the symptoms, let's try the {guide_name.replace('_',' ')} guide."
    mock_chat = MagicMock()
    mock_chat.send_message.return_value = MagicMock(text=llm_reply_suggesting_guide)
    mock_gemini_model.start_chat.return_value = mock_chat

    # Simulate user message that might trigger guide suggestion
    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": "It won\'t charge"})

    # Assertions
    assert response.status_code == 200
    reply_json = response.get_json()
    reply_text = reply_json['reply_text']
    
    # Check that the reply is the *first step* of the guide, not the LLM suggestion
    assert f"Step 1: {parsed_steps[0]}" in reply_text
    assert guide_name.replace('_',' ') in reply_text # Should mention the guide name
    assert llm_reply_suggesting_guide not in reply_text # Original LLM reply is overridden
    assert reply_json['diagnostic_state']['in_guide'] is True

    # Verify session state was updated in the (mocked) DB commit
    mock_db_session.add.assert_any_call(mock_session) # Check session was added for update
    assert mock_session.active_guide_name == guide_name
    assert mock_session.active_guide_steps == json.dumps(parsed_steps)
    assert mock_session.current_guide_step == 1
    mock_db_session.commit.assert_called_once() # Check commit happened

# --- Entity Extraction Logic Tests (within handle_message) ---
@patch('backend.app.db.session')
@patch('backend.app.model') # Mock Gemini model completely for these tests
def test_entity_extraction_success(mock_gemini_model, mock_db_session, client):
    """Test successful entity extraction and basic parsing."""
    session_id = uuid.uuid4()
    user_query = "My iPhone 13 screen is flickering badly."
    expected_entities = {
        "device_brand": "Apple", 
        "device_model": "iPhone 13", 
        "device_type": "Smartphone", 
        "symptoms": ["Screen flickering"]
    }

    # Mock session returned by db.session.get
    mock_session = MagicMock(spec=Session)
    mock_session.session_id = session_id
    mock_session.status = 'active'
    mock_session.identified_device_id = None # No device identified yet
    mock_session.active_guide_name = None 
    mock_db_session.get.return_value = mock_session

    # Mock the two potential LLM calls
    # 1. Entity Extraction Call
    mock_entity_response = MagicMock()
    # Simulate LLM potentially adding markdown backticks
    mock_entity_response.text = f"```json\n{json.dumps(expected_entities)}\n```"
    # 2. Main Diagnostic Call (mocked minimally as we focus on extraction here)
    mock_main_response = MagicMock(text="Acknowledged flickering iPhone.")
    mock_chat_instance = MagicMock()
    mock_chat_instance.send_message.return_value = mock_main_response
    
    # Configure the main model mock
    mock_gemini_model.generate_content.return_value = mock_entity_response # For extraction call
    mock_gemini_model.start_chat.return_value = mock_chat_instance # For main diagnostic call

    # Send the message
    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": user_query})

    # Assertions
    assert response.status_code == 200 # Should proceed to main call
    mock_gemini_model.generate_content.assert_called_once() # Verify extraction call happened
    # Check that the prompt contained the user query
    entity_prompt_args = mock_gemini_model.generate_content.call_args[0]
    assert user_query in entity_prompt_args[0]
    assert "JSON Output:" in entity_prompt_args[0]

    # Verify the main diagnostic call happened (implicitly checks extraction didn't crash)
    mock_gemini_model.start_chat.assert_called_once()
    # We could inspect the prompt sent to the main call here if needed to verify context
    
    # Check final reply isn't empty (it's mocked)
    assert response.get_json()['reply_text'] == "Acknowledged flickering iPhone."

@patch('backend.app.db.session')
@patch('backend.app.model')
def test_entity_extraction_json_error(mock_gemini_model, mock_db_session, client):
    """Test proceeding gracefully if entity extraction LLM returns bad JSON."""
    session_id = uuid.uuid4()
    user_query = "My screen has weird lines."

    # Mock session
    mock_session = MagicMock(spec=Session)
    mock_session.session_id = session_id; mock_session.status = 'active'; mock_session.identified_device_id = None; mock_session.active_guide_name = None
    mock_db_session.get.return_value = mock_session

    # Mock LLM calls - Entity extraction returns malformed JSON
    mock_entity_response = MagicMock(text="This is not json")
    mock_main_response = MagicMock(text="Generic help response.")
    mock_chat_instance = MagicMock(); mock_chat_instance.send_message.return_value = mock_main_response
    mock_gemini_model.generate_content.return_value = mock_entity_response
    mock_gemini_model.start_chat.return_value = mock_chat_instance

    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": user_query})

    # Assertions - Should still succeed but without specific entity context
    assert response.status_code == 200
    mock_gemini_model.generate_content.assert_called_once()
    mock_gemini_model.start_chat.assert_called_once()
    # Check the prompt sent to main diagnostic call - should lack detailed KB context
    main_call_args = mock_chat_instance.send_message.call_args[0]
    prompt_sent = main_call_args[0]
    assert "Knowledge Base info for identified device (Unknown Device)" in prompt_sent
    assert "User mentioned symptoms seem to be: None specified" in prompt_sent
    assert response.get_json()['reply_text'] == "Generic help response."

# --- Tests for Handling Structured LLM Responses --- 

@patch('backend.app.db.session')
@patch('backend.app.model') 
def test_handle_message_structured_llm_ask_clarification(mock_gemini_model, mock_db_session, client):
    """Test when LLM returns structured JSON for clarification."""
    session_id = uuid.uuid4()
    user_query = "It is broken."
    
    # Mock Session
    mock_session = MagicMock(spec=Session); mock_session.session_id = session_id; mock_session.status = 'active'; mock_session.identified_device_id = None; mock_session.active_guide_name = None
    mock_db_session.get.return_value = mock_session

    # Mock Entity Extraction (return empty)
    mock_entity_response = MagicMock(text=json.dumps({"device_brand": None, "device_model": None, "device_type": None, "symptoms": []}))
    # Mock Main Diagnostic Call (return structured JSON)
    structured_response_json = {
        "action": "ask_clarification",
        "details": "What specifically seems broken?",
        "agent_reply": "Okay, you mentioned it's broken. Could you tell me more about what specifically seems broken?"
    }
    mock_main_response = MagicMock(text=json.dumps(structured_response_json))
    mock_chat_instance = MagicMock(); mock_chat_instance.send_message.return_value = mock_main_response
    mock_gemini_model.generate_content.return_value = mock_entity_response
    mock_gemini_model.start_chat.return_value = mock_chat_instance

    # Send message
    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": user_query})

    # Assertions
    assert response.status_code == 200
    reply_json = response.get_json()
    assert reply_json['reply_text'] == structured_response_json['agent_reply']
    assert reply_json['diagnostic_state']['in_guide'] is False
    mock_db_session.commit.assert_called_once() # Check commit happened

@patch('backend.app.db.session')
@patch('backend.app.model') 
def test_handle_message_structured_llm_suggest_guide(mock_gemini_model, mock_db_session, client):
    """Test when LLM returns structured JSON suggesting a guide (initiation test redundant?)."""
    # This test overlaps significantly with test_guide_initiation_via_llm, 
    # but verifies the structured JSON parsing path specifically.
    session_id = uuid.uuid4()
    user_query = "My macbook won't charge, maybe smc reset?"
    guide_name = "smc_reset"
    guide_steps_text = "1. Step Alpha.\n2. Step Beta."
    parsed_steps = ["Step Alpha.", "Step Beta."]

    # Mock Session & Device
    mock_session = MagicMock(spec=Session); mock_session.session_id = session_id; mock_session.status = 'active'; mock_session.identified_device_id = 1; mock_session.active_guide_name = None
    mock_device = MagicMock(spec=Device); mock_device.device_id = 1; mock_device.brand = "Apple"; mock_device.model = "MacBook Pro"; mock_device.type = "Laptop"; mock_device.common_issues = []; mock_device.troubleshooting_guides = {guide_name: guide_steps_text}
    def mock_get(model_cls, pk):
        if model_cls == Session and pk == session_id: return mock_session
        if model_cls == Device and pk == 1: return mock_device
        return None
    mock_db_session.get.side_effect = mock_get

    # Mock Entity Extraction (assume it found MacBook Pro)
    mock_entity_response = MagicMock(text=json.dumps({"device_brand": "Apple", "device_model": "MacBook Pro", "device_type": "Laptop", "symptoms": ["won't charge"]}))
    # Mock Main Diagnostic Call (return structured JSON suggesting the guide)
    structured_response_json = {
        "action": "suggest_guide",
        "details": guide_name,
        "agent_reply": f"Since you mentioned it won't charge, let's try the {guide_name.replace('_',' ')} guide."
    }
    mock_main_response = MagicMock(text=json.dumps(structured_response_json))
    mock_chat_instance = MagicMock(); mock_chat_instance.send_message.return_value = mock_main_response
    mock_gemini_model.generate_content.return_value = mock_entity_response
    mock_gemini_model.start_chat.return_value = mock_chat_instance

    # Send message
    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": user_query})

    # Assertions
    assert response.status_code == 200
    reply_json = response.get_json()
    reply_text = reply_json['reply_text']
    # Verify reply is combination of LLM suggestion + first step
    assert structured_response_json['agent_reply'] in reply_text
    assert f"Step 1: {parsed_steps[0]}" in reply_text
    assert reply_json['diagnostic_state']['in_guide'] is True
    # Verify session state was updated
    assert mock_session.active_guide_name == guide_name
    assert mock_session.current_guide_step == 1
    mock_db_session.commit.assert_called_once()

@patch('backend.app.db.session')
@patch('backend.app.model') 
def test_handle_message_structured_llm_malformed_json(mock_gemini_model, mock_db_session, client):
    """Test when LLM returns malformed JSON - should fallback to text."""
    session_id = uuid.uuid4()
    user_query = "Keyboard issue."
    # Mock Session
    mock_session = MagicMock(spec=Session); mock_session.session_id = session_id; mock_session.status = 'active'; mock_session.identified_device_id = None; mock_session.active_guide_name = None
    mock_db_session.get.return_value = mock_session
    # Mock Entity Extraction
    mock_entity_response = MagicMock(text=json.dumps({"device_type": "Keyboard", "symptoms": ["issue"]}))
    # Mock Main Call - Malformed JSON
    malformed_json_text = "{\"action\": \"ask_clarification\", \"details\": \"Which keys?}" # Missing closing quote
    mock_main_response = MagicMock(text=malformed_json_text)
    mock_chat_instance = MagicMock(); mock_chat_instance.send_message.return_value = mock_main_response
    mock_gemini_model.generate_content.return_value = mock_entity_response
    mock_gemini_model.start_chat.return_value = mock_chat_instance

    response = client.post(f'/api/sessions/{session_id}/messages', json={"text": user_query})

    # Assertions
    assert response.status_code == 200
    reply_json = response.get_json()
    # Should use the raw malformed text as the reply
    assert reply_json['reply_text'] == malformed_json_text
    assert reply_json['diagnostic_state']['in_guide'] is False # Guide not initiated
    mock_db_session.commit.assert_called_once()

# -------------------------------------------------------

# TODO: Add tests for SocketIO events & WebRTC
# TODO: Add tests for guide *initiation* logic (post-LLM check)
# TODO: Add tests for helper functions (check test_helpers.py)
# TODO: Add tests for guide *initiation* logic (post-LLM check in handle_message)
# TODO: Add tests for helper functions (find_device_in_text, extract_symptoms, check_user_confirmation)
# TODO: Add tests for SocketIO events (requires pytest-socketio or mocking)
# TODO: Add tests for stateful guide logic in handle_message 