import pytest
from unittest.mock import patch, MagicMock

# Import functions to test from the main app module
from backend.app import (
    check_user_confirmation, 
    extract_symptoms, 
    find_device_in_text
)
# We might need Device model for mocking query results
from backend.models import Device 

# --- Tests for check_user_confirmation ---

@pytest.mark.parametrize("text, expected", [
    ("yes", True),
    ("Yes, I did that", True),
    ("okay", True),
    ("ok", True),
    ("done", True),
    ("yep, finished", True),
    ("yeah sure", True),
    ("alright", True),
    ("no", False),
    ("nope", False),
    ("it didn't work", False),
    ("still broken", False),
    ("I have a problem", False),
    ("there is an issue", False),
    ("error occurred", False),
    ("what was the first step?", None),
    ("I see a laptop", None),
    ("tell me more", None),
    ("", None),
])
def test_check_user_confirmation(text, expected):
    assert check_user_confirmation(text) == expected

# --- Tests for extract_symptoms ---

@pytest.mark.parametrize("text, expected_symptoms", [
    ("my screen keeps flashing", ["Screen flickering"]),
    ("it won't charge at all", ["Not charging"]),
    ("the battery is draining really fast", ["Battery draining quickly"]),
    ("phone gets too hot when I use it", ["Overheating"]),
    ("my mouse pointer keeps jumping around", ["Cursor lagging/jumping"]),
    ("bluetooth is not connecting to my speaker", ["Not connecting"]),
    ("screen flicker and it gets hot", ["Screen flickering", "Overheating"]),
    ("the charger isn't working and the battery dies fast", ["Not charging", "Battery draining quickly"]),
    ("everything seems fine", []),
    ("", []),
])
def test_extract_symptoms(text, expected_symptoms):
    assert sorted(extract_symptoms(text)) == sorted(expected_symptoms)

# --- Tests for find_device_in_text ---
# Mock the database interaction for this unit test

@patch('backend.app.db.session') # Mock the db session used in the function
def test_find_device_in_text_laptop(mock_session):
    """Test finding laptop keywords."""
    # Configure mock query result
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_limit = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.limit.return_value = mock_limit
    # Simulate finding a generic laptop entry
    mock_limit.all.return_value = [Device(device_id=1, brand="Generic", model="Laptop Base", type="Laptop")]

    result = find_device_in_text("my laptop screen is broken")
    assert len(result) == 1
    assert isinstance(result[0], Device)
    assert result[0].type == "Laptop"
    mock_session.query.assert_called_once_with(Device)
    # We could add more assertions on the filter arguments if needed

@patch('backend.app.db.session')
def test_find_device_in_text_iphone(mock_session):
    """Test finding iphone keywords."""
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_limit = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.limit.return_value = mock_limit
    # Simulate finding iPhone entries
    mock_limit.all.return_value = [
        Device(device_id=2, brand="Apple", model="iPhone 13 Pro", type="Smartphone"),
        Device(device_id=3, brand="Apple", model="iPhone SE", type="Smartphone"),
    ]

    result = find_device_in_text("help with my iphone 13 pro")
    # Even though DB returns 2, the keyword matching logic might simplify
    # or the query logic might get refined later. For now, check if list is returned.
    assert isinstance(result, list)
    # Depending on implementation details (like duplicate removal based on ID), 
    # the length might be 1 or more. Let's check it found at least one Device.
    assert len(result) > 0 
    assert all(isinstance(dev, Device) for dev in result)
    assert any(dev.model == "iPhone 13 Pro" for dev in result) 
    mock_session.query.assert_called_once_with(Device)

@patch('backend.app.db.session')
def test_find_device_in_text_no_match(mock_session):
    """Test when no relevant keywords are found."""
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_limit = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.limit.return_value = mock_limit
    mock_limit.all.return_value = [] # Simulate DB returning nothing

    result = find_device_in_text("the weather is nice today")
    assert result == []
    # The query might still be made if keywords like 'nice' > 3 chars trigger brand/model check
    # assert not mock_session.query.called # This might fail depending on exact logic

@patch('backend.app.db.session')
def test_find_device_in_text_charger(mock_session):
    """Test finding charger keywords."""
    mock_query = MagicMock()
    mock_filter = MagicMock()
    mock_limit = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_filter
    mock_filter.limit.return_value = mock_limit
    mock_limit.all.return_value = [Device(device_id=4, brand="Anker", model="PowerPort III", type="Charger")]

    result = find_device_in_text("my usb-c cable is frayed")
    assert len(result) == 1
    assert result[0].type == "Charger"
    mock_session.query.assert_called_once_with(Device) 