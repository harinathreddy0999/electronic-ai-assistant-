# API Design - Electronics Virtual Technician

This document outlines the API endpoints for communication between the frontend client and the backend server.

## Conventions

*   Base URL: `/api`
*   Authentication: (To be defined - likely token-based)
*   Data Format: JSON

## Endpoints

### 1. Sessions

*   **Endpoint:** `POST /api/sessions`
*   **Description:** Initiates a new troubleshooting session.
*   **Request Body:**
    ```json
    {
      "user_id": "optional_user_identifier",
      "initial_device_guess": "optional_string" 
    }
    ```
*   **Response Body (Success - 201 Created):**
    ```json
    {
      "session_id": "unique_session_identifier"
    }
    ```
*   **Response Body (Error):** Standard error format (e.g., 4xx/5xx with error message).

### 2. Messages

*   **Endpoint:** `POST /api/sessions/{session_id}/messages`
*   **Description:** Sends user's message (transcribed text) to the backend for processing and receives the agent's reply.
*   **Path Parameters:**
    *   `session_id`: The unique identifier for the active session.
*   **Request Body:**
    ```json
    {
      "text": "User's transcribed speech or text input."
    }
    ```
*   **Response Body (Success - 200 OK):**
    ```json
    {
      "reply_text": "Agent's response text to be synthesized or displayed.",
      "diagnostic_state": { ... } // Optional: Information about the current diagnostic state
    }
    ```
*   **Response Body (Error):** Standard error format.

### 3. Location Query

*   **Endpoint:** `GET /api/sessions/{session_id}/location_query`
*   **Description:** Requests relevant local repair shops or retailers based on location and context.
*   **Path Parameters:**
    *   `session_id`: The unique identifier for the active session (to provide context if needed).
*   **Query Parameters:**
    *   `latitude` (required, float): User's latitude.
    *   `longitude` (required, float): User's longitude.
    *   `device_type` (optional, string): e.g., "Laptop", "iPhone 13", "USB-C Charger". Inferred from session if possible.
    *   `issue_type` (optional, string): e.g., "Cracked Screen", "Battery Replacement", "Charging Port". Inferred from session if possible.
    *   `query_override` (optional, string): Allow specific search query like "Apple Authorized Service Provider".
*   **Response Body (Success - 200 OK):**
    ```json
    {
      "locations": [
        {
          "name": "Example Repair Shop",
          "address": "123 Main St, Aurora, IL 60506",
          "phone": "(630) 555-1234",
          "distance_miles": 1.5, // Approximate distance
          "type": "Electronics Repair", // e.g., Repair, Retailer, Authorized Service Provider
          "maps_url": "optional_google_maps_link"
        }
        // ... more locations
      ]
    }
    ```
*   **Response Body (Error):** Standard error format (e.g., 400 Bad Request if location is missing, 404 Not Found if no results).

## Future Considerations

*   Authentication/Authorization endpoints.
*   Endpoints related to video stream setup/signaling (WebRTC might handle data transfer directly).
*   Endpoint for providing explicit feedback on diagnosis/suggestions.
*   WebSockets for more persistent communication? 