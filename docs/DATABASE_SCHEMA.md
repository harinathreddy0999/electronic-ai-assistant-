# Database Schema - Electronics Virtual Technician

This document outlines the proposed database schema. Specific field types and constraints will depend on the chosen database system (e.g., PostgreSQL, MongoDB) and ORM/ODM.

## Table: Users (Optional)

*Purpose: Stores user account information if login/profiles are implemented.*

*   `user_id` (Primary Key): Unique identifier for the user (e.g., UUID, auto-incrementing integer).
*   `username` (String, Unique, Optional): User's chosen username.
*   `email` (String, Unique, Optional): User's email address.
*   `password_hash` (String, Optional): Hashed password.
*   `created_at` (Timestamp): When the user account was created.
*   `updated_at` (Timestamp): When the user account was last updated.

*Note: User accounts might be optional for basic functionality.*

## Table: Sessions

*Purpose: Stores information about each troubleshooting session.*

*   `session_id` (Primary Key): Unique identifier for the session (e.g., UUID). Matches the ID used in the API.
*   `user_id` (Foreign Key, Nullable): Links to the `Users` table if the user is logged in.
*   `start_time` (Timestamp): When the session began.
*   `end_time` (Timestamp, Nullable): When the session ended.
*   `identified_device_id` (Foreign Key, Nullable): Links to the `Devices` table once identified.
*   `diagnostic_summary` (Text, Nullable): A summary of the final diagnosis or outcome.
*   `last_location_query` (JSON, Nullable): Stores the parameters of the last location query (lat, lon, device, issue).
*   `status` (String): e.g., "active", "ended", "escalated_to_repair".
*   `created_at` (Timestamp)
*   `updated_at` (Timestamp)

## Table: Messages

*Purpose: Stores the conversation history for each session.*

*   `message_id` (Primary Key): Unique identifier for the message.
*   `session_id` (Foreign Key): Links to the `Sessions` table.
*   `timestamp` (Timestamp): When the message was sent/received.
*   `sender` (String): "user" or "agent".
*   `text` (Text): The content of the message (user transcription or agent reply).
*   `sequence_number` (Integer): Order of the message within the session.

## Table: Devices (Knowledge Base Core)

*Purpose: Stores information about known electronic devices.*

*   `device_id` (Primary Key): Unique identifier for the device model/type.
*   `brand` (String): e.g., "Apple", "Samsung", "Dell".
*   `model` (String): e.g., "MacBook Pro 14-inch M1", "Galaxy S21", "XPS 13".
*   `type` (String): e.g., "Laptop", "Smartphone", "Charger", "Router", "Mouse".
*   `release_year` (Integer, Nullable)
*   `specifications` (JSON, Nullable): Key technical specs.
*   `common_issues` (JSON, Nullable): List or structure describing frequent problems.
*   `troubleshooting_guides` (JSON, Nullable): Structured steps for common fixes.
*   `visual_identifiers` (JSON, Nullable): Data to help the CV model (e.g., expected port locations, button layouts, logo features - Requires significant effort).
*   `created_at` (Timestamp)
*   `updated_at` (Timestamp)

## Table: RepairLocations (Optional Cache)

*Purpose: Optionally cache results from mapping service queries to reduce API calls.*

*   `location_id` (Primary Key): Unique identifier (e.g., Google Place ID).
*   `name` (String)
*   `address` (String)
*   `phone` (String, Nullable)
*   `latitude` (Float)
*   `longitude` (Float)
*   `type` (String): e.g., "Electronics Repair", "Retailer".
*   `cached_at` (Timestamp): When this data was fetched.
*   `source` (String): e.g., "Google Places API".

## Relationships

*   One `User` can have multiple `Sessions`.
*   One `Session` belongs to zero or one `User`.
*   One `Session` can have multiple `Messages`.
*   One `Message` belongs to one `Session`.
*   One `Session` relates to zero or one `Device` (once identified).
*   One `Device` can be related to multiple `Sessions`. 