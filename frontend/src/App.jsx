import { useState, useEffect, useRef } from 'react'
import io from 'socket.io-client'; // Added socket.io-client import
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css';
import './App.css' // Keep basic styling for now
// Add Leaflet Icon fix for bundlers like Vite/Webpack
import L from 'leaflet';
import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png';
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import shadowUrl from 'leaflet/dist/images/marker-shadow.png';

// --- Constants ---
// Use hardcoded URLs for local development
// For deployment, use environment variables with VITE_ prefix (e.g., import.meta.env.VITE_SOCKET_URL)
const SOCKET_URL = 'http://localhost:5001'; 
const API_BASE_URL = 'http://localhost:5001/api';
// ----------------

// Check for Web Speech API support
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = SpeechRecognition ? new SpeechRecognition() : null;
const synth = window.speechSynthesis;

if (recognition) {
  recognition.continuous = false; // Process speech after pauses
  recognition.lang = 'en-US';
  recognition.interimResults = false; // We only want final results
  recognition.maxAlternatives = 1;
}

// --- WebRTC Configuration ---
// Example using Google's public STUN servers
const pcConfig = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
    // Add TURN server URLs here if needed for NAT traversal
  ]
};
// --------------------------

// Simple spinner component (can be replaced with a library)
const Spinner = () => <div className="spinner"></div>;

function App() {
  console.log("--- App component rendering ---"); // Add this log

  const [sessionId, setSessionId] = useState(null);
  const [isLoadingSession, setIsLoadingSession] = useState(false); // Specific loading for session start
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [error, setError] = useState(null);
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState("");
  // Location specific state
  const [locations, setLocations] = useState([]); // Added state for locations
  const [isFindingLocation, setIsFindingLocation] = useState(false); // Added state for location search loading
  const [locationError, setLocationError] = useState(null); // Added state for location errors
  // --- Voice State ---
  const [isListening, setIsListening] = useState(false);
  const [micError, setMicError] = useState(null);
  const recognitionRef = useRef(recognition);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const utteranceRef = useRef(null);
  // --- End Voice State ---
  // Video State
  const [isVideoStarting, setIsVideoStarting] = useState(false); // Specific loading for video
  const [isVideoEnabled, setIsVideoEnabled] = useState(false);
  const [videoError, setVideoError] = useState(null);
  const [localStream, setLocalStream] = useState(null);
  // Socket.IO State
  const [socket, setSocket] = useState(null);
  // WebRTC State
  const [peerConnection, setPeerConnection] = useState(null);
  const [webRtcError, setWebRtcError] = useState(null);
  const [webRtcStatus, setWebRtcStatus] = useState('inactive'); // More detailed WebRTC status
  const [cvResults, setCvResults] = useState([]); // Added state for CV results
  const [userCoords, setUserCoords] = useState(null); // Added state for user lat/lon
  const [isInGuide, setIsInGuide] = useState(false); // State for active guide status
  const [identifiedDevice, setIdentifiedDevice] = useState(null); // Added state for identified device {brand, model, type, source}

  const messagesEndRef = useRef(null);
  const localVideoRef = useRef(null);
  const sessionIdRef = useRef(sessionId); // Ref to hold the latest session ID

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages]);

  // Update sessionIdRef whenever sessionId state changes
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  // --- Socket.IO Connection Effect ---
  useEffect(() => {
    if (sessionId) {
        // Connect to Socket.IO server when session starts
        // Ensure path option matches backend configuration if needed
        const newSocket = io(SOCKET_URL, {
            // path: '/socket.io' // Default path, adjust if backend is different
            // Add authentication here if implemented later (e.g., query: { token: '... ' })
        });
        setSocket(newSocket);
        console.log(`Socket connecting for session: ${sessionId}...`);

        newSocket.on('connect', () => {
            console.log(`Socket connected: ${newSocket.id}`);
            // Join the session room
            newSocket.emit('join_session', { session_id: sessionId });
        });

        newSocket.on('disconnect', (reason) => {
            console.log(`Socket disconnected: ${reason}`);
            setSocket(null); // Clear socket state on disconnect
            // Handle potential reconnection logic here if needed
        });

        newSocket.on('connect_error', (err) => {
            console.error("Socket connection error:", err);
            setError(`Socket connection failed: ${err.message}`);
            setSocket(null);
        });

        // Updated Signal Listener for WebRTC
        newSocket.on('signal', async (data) => {
            console.log('Received signal:', data);
            if (!peerConnection) {
                console.error("Received signal but no peer connection is established.");
                return;
            }
            try {
                if (data.signal.type === 'answer') {
                    console.log("Setting remote description (answer)...");
                    await peerConnection.setRemoteDescription(new RTCSessionDescription(data.signal));
                } else if (data.signal.candidate) {
                    console.log("Adding ICE candidate...");
                    await peerConnection.addIceCandidate(new RTCIceCandidate(data.signal.candidate));
                } else {
                     console.warn("Received unknown signal type:", data.signal);
                }
            } catch (err) {
                 console.error("Error handling received signal:", err);
                 setWebRtcError(`Error handling signal: ${err.message}`);
            }
        });

        // **** Added CV Results Listener ****
        newSocket.on('cv_results', (data) => {
            console.log('Received CV results:', data.detections);
            // Update state with the latest detections
            setCvResults(data.detections || []); 
            // TODO: Potentially use these results to update context for the agent
            // or display bounding boxes.
        });
        // **********************************

        // Cleanup on component unmount or session change
        return () => {
            console.log("Disconnecting socket...");
            newSocket.disconnect();
            setSocket(null);
        };
    } else {
        // If sessionId becomes null, ensure socket is disconnected
        if (socket) {
            console.log("Session ended, disconnecting socket...");
            socket.disconnect();
            setSocket(null);
        }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, peerConnection]); // Add peerConnection dependency for signal handler
  // --- End Socket.IO Connection Effect ---

  // --- Speech Recognition Effect Hook ---
  useEffect(() => {
    const currentRecognition = recognitionRef.current;
    if (!currentRecognition) {
      setMicError("Speech recognition not supported by this browser.");
      return;
    }
    // --- Handler Setup ---
    const handleResult = (event) => {
      const transcript = event.results[0][0].transcript;
      console.log('Transcript:', transcript);
      // Access latest sessionId via ref
      const currentSessionId = sessionIdRef.current;
      if (currentSessionId) {
           console.log(`STT handleResult: Found sessionId ${currentSessionId}, calling sendMessage.`);
           sendMessage(transcript);
      } else {
           console.error("STT handleResult: sessionIdRef.current is null, cannot send message.");
           // Optionally inform the user via state/message?
      }
      setIsListening(false); 
    };
    const handleError = (event) => { 
        console.error('Speech recognition error', event.error);
        setMicError(`Mic Error: ${event.error}`); 
        setIsListening(false); 
    };
    const handleEnd = () => {
      console.log('Speech recognition ended.');
      setIsListening(false);
    };
    // --- Attach Handlers ---
    currentRecognition.addEventListener('result', handleResult);
    currentRecognition.addEventListener('error', handleError);
    currentRecognition.addEventListener('end', handleEnd);
    // --- Cleanup ---
    return () => {
      currentRecognition.removeEventListener('result', handleResult);
      currentRecognition.removeEventListener('error', handleError);
      currentRecognition.removeEventListener('end', handleEnd);
      try { currentRecognition.stop(); } catch(e) { /* Ignore */ }
    };
  }, []); // Empty dependency array, runs once on mount
  // --- End Speech Recognition Effect Hook ---

  // --- Speech Synthesis Effect Hook ---
  useEffect(() => {
    const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;

    if (lastMessage && lastMessage.sender === 'agent' && !isSpeaking) {
      speak(lastMessage.text);
    }
    // Cancel speech if component unmounts or message changes abruptly
    return () => {
      if (synth.speaking) {
        synth.cancel();
        setIsSpeaking(false);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages]); // Depend on messages array
  // --- End Speech Synthesis Effect Hook ---

  // --- TTS Function ---
  const speak = (text) => {
    if (!synth) {
      console.error("Speech synthesis not supported.");
      return;
    }
    if (synth.speaking) {
      console.warn("Speech synthesis already speaking, cancelling previous.");
      synth.cancel(); 
    }

    // Simple text replacements for better pronunciation
    let textToSpeak = text;
    textToSpeak = textToSpeak.replace(/e\.g\./gi, "for example");
    textToSpeak = textToSpeak.replace(/i\.e\./gi, "that is");
    // Add more replacements as needed

    utteranceRef.current = new SpeechSynthesisUtterance(textToSpeak);

    // --- Voice Selection Logic ---
    try {
        const voices = synth.getVoices();
        if (voices.length > 0) {
            // Try to find a non-default US or GB English voice
            let selectedVoice = voices.find(voice => voice.lang === 'en-US' && !voice.default);
            if (!selectedVoice) {
                selectedVoice = voices.find(voice => voice.lang === 'en-GB' && !voice.default);
            }
            // Fallback to any US English voice
            if (!selectedVoice) {
                selectedVoice = voices.find(voice => voice.lang === 'en-US');
            }
            // Fallback to any GB English voice
             if (!selectedVoice) {
                selectedVoice = voices.find(voice => voice.lang === 'en-GB');
            }
            // Fallback to the first available voice if no English found (less ideal)
             if (!selectedVoice) {
                 selectedVoice = voices[0];
             }

            if (selectedVoice) {
                utteranceRef.current.voice = selectedVoice;
                console.log("TTS using voice:", selectedVoice.name, selectedVoice.lang);
            } else {
                 console.log("Could not find a preferred voice, using browser default.");
            }
        } else {
             console.log("No voices available from synth.getVoices() at this time.");
        }
    } catch (e) {
         console.error("Error during voice selection:", e);
    }
    // --- End Voice Selection Logic ---

    utteranceRef.current.onstart = () => {
        console.log("Speech synthesis started.");
        setIsSpeaking(true);
    };
    utteranceRef.current.onend = () => {
      console.log("Speech synthesis ended.");
      setIsSpeaking(false);
    };
    utteranceRef.current.onerror = (event) => {
      console.error("Speech synthesis error", event);
      setIsSpeaking(false);
    };

    synth.speak(utteranceRef.current);
  };
  // --- End TTS Function ---

  const startSession = async () => {
    setIsLoadingSession(true); // Use specific loader
    setError(null);
    setMicError(null);
    setSessionId(null);
    setMessages([]);
    setLocations([]); // Clear locations on new session
    setLocationError(null); // Clear location errors
    setUserCoords(null); // Clear user coords
    if (synth.speaking) synth.cancel(); // Stop any ongoing speech
    setIsSpeaking(false);
    if (localStream) { // Stop existing video stream if any
        localStream.getTracks().forEach(track => track.stop());
        setLocalStream(null);
    }
    setIsVideoEnabled(false);
    if (peerConnection) { peerConnection.close(); setPeerConnection(null); } // Close existing peer connection
    setCvResults([]); // Clear CV results
    setWebRtcStatus('inactive'); setWebRtcError(null); // Reset WebRTC status
    setIsInGuide(false); // Reset guide state
    setIdentifiedDevice(null); // Reset identified device

    try {
      const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        let errorMsg = `HTTP error! status: ${response.status}`;
        try {
            const errorData = await response.json();
            errorMsg = errorData.error || errorMsg;
        } catch (e) { /* Ignore */ }
        throw new Error(errorMsg);
      }

      const data = await response.json();
      setSessionId(data.session_id);
      console.log("Session started:", data.session_id);

    } catch (err) {
      console.error("Failed to start session:", err);
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setIsLoadingSession(false);
    }
  };

  const sendMessage = async (textToSend = null) => {
    console.log("sendMessage called. textToSend:", textToSend, "currentMessage:", currentMessage);
    const messageText = textToSend ?? currentMessage;
    console.log("Effective messageText:", messageText);
    const currentSessionId = sessionIdRef.current; // Use ref here too for consistency
    
    // Use the ref for the check
    if (!messageText.trim()) { console.log("sendMessage aborted: messageText is empty/whitespace."); return; }
    if (!currentSessionId) { console.log("sendMessage aborted: sessionIdRef.current is null."); return; }
    if (isSendingMessage) { console.log("sendMessage aborted: already sending a message."); return; }
    
    console.log("sendMessage proceeding with session:", currentSessionId);

    const userMessage = { sender: 'user', text: messageText };
    // Log state *before* setting
    console.log("Messages state BEFORE optimistic update:", messages);
    setMessages(prevMessages => {
        const newMessages = [...prevMessages, userMessage];
        console.log("Messages state AFTER optimistic update:", newMessages);
        return newMessages;
    });
    
    // Only clear text input if message *wasn't* passed directly (i.e. came from typing)
    if (!textToSend) {
        setCurrentMessage(""); 
    }
    setIsSendingMessage(true);
    // ... (Reset errors, stop synth)
    setError(null); setMicError(null); setLocationError(null); setVideoError(null); setWebRtcError(null);
    if (synth.speaking) synth.cancel(); setIsSpeaking(false);

    try {
       console.log("sendMessage: Sending fetch request to backend...");
       const response = await fetch(`${API_BASE_URL}/sessions/${currentSessionId}/messages`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: messageText }),
        });
        console.log("sendMessage: Fetch response received, status:", response.status);
        
        if (!response.ok) { 
            let errorMsg = `HTTP error! status: ${response.status}`;
            try { const errorData = await response.json(); errorMsg = errorData.error || errorData.details || errorMsg; } catch (e) { /* Ignore */ }
            console.error("sendMessage fetch error:", errorMsg);
            throw new Error(errorMsg);
        }
        const data = await response.json();
        console.log("sendMessage: Backend reply data:", data);
        const agentMessage = { sender: 'agent', text: data.reply_text };
        setMessages(prevMessages => [...prevMessages, agentMessage]);
        
        // Update guide state 
        if (data.diagnostic_state && typeof data.diagnostic_state.in_guide === 'boolean') {
            setIsInGuide(data.diagnostic_state.in_guide);
            console.log(`Guide state updated: ${data.diagnostic_state.in_guide}`);
        }
        // Update identified device state
        if (data.diagnostic_state && data.diagnostic_state.identified_device) {
            setIdentifiedDevice(data.diagnostic_state.identified_device);
            console.log("Identified device updated:", data.diagnostic_state.identified_device);
        } else {
             // setIdentifiedDevice(null); 
        }

    } catch (err) {
       console.error("sendMessage caught error:", err);
       setError(err.message || "Failed to get reply.");
       setMessages(prev => [...prev, { sender: 'system', text: `Error: ${err.message}` }]);
       setIsInGuide(false); 
    } finally {
        console.log("sendMessage finally block, setting isSendingMessage to false.");
        setIsSendingMessage(false);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  // Function to get location and find nearby repair options
  const findLocalRepair = () => {
    if (!navigator.geolocation) {
      setLocationError("Geolocation is not supported by your browser.");
      return;
    }

    setIsFindingLocation(true);
    setLocationError(null);
    setLocations([]);
    setUserCoords(null); // Clear previous coords
    if (synth.speaking) synth.cancel(); // Stop TTS
    setIsSpeaking(false);

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { latitude, longitude } = position.coords;
        setUserCoords({ lat: latitude, lng: longitude }); // Store user coords
        console.log(`Location obtained: Lat: ${latitude}, Lon: ${longitude}`);
        // Add a message to chat indicating location lookup
        setMessages(prev => [...prev, { sender: 'system', text: 'Looking for nearby repair options...'}]);

        try {
          // TODO: Get device_type and issue_type from session context or state
          const deviceType = ""; // Placeholder
          const issueType = ""; // Placeholder

          const queryParams = new URLSearchParams({
            latitude: latitude,
            longitude: longitude,
            ...(deviceType && { device_type: deviceType }),
            ...(issueType && { issue_type: issueType }),
          });

          const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/location_query?${queryParams}`);

          if (!response.ok) {
            let errorMsg = `HTTP error! status: ${response.status}`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || errorData.details || errorMsg;
            } catch (e) { /* Ignore */ }
            throw new Error(errorMsg);
          }

          const data = await response.json();
          setLocations(data.locations || []);
          if (!data.locations || data.locations.length === 0) {
             setMessages(prev => [...prev, { sender: 'system', text: 'No nearby locations found matching the criteria.'}]);
          }
          console.log("Locations found:", data.locations);

        } catch (err) {
          console.error("Failed to fetch locations:", err);
          setLocationError(err.message || "Failed to fetch repair locations.");
           setMessages(prev => [...prev, { sender: 'system', text: `Error finding locations: ${err.message}`}]);
        } finally {
          setIsFindingLocation(false);
        }
      },
      (error) => {
        console.error("Geolocation error:", error);
        let errorMsg = "An unknown error occurred while getting location.";
        switch(error.code) {
          case error.PERMISSION_DENIED:
            errorMsg = "Geolocation permission denied. Please enable location services in your browser settings.";
            break;
          case error.POSITION_UNAVAILABLE:
            errorMsg = "Location information is unavailable.";
            break;
          case error.TIMEOUT:
            errorMsg = "The request to get user location timed out.";
            break;
        }
        setLocationError(errorMsg);
        setMessages(prev => [...prev, { sender: 'system', text: `Geolocation Error: ${errorMsg}`}]);
        setIsFindingLocation(false);
      },
      { timeout: 10000 } // Optional timeout
    );
  };

  // --- Toggle Listening Function ---
  const toggleListening = () => {
    const currentRecognition = recognitionRef.current;
    if (!currentRecognition) {
        setMicError("Speech recognition not supported or not initialized.");
        return;
    }
    if (isListening) {
      currentRecognition.stop();
      setIsListening(false);
    } else {
      try {
        if (synth.speaking) synth.cancel(); // Stop TTS before listening
        setIsSpeaking(false);
        currentRecognition.start();
        setIsListening(true);
        setMicError(null); // Clear previous mic errors
        console.log("Speech recognition started");
      } catch (err) {
        // Handle errors like starting recognition when already started
        console.error("Error starting speech recognition:", err);
        setMicError(`Could not start microphone: ${err.message}`);
        setIsListening(false);
      }
    }
  };
  // --- End Toggle Listening Function ---

  // --- Video Function --- 
  const toggleVideo = async () => {
    setVideoError(null);
    setWebRtcError(null);

    if (isVideoEnabled && localStream) {
      // Stop the stream
      localStream.getTracks().forEach(track => track.stop());
      setLocalStream(null);
      setIsVideoEnabled(false);
      setIsVideoStarting(false); // Ensure this is false
      closePeerConnection(); // Close WebRTC connection when video stops
      console.log("Video stream stopped & PeerConnection closed.");
      setIdentifiedDevice(null); // Clear identified device when video stops
    } else {
      // Start the stream
      setIsVideoStarting(true); // Set loading state for video start
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        setLocalStream(stream);
        setIsVideoEnabled(true);
        console.log("Video stream started.");
        // Setup WebRTC connection *after* getting the stream
        // Let an effect handle this based on localStream and socket availability?
        // setupPeerConnection(); // Call directly here for now
      } catch (err) {
        console.error("Error accessing camera:", err);
        setVideoError(`Error accessing camera: ${err.name} - ${err.message}`);
        setIsVideoEnabled(false); setLocalStream(null);
        closePeerConnection(); // Ensure cleanup on error
      } finally {
          setIsVideoStarting(false); // Clear loading state
      }
    }
  };
  // --- End Video Function ---

  // --- WebRTC Setup/Teardown ---
  const setupPeerConnection = () => {
    if (!localStream || !socket || peerConnection) return; // Need stream, socket, and no existing connection
    console.log("Setting up PeerConnection...");
    setWebRtcError(null);
    setWebRtcStatus('connecting'); // Update status

    try {
      const pc = new RTCPeerConnection(pcConfig);
      setPeerConnection(pc);

      // Handle ICE candidates
      pc.onicecandidate = (event) => {
        if (event.candidate && socket) {
          console.log("Sending ICE candidate...");
          socket.emit('signal', { 
              session_id: sessionId, 
              signal: { candidate: event.candidate } 
          });
        } else {
            console.log("All ICE candidates have been sent.");
        }
      };

      // Handle connection state changes (for debugging/info)
      pc.oniceconnectionstatechange = () => {
        console.log(`ICE Connection State: ${pc.iceConnectionState}`);
        if (['failed', 'disconnected', 'closed'].includes(pc.iceConnectionState)) {
             setWebRtcError(`WebRTC connection state: ${pc.iceConnectionState}`);
             // Consider cleanup or attempting restart here?
        }
      };

       // Add tracks from local stream to the connection
       localStream.getTracks().forEach(track => {
            console.log(`Adding track: ${track.kind}`);
            pc.addTrack(track, localStream);
       });

       // Create offer and send it
       createOfferAndSend(pc);

    } catch (err) {
        console.error("Error setting up PeerConnection:", err);
        setWebRtcError(`PeerConnection setup failed: ${err.message}`);
        if (peerConnection) { peerConnection.close(); } // Attempt cleanup
        setPeerConnection(null);
        setWebRtcStatus('failed'); // Update status
    }
  };

  const createOfferAndSend = async (pc) => {
     if (!pc || !socket) return;
     try {
        console.log("Creating SDP offer...");
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        console.log("Sending SDP offer...");
        socket.emit('signal', { 
            session_id: sessionId,
            signal: { type: 'offer', sdp: pc.localDescription.sdp } 
        });
     } catch (err) {
        console.error("Error creating/sending offer:", err);
        setWebRtcError(`Failed to create/send offer: ${err.message}`);
     }
  };

  const closePeerConnection = () => {
      if (peerConnection) {
          console.log("Closing PeerConnection...");
          peerConnection.close();
          setPeerConnection(null);
          setWebRtcStatus('closed'); // Update status
      }
      setCvResults([]); // Clear CV results when connection closes
      setWebRtcError(null);
      setIsInGuide(false); // Assume closing video stops any active guide? Or handle separately?
      setIdentifiedDevice(null); // Clear identified device when video stops
  };
  // --- End WebRTC Setup/Teardown ---

  // --- PeerConnection Setup Effect Hook ---
  useEffect(() => {
      if (localStream && socket && !peerConnection && isVideoEnabled) {
          setupPeerConnection();
      }
      // Cleanup effect if stream or socket goes away
      return () => {
          // If component unmounts or dependencies change such that the condition is false,
          // ensure connection is closed if it was initiated by this effect.
          // The logic inside toggleVideo already handles closing when video is explicitly stopped.
          // This cleanup might be redundant or could interfere if not careful.
          // For now, rely on toggleVideo and session end for cleanup.
      };
  }, [localStream, socket, peerConnection, isVideoEnabled]); // Dependencies
  // --- End PeerConnection Setup Effect ---

  // --- Attach local stream to video element ---
  useEffect(() => {
     if (localStream && localVideoRef.current) { localVideoRef.current.srcObject = localStream; }
     else if (!localStream && localVideoRef.current) { localVideoRef.current.srcObject = null; }
  }, [localStream]);
  // --- End Attach local stream to video element ---

  // Update WebRTC Status handler
  useEffect(() => {
      if (peerConnection) {
        const handleConnectionStateChange = () => {
             if (peerConnection) { // Check again as state might change during async handler
                setWebRtcStatus(peerConnection.connectionState);
                console.log(`WebRTC Connection State: ${peerConnection.connectionState}`);
                if (['failed', 'disconnected', 'closed'].includes(peerConnection.connectionState)) {
                    setWebRtcError(`WebRTC connection state: ${peerConnection.connectionState}`);
                }
             }
        };
        peerConnection.addEventListener('connectionstatechange', handleConnectionStateChange);
        // Initial state check
        handleConnectionStateChange(); 
        return () => {
          if (peerConnection) {
             peerConnection.removeEventListener('connectionstatechange', handleConnectionStateChange);
          }
        };
      } else {
        setWebRtcStatus('inactive'); // Reset status if PC is null
      }
  }, [peerConnection]);

  // --- Leaflet Icon Fix Effect (Run Once) ---
  useEffect(() => {
    delete L.Icon.Default.prototype._getIconUrl;
    L.Icon.Default.mergeOptions({
        iconRetinaUrl: iconRetinaUrl,
        iconUrl: iconUrl,
        shadowUrl: shadowUrl,
    });
    console.log("Leaflet default icon paths set inside useEffect.");
  }, []); // Empty dependency array ensures it runs only once
  // --- End Leaflet Icon Fix Effect ---

  return (
    <div className="App">
      <h1>Electronics Virtual Technician</h1>

      {!sessionId && (
        <button onClick={startSession} disabled={isLoadingSession}>
          {isLoadingSession ? <Spinner /> : 'Start Troubleshooting Session'}
        </button>
      )}

      {sessionId && (
        <div className="session-active">
          
          {/* Left Column: Chat, Location, Audio Status */} 
          <div className="left-column">
             {/* Identified Device Display */} 
             {identifiedDevice && (
                 <div className="identified-device-display">
                     Identified: 
                     <strong>{identifiedDevice.brand || ''} {identifiedDevice.model || identifiedDevice.type || 'Unknown Device'}</strong> 
                     <span className="source-tag">(via {identifiedDevice.source || '?'})</span>
                 </div>
             )}
             {/* Guide Status Indicator */} 
             {isInGuide && (
                <div className="guide-status-indicator">
                    <p>➡️ Following Troubleshooting Guide...</p>
                </div>
             )}
             {/* Chat Interface */}
             <div className="chat-container">
                 <div className="messages-area">
                    {messages.map((msg, index) => (
                        <div key={index} className={`message-row message-${msg.sender}`}>
                         {/* Add sender label optionally? */} 
                         {/* {msg.sender !== 'system' && <span className="sender-label">{msg.sender === 'user' ? 'You' : 'Agent'}</span>} */} 
                         <div className={`message-bubble message-bubble-${msg.sender}`}>
                            {typeof msg.text === 'string' ? msg.text.split('\n').map((line, i) => (
                                <span key={i}>{line}<br/></span>
                            )) : JSON.stringify(msg.text)}
                         </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                 </div>
                 <div className="input-area">
                    <textarea
                        value={currentMessage}
                        onChange={(e) => setCurrentMessage(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Type or use mic..."
                        rows={2} 
                        className="chat-input"
                    />
                    {recognitionRef.current && (
                        <button className="mic-button" onClick={toggleListening} disabled={isSendingMessage} title={isListening ? "Stop Listening" : "Start Listening"} style={{ background: isListening ? 'var(--mic-active-bg)' : 'var(--mic-idle-bg)' }}> 🎙️ </button>
                    )}
                     <button className="send-button" onClick={() => sendMessage()} disabled={isSendingMessage || !currentMessage.trim() || isListening}>
                        {isSendingMessage ? <Spinner/> : 'Send'}
                    </button>
                 </div>
             </div>

            {/* Location Finder Section */}
            <div className="location-finder">
                <h4>Find Repair Options {isFindingLocation && <Spinner/>}</h4>
                <button onClick={findLocalRepair} disabled={isFindingLocation}> Find Nearby </button>
                {locationError && <div className="error-text"><p>{locationError}</p></div>}
                
                {/* Map Display */} 
                {locations.length > 0 && userCoords && (
                    <div className="map-container" style={{ height: '300px', width: '100%', marginTop: '1rem', borderRadius: '8px', overflow: 'hidden' }}>
                        <MapContainer center={[userCoords.lat, userCoords.lng]} zoom={13} scrollWheelZoom={false} style={{ height: "100%", width: "100%" }}>
                            <TileLayer
                                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                            />
                            {/* User Location Marker */} 
                             <Marker position={[userCoords.lat, userCoords.lng]}>
                                <Popup>Your approximate location</Popup>
                             </Marker>
                            {/* Repair Location Markers */} 
                            {locations.map((loc, index) => (
                                loc.latitude && loc.longitude && (
                                    <Marker key={loc.google_place_id || index} position={[loc.latitude, loc.longitude]}>
                                        <Popup>
                                            <strong>{loc.name}</strong><br />
                                            {loc.address}
                                            {loc.maps_url && <div><a href={loc.maps_url} target="_blank" rel="noopener noreferrer">View on Google Maps</a></div>}
                                        </Popup>
                                    </Marker>
                                )
                            ))}
                        </MapContainer>
                    </div>
                 )}

                 {/* Location List (remains) */} 
                 {locations.length > 0 && (
                    <div className="locations-list">
                        <h4>Nearby Options:</h4>
                        <ul>
                        {locations.map((loc, index) => (
                            <li key={loc.google_place_id || index}>
                            <strong>{loc.name}</strong>
                            <p>{loc.address}</p>
                            {loc.maps_url && <a href={loc.maps_url} target="_blank" rel="noopener noreferrer">View on Map</a>}
                            </li>
                        ))}
                        </ul>
                    </div>
                 )}
            </div>

             {/* Audio Interface Status */}
             <div className="audio-controls">
                <h4>Audio Status</h4>
                {micError && <p>Mic Error: {micError}</p>}
                <div className="mic-status"> <span>🎙️ Mic:</span> <span>{isListening ? "Listening..." : "Idle"}</span> </div>
                <div className="audio-output"> <span>🔊 Playback:</span> <span>{isSpeaking ? "Speaking..." : "Idle"}</span> </div>
             </div>

             {/* WebRTC Status/Error */} 
             <div className="webrtc-status">
                 <h4>WebRTC Status</h4>
                 <p> 
                     Connection State: 
                     <span>{peerConnection ? (peerConnection.connectionState || 'connecting...') : 'inactive'}</span>
                 </p>
                 {webRtcError && <p>WebRTC Error: {webRtcError}</p>} 
             </div>

          </div> {/* End Left Column */} 

          {/* Right Column: Video */} 
          <div className="right-column">
             <h4>Video Feed {isVideoStarting && <Spinner/>}</h4>
             {videoError && <p className="error-text">{videoError}</p>} 
             <div className="video-display">
                 <video 
                     ref={localVideoRef} 
                     autoPlay 
                     playsInline 
                     muted // Mute local playback to avoid feedback
                     style={{ width: '100%', height: '100%', display: isVideoEnabled ? 'block' : 'none' }} 
                 />
                 {!isVideoEnabled && <p>Video is off</p>} 
                 {/* CV Analysis Indicator */} 
                 {isVideoEnabled && webRtcStatus === 'connected' && (
                     <div className="cv-analysis-indicator">
                        <Spinner /> Analyzing...
                     </div>
                 )}
             </div>
             <button onClick={toggleVideo} style={{ marginTop: '0.5rem' }} disabled={isLoadingSession || isVideoStarting || !socket}> 
                 {isVideoEnabled ? 'Stop Camera' : 'Start Camera'}
             </button>
             <p className="video-note">
                 (Video stream analysis active when camera is on...)
             </p>
             {/* **** Added CV Results Display **** */} 
             <div className="cv-results-display">
                 <strong>Detected Objects:</strong> 
                 {cvResults.length > 0 ? (
                     <ul>
                         {cvResults.map((res, i) => ( 
                             <li key={i}>{res.label} ({ (res.confidence * 100).toFixed(0) }%)</li>
                         ))}
                     </ul>
                 ) : (
                     <span className="status-indicator idle"> None</span>
                 )}
             </div>
             {/* ********************************* */} 
          </div> {/* End Right Column */} 

        </div>
      )}

      {error && (
        <div className="error-text">
          <p>Error: {error}</p>
        </div>
      )}

    </div>
  );
}

export default App;
