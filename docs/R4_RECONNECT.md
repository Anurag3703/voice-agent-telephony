# R4 – WebSocket Heartbeats + Reconnect

## Client (`frontend/ws.js` – ResilientWS)
- Heartbeat: `{"type":"ping"}` every **5s**
- Liveness: any inbound message refreshes `lastRecv`
- Dead if no recv for **15s** → force close → reconnect
- Backoff: 500ms × 1.7, capped at 8s, with jitter
- On open: `client_hello`; ignores pure `pong` for app logic

## Server
- Answers `ping` → `pong`
- Answers `client_hello` → `server_hello`

## UI
- Status: Connecting / Connected / Reconnecting / Connection dead / Disconnected
- Event log notes reconnect attempts

## Notes
- Mic MediaStream is local; after a long drop the user may need to click Start Mic again if the browser suspended capture
- Turn epochs (R1) still protect against late audio after reconnect mid-turn
