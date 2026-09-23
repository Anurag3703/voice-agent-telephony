/**
 * R4 – Resilient WebSocket with heartbeats + auto-reconnect
 *
 * - Sends {"type":"ping"} every heartbeatMs
 * - If no message (including pong) within deadMs → force close + reconnect
 * - Exponential backoff reconnect with jitter
 * - Emits lifecycle callbacks for UI
 */

export class ResilientWS {
  /**
   * @param {object} opts
   * @param {() => string} opts.urlFn
   * @param {(data:any) => void} opts.onMessage
   * @param {(status:string, detail?:string) => void} [opts.onStatus]
   * @param {number} [opts.heartbeatMs]
   * @param {number} [opts.deadMs]
   * @param {number} [opts.maxBackoffMs]
   */
  constructor(opts) {
    this.urlFn = opts.urlFn;
    this.onMessage = opts.onMessage;
    this.onStatus = opts.onStatus || (() => {});
    this.heartbeatMs = opts.heartbeatMs ?? 5000;
    this.deadMs = opts.deadMs ?? 15000;
    this.maxBackoffMs = opts.maxBackoffMs ?? 10000;

    this.ws = null;
    this.wanted = false;
    this.hbTimer = null;
    this.watchTimer = null;
    this.reconnectTimer = null;
    this.lastRecv = 0;
    this.backoff = 500;
    this.reconnectAttempts = 0;
  }

  connect() {
    this.wanted = true;
    this._open();
  }

  close() {
    this.wanted = false;
    this._clearTimers();
    if (this.ws) {
      try { this.ws.close(); } catch (_) {}
      this.ws = null;
    }
    this.onStatus("disconnected");
  }

  get ready() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  send(data) {
    if (!this.ready) return false;
    try {
      this.ws.send(data);
      return true;
    } catch (_) {
      return false;
    }
  }

  sendJson(obj) {
    return this.send(JSON.stringify(obj));
  }

  _open() {
    this._clearTimers();
    if (this.ws) {
      try { this.ws.close(); } catch (_) {}
      this.ws = null;
    }

    const url = this.urlFn();
    this.onStatus(this.reconnectAttempts > 0 ? "reconnecting" : "connecting", url);

    let ws;
    try {
      ws = new WebSocket(url);
    } catch (err) {
      this.onStatus("error", String(err));
      this._scheduleReconnect();
      return;
    }
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      this.lastRecv = Date.now();
      this.backoff = 500;
      const wasReconnect = this.reconnectAttempts > 0;
      this.reconnectAttempts = 0;
      this.onStatus("connected", wasReconnect ? "resumed" : "fresh");
      this._startHeartbeat();
      // Announce client session for future resume hooks
      this.sendJson({ type: "client_hello", ts: Date.now() });
    };

    ws.onmessage = (ev) => {
      this.lastRecv = Date.now();
      if (typeof ev.data === "string") {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === "pong") return; // heartbeat only
          this.onMessage(data);
        } catch (e) {
          this.onMessage({ type: "error", message: "bad json: " + e.message });
        }
      }
    };

    ws.onerror = () => {
      this.onStatus("error", "socket error");
    };

    ws.onclose = () => {
      this._clearHeartbeat();
      this.ws = null;
      if (this.wanted) {
        this.onStatus("disconnected", "retrying");
        this._scheduleReconnect();
      } else {
        this.onStatus("disconnected");
      }
    };
  }

  _startHeartbeat() {
    this._clearHeartbeat();
    this.hbTimer = setInterval(() => {
      if (!this.ready) return;
      this.sendJson({ type: "ping", t: Date.now() });
    }, this.heartbeatMs);

    this.watchTimer = setInterval(() => {
      if (!this.wanted) return;
      const silence = Date.now() - this.lastRecv;
      if (silence > this.deadMs) {
        this.onStatus("dead", `no recv ${silence}ms`);
        try { this.ws && this.ws.close(); } catch (_) {}
        // onclose will schedule reconnect
      }
    }, Math.min(2000, this.heartbeatMs));
  }

  _clearHeartbeat() {
    if (this.hbTimer) clearInterval(this.hbTimer);
    if (this.watchTimer) clearInterval(this.watchTimer);
    this.hbTimer = null;
    this.watchTimer = null;
  }

  _clearTimers() {
    this._clearHeartbeat();
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
  }

  _scheduleReconnect() {
    if (!this.wanted) return;
    if (this.reconnectTimer) return;
    this.reconnectAttempts += 1;
    const jitter = Math.random() * 200;
    const delay = Math.min(this.maxBackoffMs, this.backoff) + jitter;
    this.onStatus("reconnecting", `attempt ${this.reconnectAttempts} in ${Math.round(delay)}ms`);
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.backoff = Math.min(this.maxBackoffMs, this.backoff * 1.7);
      this._open();
    }, delay);
  }
}
