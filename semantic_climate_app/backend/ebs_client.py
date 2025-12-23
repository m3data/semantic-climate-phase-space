"""
EarthianBioSense WebSocket Client

Copyright (c) 2025 Mathew Mark Mytka
SPDX-License-Identifier: LicenseRef-ESL-A

Licensed under the Earthian Stewardship License (ESL-A).
See LICENSE file for full terms.

Connects to EBS WebSocket server for real-time biosignal streaming.
Implements ws://localhost:8765/stream protocol from websocket-api-v0.1.md
"""

import asyncio
import json
from datetime import datetime
from typing import Callable, Optional
from dataclasses import dataclass, field


@dataclass
class PhaseData:
    """Biosignal phase dynamics from EBS."""
    ts: str
    hr: int
    position: list  # [coherence, breath_norm, amp_norm]
    velocity: list
    velocity_mag: float
    curvature: float
    stability: float
    coherence: float
    phase_label: str

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "hr": self.hr,
            "position": self.position,
            "velocity": self.velocity,
            "velocity_mag": self.velocity_mag,
            "curvature": self.curvature,
            "stability": self.stability,
            "coherence": self.coherence,
            "phase_label": self.phase_label
        }


@dataclass
class EBSConnectionState:
    """Connection state for EBS WebSocket."""
    connected: bool = False
    session_id: Optional[str] = None
    device: Optional[str] = None
    device_connected: bool = False
    device_battery: Optional[int] = None


class EBSClient:
    """
    WebSocket client for EarthianBioSense biosignal stream.

    Usage:
        client = EBSClient()
        await client.connect()
        asyncio.create_task(client.start_listening())
        # Register callbacks
        client.subscribe(on_phase_callback)
        # Later...
        await client.send_semiotic_marker(metrics)
    """

    def __init__(self, url: str = "ws://localhost:8765/stream"):
        self.url = url
        self.ws = None
        self.state = EBSConnectionState()
        self._reconnect_delay = 2.0
        self._should_reconnect = True
        self._listen_task: Optional[asyncio.Task] = None
        self._phase_subscribers: list = []  # Multiple callbacks
        self._device_status_subscribers: list = []
        self._listening = False
        self.latest_phase: Optional[PhaseData] = None  # Cache latest for new subscribers

    async def connect(self) -> dict:
        """
        Connect to EBS and perform handshake.

        Returns:
            Welcome message dict from EBS
        """
        try:
            import websockets
        except ImportError:
            print("⚠️  websockets package not installed. Install with: pip install websockets")
            return {"error": "websockets not installed"}

        try:
            self.ws = await websockets.connect(self.url)

            # Send hello
            hello = {
                "type": "hello",
                "client": "semantic-climate",
                "version": "0.1",
                "session_id": self.state.session_id  # Reuse if reconnecting
            }
            await self.ws.send(json.dumps(hello))

            # Receive welcome
            welcome_raw = await self.ws.recv()
            welcome = json.loads(welcome_raw)

            if welcome.get("type") == "welcome":
                self.state.connected = True
                self.state.session_id = welcome.get("session_id")
                self.state.device = welcome.get("device")
                self.state.device_connected = welcome.get("status") == "streaming"

                print(f"🔗 Connected to EBS: {welcome.get('server')}")
                print(f"   Session: {self.state.session_id}")
                print(f"   Device: {self.state.device or 'waiting for device'}")

            return welcome

        except Exception as e:
            print(f"⚠️  Failed to connect to EBS at {self.url}: {e}")
            self.state.connected = False
            return {"error": str(e)}

    def subscribe(self, on_phase: Callable[[PhaseData], None],
                  on_device_status: Optional[Callable[[dict], None]] = None) -> int:
        """
        Subscribe to phase updates.

        Args:
            on_phase: Callback for phase updates (1Hz)
            on_device_status: Optional callback for device status changes

        Returns:
            Subscriber ID for unsubscribing
        """
        sub_id = len(self._phase_subscribers)
        self._phase_subscribers.append(on_phase)
        if on_device_status:
            self._device_status_subscribers.append(on_device_status)
        return sub_id

    def unsubscribe(self, sub_id: int):
        """Remove a subscriber (set to None to preserve indices)."""
        if 0 <= sub_id < len(self._phase_subscribers):
            self._phase_subscribers[sub_id] = None

    async def start_listening(self):
        """
        Start the background listen task (call once on startup).
        """
        if self._listening:
            return  # Already listening
        self._listening = True
        await self._listen_loop()

    async def _listen_loop(self):
        """Internal listen loop that broadcasts to all subscribers."""
        while self._should_reconnect:
            if not self.ws or not self.state.connected:
                await self._try_reconnect()
                continue

            try:
                async for message in self.ws:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "phase":
                        # EBS sends "entrainment" but we map to "coherence" for compatibility
                        phase = PhaseData(
                            ts=data.get("ts"),
                            hr=data.get("hr", 0),
                            position=data.get("position", [0, 0, 0]),
                            velocity=data.get("velocity", [0, 0, 0]),
                            velocity_mag=data.get("velocity_mag", 0),
                            curvature=data.get("curvature", 0),
                            stability=data.get("stability", 0),
                            coherence=data.get("entrainment", 0),  # Map entrainment → coherence
                            phase_label=data.get("phase_label", "")
                        )
                        self.latest_phase = phase
                        # Broadcast to all subscribers
                        for callback in self._phase_subscribers:
                            if callback:
                                try:
                                    # Handle both sync and async callbacks
                                    result = callback(phase)
                                    if asyncio.iscoroutine(result):
                                        await result
                                except Exception as e:
                                    print(f"⚠️  Phase callback error: {e}")

                    elif msg_type == "device_status":
                        self.state.device_connected = data.get("connected", False)
                        self.state.device = data.get("device")
                        self.state.device_battery = data.get("battery")
                        for callback in self._device_status_subscribers:
                            if callback:
                                try:
                                    callback(data)
                                except Exception:
                                    pass

                    elif msg_type == "session_end":
                        print(f"📊 EBS session ended: {data.get('duration_sec')}s, {data.get('samples')} samples")
                        self.state.connected = False

                    elif msg_type == "error":
                        print(f"⚠️  EBS error: {data.get('message')}")

            except Exception as e:
                print(f"⚠️  EBS connection lost: {e}")
                self.state.connected = False
                await self._try_reconnect()

    # Keep old listen() for backwards compatibility
    async def listen(self, on_phase: Callable[[PhaseData], None],
                     on_device_status: Optional[Callable[[dict], None]] = None):
        """Legacy listen method - wraps subscribe + start_listening."""
        self.subscribe(on_phase, on_device_status)
        await self.start_listening()

    async def _try_reconnect(self):
        """Attempt to reconnect after delay."""
        if not self._should_reconnect:
            return

        print(f"🔄 Reconnecting to EBS in {self._reconnect_delay}s...")
        await asyncio.sleep(self._reconnect_delay)
        await self.connect()

    async def send_semiotic_marker(self, metrics: dict):
        """
        Send semantic metrics to EBS for JSONL logging.

        Args:
            metrics: Dict with delta_kappa, delta_h, psi_composite, attractor_basin, etc.
        """
        if not self.ws or not self.state.connected:
            return  # Silently skip if not connected

        marker = {
            "type": "semiotic_marker",
            "ts": datetime.now().isoformat(),
            "curvature_delta": metrics.get("metrics", {}).get("delta_kappa", 0),
            "entropy_delta": metrics.get("metrics", {}).get("delta_h", 0),
            "coupling_psi": metrics.get("psi_composite", 0),
            "coupling_mode": metrics.get("coupling_mode", {}).get("compound_label", "unknown"),
            "coherence_pattern": metrics.get("coupling_mode", {}).get("coherence", {}).get("pattern", "unknown"),
            "label": f"basin_{metrics.get('attractor_basin', {}).get('name', 'unknown')}"
        }

        try:
            await self.ws.send(json.dumps(marker))
        except Exception as e:
            print(f"⚠️  Failed to send semiotic marker: {e}")

    async def send_field_event(self, event: str, note: str = ""):
        """
        Send manual field event marker to EBS.

        Args:
            event: Event type (e.g., "breath_shift", "insight", "felt_sense")
            note: Optional annotation
        """
        if not self.ws or not self.state.connected:
            return

        field_event = {
            "type": "field_event",
            "ts": datetime.now().isoformat(),
            "event": event,
            "note": note
        }

        try:
            await self.ws.send(json.dumps(field_event))
        except Exception as e:
            print(f"⚠️  Failed to send field event: {e}")

    async def disconnect(self):
        """Cleanly disconnect from EBS."""
        self._should_reconnect = False
        if self.ws:
            await self.ws.close()
        self.state.connected = False
        print("🔌 Disconnected from EBS")

    def is_connected(self) -> bool:
        """Check if currently connected to EBS."""
        return self.state.connected and self.ws is not None

    def get_state(self) -> dict:
        """Get current connection state for frontend."""
        return {
            "connected": self.state.connected,
            "session_id": self.state.session_id,
            "device": self.state.device,
            "device_connected": self.state.device_connected,
            "device_battery": self.state.device_battery
        }


# Singleton instance for app-wide use
_ebs_client: Optional[EBSClient] = None


def get_ebs_client() -> EBSClient:
    """Get or create the global EBS client instance."""
    global _ebs_client
    if _ebs_client is None:
        _ebs_client = EBSClient()
    return _ebs_client
