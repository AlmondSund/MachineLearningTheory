"""xdg-desktop-portal ScreenCast helpers for PipeWire capture."""

from __future__ import annotations

import importlib
import os
import secrets
from dataclasses import dataclass
from typing import Any

from voxter.contracts import CaptureRecordError

PORTAL_BUS_NAME = "org.freedesktop.portal.Desktop"
PORTAL_OBJECT_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_INTERFACE = "org.freedesktop.portal.ScreenCast"
REQUEST_INTERFACE = "org.freedesktop.portal.Request"


@dataclass(frozen=True, slots=True)
class PortalScreenCastStream:
    """One PipeWire stream returned by the ScreenCast portal."""

    node_id: int
    properties: dict[str, object]


@dataclass(frozen=True, slots=True)
class PortalScreenCastSession:
    """Active ScreenCast session and PipeWire remote descriptor."""

    session_handle: str
    pipewire_fd: int
    streams: tuple[PortalScreenCastStream, ...]


def portal_token(prefix: str) -> str:
    """Return an xdg-desktop-portal handle token."""

    return f"{prefix}_{secrets.token_hex(8)}"


def parse_portal_streams(streams_value: object) -> tuple[PortalScreenCastStream, ...]:
    """Parse the portal `streams` result into stable typed records."""

    streams: list[PortalScreenCastStream] = []
    if not isinstance(streams_value, (list, tuple)):
        raise CaptureRecordError("portal streams result must be a sequence")

    for item in streams_value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise CaptureRecordError("portal stream item must be (node_id, properties)")
        node_id, properties = item
        if not isinstance(node_id, int) or node_id <= 0:
            raise CaptureRecordError("portal stream node_id must be a positive integer")
        if not isinstance(properties, dict):
            raise CaptureRecordError("portal stream properties must be a dictionary")
        streams.append(
            PortalScreenCastStream(
                node_id=node_id,
                properties={
                    str(key): _variant_to_python(value)
                    for key, value in properties.items()
                },
            )
        )

    if not streams:
        raise CaptureRecordError("portal did not return any PipeWire streams")
    return tuple(streams)


def open_portal_screencast(
    *,
    source_types: int = 1,
    cursor_mode: int = 1,
    restore_token: str | None = None,
    request_timeout_s: int = 20,
) -> PortalScreenCastSession:
    """Open an interactive portal screencast session.

    `source_types=1` selects monitor sources. The portal may still present a
    compositor-controlled chooser. This function requires a graphical session
    and should be called only for explicit capture commands.
    """

    Gio, GLib = _load_gio_glib()
    try:
        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    except Exception as exc:
        raise CaptureRecordError(
            f"failed to connect to xdg portal session bus: {exc}"
        ) from exc
    portal = Gio.DBusProxy.new_sync(
        bus,
        Gio.DBusProxyFlags.NONE,
        None,
        PORTAL_BUS_NAME,
        PORTAL_OBJECT_PATH,
        SCREENCAST_INTERFACE,
        None,
    )

    session_handle = _create_session(
        portal,
        bus,
        Gio,
        GLib,
        restore_token,
        request_timeout_s=request_timeout_s,
    )
    _select_sources(
        portal,
        bus,
        Gio,
        GLib,
        session_handle=session_handle,
        source_types=source_types,
        cursor_mode=cursor_mode,
        request_timeout_s=request_timeout_s,
    )
    streams = _start_session(
        portal,
        bus,
        Gio,
        GLib,
        session_handle=session_handle,
        request_timeout_s=request_timeout_s,
    )
    pipewire_fd = _open_pipewire_remote(portal, Gio, GLib, session_handle)
    return PortalScreenCastSession(
        session_handle=session_handle,
        pipewire_fd=pipewire_fd,
        streams=streams,
    )


def _create_session(
    portal: Any,
    bus: Any,
    Gio: Any,
    GLib: Any,
    restore_token: str | None,
    *,
    request_timeout_s: int,
) -> str:
    handle_token = portal_token("create")
    options: dict[str, Any] = {
        "handle_token": GLib.Variant("s", handle_token),
        "session_handle_token": GLib.Variant("s", portal_token("session")),
    }
    if restore_token:
        options["restore_token"] = GLib.Variant("s", restore_token)

    request_handle = _expected_request_handle(bus.get_unique_name(), handle_token)
    results = _call_portal_request_and_wait(
        portal,
        bus,
        Gio,
        GLib,
        "CreateSession",
        GLib.Variant("(a{sv})", (options,)),
        request_handle,
        request_timeout_s,
    )
    session_handle = _variant_to_python(results.get("session_handle"))
    if not isinstance(session_handle, str) or not session_handle:
        raise CaptureRecordError("portal CreateSession did not return session_handle")
    return session_handle


def _select_sources(
    portal: Any,
    bus: Any,
    Gio: Any,
    GLib: Any,
    *,
    session_handle: str,
    source_types: int,
    cursor_mode: int,
    request_timeout_s: int,
) -> None:
    handle_token = portal_token("select")
    options = {
        "handle_token": GLib.Variant("s", handle_token),
        "types": GLib.Variant("u", source_types),
        "multiple": GLib.Variant("b", False),
        "cursor_mode": GLib.Variant("u", cursor_mode),
    }
    request_handle = _expected_request_handle(bus.get_unique_name(), handle_token)
    _call_portal_request_and_wait(
        portal,
        bus,
        Gio,
        GLib,
        "SelectSources",
        GLib.Variant("(oa{sv})", (session_handle, options)),
        request_handle,
        request_timeout_s,
    )


def _start_session(
    portal: Any,
    bus: Any,
    Gio: Any,
    GLib: Any,
    *,
    session_handle: str,
    request_timeout_s: int,
) -> tuple[PortalScreenCastStream, ...]:
    handle_token = portal_token("start")
    options = {"handle_token": GLib.Variant("s", handle_token)}
    request_handle = _expected_request_handle(bus.get_unique_name(), handle_token)
    results = _call_portal_request_and_wait(
        portal,
        bus,
        Gio,
        GLib,
        "Start",
        GLib.Variant("(osa{sv})", (session_handle, "", options)),
        request_handle,
        request_timeout_s,
    )
    streams_value = _variant_to_python(results.get("streams"))
    return parse_portal_streams(streams_value)


def _open_pipewire_remote(
    portal: Any,
    Gio: Any,
    GLib: Any,
    session_handle: str,
) -> int:
    result, fd_list = portal.call_with_unix_fd_list_sync(
        "OpenPipeWireRemote",
        GLib.Variant("(oa{sv})", (session_handle, {})),
        Gio.DBusCallFlags.NONE,
        -1,
        None,
        None,
    )
    fd_index = result.unpack()[0]
    fd = os.dup(int(fd_list.get(fd_index)))
    if fd < 0:
        raise CaptureRecordError("portal returned invalid PipeWire fd")
    return fd


def _call_portal_request_and_wait(
    portal: Any,
    bus: Any,
    Gio: Any,
    GLib: Any,
    method: str,
    parameters: Any,
    request_handle: str,
    request_timeout_s: int,
) -> dict[str, Any]:
    waiter = _PortalRequestWaiter(
        bus,
        Gio,
        GLib,
        request_handle,
        request_timeout_s,
    )
    result = portal.call_sync(method, parameters, 0, -1, None)
    returned_handle = result.unpack()[0]
    if not isinstance(returned_handle, str) or not returned_handle:
        raise CaptureRecordError(f"portal {method} did not return request handle")
    if returned_handle != request_handle:
        waiter.close()
        return _wait_for_request(returned_handle, Gio, GLib, request_timeout_s)
    return waiter.wait()


class _PortalRequestWaiter:
    def __init__(
        self,
        bus: Any,
        Gio: Any,
        GLib: Any,
        request_handle: str,
        request_timeout_s: int,
    ) -> None:
        self._bus = bus
        self._Gio = Gio
        self._GLib = GLib
        self._request_handle = request_handle
        self._request_timeout_s = request_timeout_s
        self._loop = GLib.MainLoop()
        self._response: dict[str, Any] = {}
        self._timed_out = False
        self._subscription_id = bus.signal_subscribe(
            PORTAL_BUS_NAME,
            REQUEST_INTERFACE,
            "Response",
            request_handle,
            None,
            Gio.DBusSignalFlags.NONE,
            self._on_signal,
        )
        self._timeout_id = GLib.timeout_add_seconds(
            request_timeout_s,
            self._on_timeout,
        )

    def wait(self) -> dict[str, Any]:
        self._loop.run()
        self.close()

        if self._timed_out:
            raise CaptureRecordError(
                f"portal request timed out after "
                f"{self._request_timeout_s}s: {self._request_handle}"
            )
        if self._response.get("code") != 0:
            raise CaptureRecordError(
                f"portal request was cancelled or failed: {self._response}"
            )
        results = self._response.get("results")
        if not isinstance(results, dict):
            raise CaptureRecordError("portal response results must be a dictionary")
        return results

    def close(self) -> None:
        if self._subscription_id:
            self._bus.signal_unsubscribe(self._subscription_id)
            self._subscription_id = 0
        if self._timeout_id:
            self._GLib.source_remove(self._timeout_id)
            self._timeout_id = 0

    def _on_timeout(self) -> bool:
        self._timed_out = True
        self._loop.quit()
        return False

    def _on_signal(
        self,
        _connection: Any,
        _sender_name: str,
        _object_path: str,
        _interface_name: str,
        _signal_name: str,
        parameters: Any,
    ) -> None:
        response_code, results = parameters.unpack()
        self._response["code"] = response_code
        self._response["results"] = results
        self._loop.quit()


def _wait_for_request(
    request_handle: str,
    Gio: Any,
    GLib: Any,
    request_timeout_s: int,
) -> dict[str, Any]:
    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    request = Gio.DBusProxy.new_sync(
        bus,
        Gio.DBusProxyFlags.NONE,
        None,
        PORTAL_BUS_NAME,
        request_handle,
        REQUEST_INTERFACE,
        None,
    )
    loop = GLib.MainLoop()
    response: dict[str, Any] = {}
    timed_out = {"value": False}

    def on_timeout() -> bool:
        timed_out["value"] = True
        loop.quit()
        return False

    def on_signal(
        _proxy: Any,
        _sender: str,
        signal_name: str,
        parameters: Any,
    ) -> None:
        if signal_name != "Response":
            return
        response_code, results = parameters.unpack()
        response["code"] = response_code
        response["results"] = results
        loop.quit()

    request.connect("g-signal", on_signal)
    GLib.timeout_add_seconds(request_timeout_s, on_timeout)
    loop.run()

    if timed_out["value"]:
        raise CaptureRecordError(
            f"portal request timed out after {request_timeout_s}s: {request_handle}"
        )

    if response.get("code") != 0:
        raise CaptureRecordError(f"portal request was cancelled or failed: {response}")
    results = response.get("results")
    if not isinstance(results, dict):
        raise CaptureRecordError("portal response results must be a dictionary")
    return results


def _expected_request_handle(unique_name: str, handle_token: str) -> str:
    sender = unique_name.removeprefix(":").replace(".", "_")
    return f"{PORTAL_OBJECT_PATH}/request/{sender}/{handle_token}"


def _load_gio_glib() -> tuple[Any, Any]:
    try:
        gi = importlib.import_module("gi")
    except ModuleNotFoundError as exc:
        raise CaptureRecordError(
            "Portal screencast requires PyGObject. Use system Python or a "
            "virtual environment created with --system-site-packages."
        ) from exc
    gi.require_version("Gio", "2.0")
    return (
        importlib.import_module("gi.repository.Gio"),
        importlib.import_module("gi.repository.GLib"),
    )


def _variant_to_python(value: object) -> object:
    if hasattr(value, "unpack"):
        return value.unpack()
    return value
