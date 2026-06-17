"""PipeWire/GStreamer frame capture backend."""

from __future__ import annotations

import importlib
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxter.capture.frames import FrameCaptureError, FrameCaptureRecord, parse_geometry
from voxter.capture.portal import PortalScreenCastSession, open_portal_screencast
from voxter.contracts import ActionState, CaptureRecordError

GST_SAMPLE_TIMEOUT_NS = 250_000_000


@dataclass(frozen=True, slots=True)
class RgbFrame:
    """One RGB frame pulled from GStreamer."""

    width: int
    height: int
    data: bytes


@dataclass(frozen=True, slots=True)
class GrayFrame:
    """One grayscale frame pulled from GStreamer."""

    width: int
    height: int
    data: bytes


@dataclass(frozen=True, slots=True)
class EncodedFrame:
    """One encoded frame pulled from GStreamer."""

    data: bytes


@dataclass(frozen=True, slots=True)
class FrameWriteJob:
    """One frame payload waiting for persistence."""

    path: Path
    data: bytes


@dataclass(frozen=True, slots=True)
class PipeWireFramePayload:
    """One in-memory frame payload pulled from PipeWire."""

    data: bytes
    image_format: str
    frame_width: int
    frame_height: int
    source_width: int
    source_height: int
    capture_resized: bool
    capture_duration_s: float


def pipewire_pipeline_description(
    *,
    pipewire_fd: int,
    node_id: int,
    image_format: str = "ppm",
    jpeg_quality: int = 70,
    output_width: int | None = None,
    output_height: int | None = None,
) -> str:
    """Return the GStreamer pipeline for low-latency PipeWire frame pulls."""

    if pipewire_fd < 0:
        raise CaptureRecordError("pipewire_fd must be non-negative")
    if node_id <= 0:
        raise CaptureRecordError("node_id must be positive")
    if image_format not in {"ppm", "jpeg", "gray8"}:
        raise CaptureRecordError("pipewire image_format must be ppm, jpeg, or gray8")
    if not 0 <= jpeg_quality <= 100:
        raise CaptureRecordError("jpeg_quality must be between 0 and 100")
    if (output_width is None) != (output_height is None):
        raise CaptureRecordError("output_width and output_height must be set together")
    if output_width is not None and output_width <= 0:
        raise CaptureRecordError("output_width must be positive")
    if output_height is not None and output_height <= 0:
        raise CaptureRecordError("output_height must be positive")

    prefix = (
        f"pipewiresrc fd={pipewire_fd} path={node_id} do-timestamp=true ! videoconvert "
    )
    caps_format = "GRAY8" if image_format == "gray8" else "RGB"
    caps = f"! video/x-raw,format={caps_format} "
    if output_width is not None and output_height is not None:
        caps = (
            "! videoscale "
            f"! video/x-raw,format={caps_format},"
            f"width={output_width},height={output_height} "
        )
    sink = "! appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
    if image_format in {"ppm", "gray8"}:
        return prefix + caps + sink
    return prefix + caps + f"! jpegenc quality={jpeg_quality} " + sink


def write_rgb_ppm(path: Path, frame: RgbFrame) -> None:
    """Write an RGB frame as binary PPM without compression."""

    write_frame_bytes(path, encode_rgb_ppm(frame))


def write_gray_pgm(path: Path, frame: GrayFrame) -> None:
    """Write a grayscale frame as binary PGM without compression."""

    write_frame_bytes(path, encode_gray_pgm(frame))


def encode_rgb_ppm(frame: RgbFrame) -> bytes:
    """Return a binary PPM payload for an RGB frame."""

    if frame.width <= 0 or frame.height <= 0:
        raise CaptureRecordError("RGB frame dimensions must be positive")
    expected_size = frame.width * frame.height * 3
    if len(frame.data) < expected_size:
        raise CaptureRecordError("RGB frame buffer is smaller than width*height*3")

    header = f"P6\n{frame.width} {frame.height}\n255\n".encode("ascii")
    return header + frame.data[:expected_size]


def encode_gray_pgm(frame: GrayFrame) -> bytes:
    """Return a binary PGM payload for a grayscale frame."""

    if frame.width <= 0 or frame.height <= 0:
        raise CaptureRecordError("grayscale frame dimensions must be positive")
    expected_size = frame.width * frame.height
    if len(frame.data) < expected_size:
        raise CaptureRecordError("grayscale frame buffer is smaller than width*height")

    header = f"P5\n{frame.width} {frame.height}\n255\n".encode("ascii")
    return header + frame.data[:expected_size]


def write_encoded_frame(path: Path, frame: EncodedFrame) -> None:
    """Write an already encoded image frame."""

    if not frame.data:
        raise CaptureRecordError("encoded frame buffer must be non-empty")
    write_frame_bytes(path, frame.data)


def write_frame_bytes(path: Path, data: bytes) -> None:
    """Write one already encoded frame payload."""

    if not data:
        raise CaptureRecordError("frame payload must be non-empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


class AsyncFrameWriter:
    """Bounded background writer for frame payloads."""

    def __init__(self, *, max_queue_size: int = 8) -> None:
        if max_queue_size <= 0:
            raise CaptureRecordError("max_queue_size must be positive")
        self._queue: queue.Queue[FrameWriteJob | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="voxter-frame-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, path: Path, data: bytes) -> None:
        """Queue one frame write without blocking the capture cadence."""

        if self._closed:
            raise FrameCaptureError("frame writer is closed")
        self._raise_error_if_present()
        try:
            self._queue.put_nowait(FrameWriteJob(path=path, data=data))
        except queue.Full as exc:
            raise FrameCaptureError("frame writer queue is full") from exc

    def close(self) -> None:
        """Flush queued writes and stop the writer."""

        if self._closed:
            self._raise_error_if_present()
            return
        self._closed = True
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._raise_error_if_present()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                write_frame_bytes(job.path, job.data)
            except BaseException as exc:  # noqa: BLE001
                if self._error is None:
                    self._error = exc
            finally:
                self._queue.task_done()

    def _raise_error_if_present(self) -> None:
        if self._error is not None:
            raise FrameCaptureError(f"frame writer failed: {self._error}") from (
                self._error
            )


class PipeWireGStreamerFrameCapture:
    """Real-time frame capture through xdg portal, PipeWire, and GStreamer."""

    def __init__(
        self,
        geometry: str,
        *,
        portal_session: PortalScreenCastSession | None = None,
        source_types: int = 1,
        cursor_mode: int = 1,
        portal_request_timeout_s: int = 20,
        image_format: str = "ppm",
        jpeg_quality: int = 70,
        async_writes: bool = True,
        write_queue_size: int = 8,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> None:
        self._source_geometry = parse_geometry(geometry)
        self.geometry = str(self._source_geometry)
        if image_format not in {"ppm", "jpeg", "gray8"}:
            raise CaptureRecordError(
                "pipewire image_format must be ppm, jpeg, or gray8"
            )
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.output_width = output_width
        self.output_height = output_height
        self._async_writes = async_writes
        self._write_queue_size = write_queue_size
        self._writer: AsyncFrameWriter | None = None
        self._portal_session = portal_session or open_portal_screencast(
            source_types=source_types,
            cursor_mode=cursor_mode,
            request_timeout_s=portal_request_timeout_s,
        )
        self._node_id = self._portal_session.streams[0].node_id
        self._fd = self._portal_session.pipewire_fd
        self._Gst = _load_gst()
        self._pipeline = self._Gst.parse_launch(
            pipewire_pipeline_description(
                pipewire_fd=self._fd,
                node_id=self._node_id,
                image_format=self.image_format,
                jpeg_quality=self.jpeg_quality,
                output_width=self.output_width,
                output_height=self.output_height,
            )
        )
        self._sink = self._pipeline.get_by_name("sink")
        if self._sink is None:
            raise FrameCaptureError("GStreamer pipeline did not create appsink")
        self._set_pipeline_playing()
        if self._async_writes:
            self._writer = AsyncFrameWriter(max_queue_size=self._write_queue_size)

    @property
    def file_suffix(self) -> str:
        """Return the file suffix used by this backend."""

        if self.image_format == "ppm":
            return ".ppm"
        if self.image_format == "gray8":
            return ".pgm"
        return ".jpg"

    @property
    def backend_name(self) -> str:
        """Return a stable backend identifier for logs."""

        return f"pipewire-gstreamer-{self.image_format}"

    def close(self) -> None:
        """Stop the GStreamer pipeline and close the portal PipeWire fd."""

        try:
            self._pipeline.set_state(self._Gst.State.NULL)
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
        finally:
            if self._writer is not None:
                self._writer.close()

    def restart_pipeline(self) -> None:
        """Restart the GStreamer pipeline while keeping the portal stream open."""

        self._pipeline.set_state(self._Gst.State.NULL)
        self._set_pipeline_playing()

    def capture(
        self,
        frame_path: Path,
        *,
        run_id: str,
        attempt_id: str | None,
        frame_index: int,
        action: ActionState,
        action_sample_timestamp: float,
    ) -> FrameCaptureRecord:
        """Pull one PipeWire frame, write it, and return its row."""

        payload = self.pull_frame_payload()
        frame_data = (
            encode_rgb_ppm(
                RgbFrame(
                    width=payload.frame_width,
                    height=payload.frame_height,
                    data=payload.data,
                )
            )
            if payload.image_format == "rgb"
            else payload.data
        )
        if payload.image_format == "gray8":
            frame_data = encode_gray_pgm(
                GrayFrame(
                    width=payload.frame_width,
                    height=payload.frame_height,
                    data=payload.data,
                )
            )
        if self._writer is None:
            write_frame_bytes(frame_path, frame_data)
        else:
            self._writer.submit(frame_path, frame_data)
        frame_timestamp = time.time()

        return FrameCaptureRecord(
            run_id=run_id,
            attempt_id=attempt_id,
            frame_index=frame_index,
            timestamp=frame_timestamp,
            frame_path=str(frame_path),
            action=action,
            action_sample_timestamp=action_sample_timestamp,
            geometry=self.geometry,
            capture_duration_s=payload.capture_duration_s,
            capture_backend=self.backend_name,
            image_format="pgm" if self.image_format == "gray8" else self.image_format,
            frame_width=payload.frame_width,
            frame_height=payload.frame_height,
            source_width=payload.source_width,
            source_height=payload.source_height,
            capture_resized=payload.capture_resized,
        )

    def pull_frame_payload(self) -> PipeWireFramePayload:
        """Pull one frame payload without writing it to disk.

        JPEG mode returns an encoded JPEG payload. PPM mode returns raw RGB bytes
        because PPM encoding is only needed for persistence.
        """

        started_at = time.monotonic()
        if self.image_format == "ppm":
            rgb_frame = self._pull_rgb_frame()
            finished_at = time.monotonic()
            return PipeWireFramePayload(
                data=rgb_frame.data,
                image_format="rgb",
                frame_width=rgb_frame.width,
                frame_height=rgb_frame.height,
                source_width=self._source_geometry.width,
                source_height=self._source_geometry.height,
                capture_resized=self.output_width is not None,
                capture_duration_s=finished_at - started_at,
            )
        if self.image_format == "gray8":
            gray_frame = self._pull_gray_frame()
            finished_at = time.monotonic()
            return PipeWireFramePayload(
                data=gray_frame.data,
                image_format="gray8",
                frame_width=gray_frame.width,
                frame_height=gray_frame.height,
                source_width=self._source_geometry.width,
                source_height=self._source_geometry.height,
                capture_resized=self.output_width is not None,
                capture_duration_s=finished_at - started_at,
            )
        encoded_frame = self._pull_encoded_frame()
        finished_at = time.monotonic()
        return PipeWireFramePayload(
            data=encoded_frame.data,
            image_format="jpeg",
            frame_width=self.output_width or self._source_geometry.width,
            frame_height=self.output_height or self._source_geometry.height,
            source_width=self._source_geometry.width,
            source_height=self._source_geometry.height,
            capture_resized=self.output_width is not None,
            capture_duration_s=finished_at - started_at,
        )

    def _set_pipeline_playing(self) -> None:
        result = self._pipeline.set_state(self._Gst.State.PLAYING)
        if result == self._Gst.StateChangeReturn.FAILURE:
            raise FrameCaptureError("failed to start GStreamer PipeWire pipeline")
        self._raise_bus_error_if_present()

    def _pull_rgb_frame(self) -> RgbFrame:
        sample = self._pull_sample()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        return RgbFrame(
            width=width,
            height=height,
            data=self._sample_bytes(sample, empty_error="RGB frame buffer is empty"),
        )

    def _pull_gray_frame(self) -> GrayFrame:
        sample = self._pull_sample()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))
        return GrayFrame(
            width=width,
            height=height,
            data=self._sample_bytes(
                sample,
                empty_error="grayscale frame buffer is empty",
            ),
        )

    def _pull_encoded_frame(self) -> EncodedFrame:
        return EncodedFrame(
            data=self._sample_bytes(
                self._pull_sample(),
                empty_error="encoded frame buffer is empty",
            )
        )

    def _pull_sample(self) -> Any:
        sample = self._sink.emit("try-pull-sample", GST_SAMPLE_TIMEOUT_NS)
        if sample is None:
            self._raise_bus_error_if_present()
            raise FrameCaptureError("timed out waiting for PipeWire frame")
        return sample

    def _sample_bytes(self, sample: Any, *, empty_error: str) -> bytes:
        buffer = sample.get_buffer()
        success, map_info = buffer.map(self._Gst.MapFlags.READ)
        if not success:
            raise FrameCaptureError("failed to map GStreamer frame buffer")
        try:
            data = bytes(map_info.data)
            if not data:
                raise FrameCaptureError(empty_error)
            return data
        finally:
            buffer.unmap(map_info)

    def _raise_bus_error_if_present(self) -> None:
        bus = self._pipeline.get_bus()
        message = bus.pop_filtered(
            self._Gst.MessageType.ERROR | self._Gst.MessageType.WARNING
        )
        if message is None:
            return
        if message.type == self._Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            raise FrameCaptureError(f"GStreamer error: {error}; debug={debug}")
        warning, debug = message.parse_warning()
        raise FrameCaptureError(f"GStreamer warning: {warning}; debug={debug}")


def _load_gst() -> Any:
    try:
        gi = importlib.import_module("gi")
    except ModuleNotFoundError as exc:
        raise FrameCaptureError(
            "PipeWire capture requires PyGObject. Use system Python or a venv "
            "created with --system-site-packages."
        ) from exc
    gi.require_version("Gst", "1.0")
    Gst = importlib.import_module("gi.repository.Gst")
    Gst.init(None)
    return Gst
