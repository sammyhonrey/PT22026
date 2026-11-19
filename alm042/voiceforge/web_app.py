"""
Dependency-free web interface for VoiceForge.

The user interface is a separate static application in ``frontend/`` (HTML,
CSS and ES modules).  This module only serves those files and the JSON API:

* ``GET  /``                 the application shell (``frontend/index.html``)
* ``GET  /css/...``, ``/js/...``, ``/assets/...``  static assets
* ``GET  /health``           engine capabilities and limits
* ``GET  /audio/<file>``     a generated WAV (playback and download)
* ``POST /generate``         JSON ``{"text": ...}`` -> generated WAV metadata
"""

from __future__ import annotations

import base64
import binascii
import json
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from voiceforge.app_core import (
    MAX_TEXT_LENGTH,
    AppError,
    VoiceForgeApp,
)
from voiceforge.jobs import (
    ArchiveUnavailable,
    JobManager,
    JobNotFound,
    JobQueueFull,
)
from voiceforge.cache import cleanup_mp3_cache
from voiceforge.document_reader import extract_document
from voiceforge.library import (
    DEFAULT_PAGE_SIZE,
    Library,
)
from voiceforge.mp3 import (
    DEFAULT_BITRATE,
    SUPPORTED_BITRATES,
    cached_mp3,
    validate_bitrate,
)
from voiceforge.pronunciation import apply_dictionary
from voiceforge.uploads import validate_uploads

#: Static frontend root (``alm042/frontend``).
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

#: URL prefixes that may be served from :data:`FRONTEND_DIR`.
STATIC_PREFIXES = ("css/", "js/", "assets/")

#: Only these extensions are served, with an explicit content type.
STATIC_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}

#: Base64 inflates uploads by ~33 %: a 10 MB document needs ~13.4 MB, and
#: the 5 MB TXT total needs ~6.7 MB. 15 MB leaves headroom for the JSON
#: wrapper without allowing unbounded bodies.
MAX_BODY_BYTES = 15 * 1024 * 1024


class VoiceForgeServer(ThreadingHTTPServer):
    """HTTP server that carries the application and job manager."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        app: VoiceForgeApp,
        jobs: JobManager | None = None,
        library: Library | None = None,
    ):
        super().__init__(server_address, VoiceForgeHandler)
        self.app = app
        self.library = library if library is not None else Library()
        self.jobs = (
            jobs
            if jobs is not None
            else JobManager(app, library=self.library)
        )

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            # Stop the worker and fail anything still queued.
            self.jobs.stop()


class VoiceForgeHandler(BaseHTTPRequestHandler):
    """Handles the interface page, speech generation and audio files."""

    server_version = "VoiceForge/1.0"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def app(self) -> VoiceForgeApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")

        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)

        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The client closed the connection (tab closed, reload...).
            return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_error_json(
        self,
        status: int,
        message: str,
        code: str = "bad_request",
    ) -> None:
        """
        One consistent error shape for every API endpoint:

            {"ok": false, "error": "<human message>", "code": "<machine code>"}
        """

        self._send_json(status, {"ok": False, "error": message, "code": code})

    def _send_app_error(self, error: AppError) -> None:
        """Send an expected application error with its status and code."""

        self._send_error_json(error.status, error.message, error.code)

    def _log_internal_error(self) -> None:
        """Print the traceback on the server; never send it to the client."""

        print("[web] internal error:", flush=True)
        traceback.print_exc()

    def _send_internal_error(self) -> None:
        try:
            self._send_error_json(
                500,
                "The server could not complete the request. "
                "Please try again.",
                "internal_error",
            )
        except Exception:  # noqa: BLE001 - client already disconnected
            pass

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise AppError("Invalid request length.") from None

        if length <= 0:
            raise AppError("The request was empty.")

        if length > MAX_BODY_BYTES:
            raise AppError(
                "The request is too large.",
                code="request_too_large",
                status=413,
            )

        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise AppError("The request was not valid JSON.") from None

        if not isinstance(payload, dict):
            raise AppError("The request was not valid JSON.")

        return payload

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 (standard library naming)
        try:
            self._handle_get()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:  # noqa: BLE001 - never leak a traceback
            self._log_internal_error()
            self._send_internal_error()

    def _handle_get(self) -> None:
        # Unquote so that encoded traversal attempts are inspected as paths.
        path = unquote(self.path.split("?", 1)[0])

        if path in ("/", "/index.html"):
            self._send_file(
                FRONTEND_DIR / "index.html",
                "text/html; charset=utf-8",
            )
            return

        if path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "model_loaded": self.app.is_loaded,
                    "checkpoint": str(self.app.checkpoint_path),
                    "limits": {"max_text_length": MAX_TEXT_LENGTH},
                    "engines": self.app.describe_engines(),
                },
            )
            return

        if path == "/api/engines":
            self._send_json(
                200,
                {"ok": True, **self.app.describe_engine_details()},
            )
            return

        if path.startswith("/api/jobs/"):
            self._handle_job_get(path[len("/api/jobs/"):])
            return

        if path == "/api/library" or path.startswith("/api/library/"):
            self._handle_library_get(path)
            return

        if path == "/api/pronunciation" or path.startswith("/api/pronunciation/"):
            self._handle_pronunciation_get(path)
            return

        if path == "/api/presets" or path.startswith("/api/presets/"):
            self._handle_presets_get(path)
            return

        if path.startswith("/audio/"):
            self._send_audio(path[len("/audio/"):])
            return

        relative = path.lstrip("/")

        if relative.startswith(STATIC_PREFIXES):
            self._send_static(relative)
            return

        self._send_error_json(404, "Page not found.")

    def _send_file(self, path: Path, content_type: str) -> None:
        """Send one file from the frontend directory."""

        try:
            data = path.read_bytes()
        except OSError:
            self._send_error_json(404, "Page not found.")
            return

        self._send_bytes(200, data, content_type)

    def _send_static(self, relative: str) -> None:
        """
        Serve a CSS/JS/asset file, refusing anything outside ``frontend/``.

        Traversal is blocked three ways: explicit ``..`` segments, a resolved
        path that must stay inside the frontend root, and an allow-list of
        extensions (so a stray .py or .pt file can never be served).
        """

        parts = Path(relative).parts

        if ".." in parts or relative.startswith("/") or ":" in parts[0]:
            self._send_error_json(404, "Page not found.")
            return

        root = FRONTEND_DIR.resolve()
        candidate = (root / relative).resolve()

        if root not in candidate.parents or not candidate.is_file():
            self._send_error_json(404, "Page not found.")
            return

        content_type = STATIC_MIME_TYPES.get(candidate.suffix.lower())

        if content_type is None:
            self._send_error_json(404, "Page not found.")
            return

        self._send_file(candidate, content_type)

    def _send_audio(self, filename: str) -> None:
        try:
            path = self.app.resolve_audio(filename)
        except AppError as error:
            # Audio failures are always 404 so the endpoint never reveals
            # whether a file exists outside the output folder.
            self._send_error_json(404, error.message, error.code)
            return

        data = path.read_bytes()

        self._send_bytes(
            200,
            data,
            "audio/wav",
            {"Content-Disposition": f'inline; filename="{path.name}"'},
        )

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802 (standard library naming)
        try:
            self._handle_post()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:  # noqa: BLE001 - never leak a traceback
            self._log_internal_error()
            self._send_internal_error()

    def _handle_post(self) -> None:
        path = self.path.split("?", 1)[0]

        if path in ("/generate", "/api/generate"):
            self._handle_generate()
            return

        if path == "/api/jobs":
            self._handle_submit_job()
            return

        if path == "/api/uploads":
            self._handle_upload()
            return

        if path == "/api/documents/extract":
            self._handle_document_extract()
            return

        if path == "/api/pronunciation":
            self._handle_pronunciation_create()
            return

        if path == "/api/presets":
            self._handle_preset_create()
            return

        self._send_error_json(404, "Page not found.", "not_found")

    # ------------------------------------------------------------------
    # Jobs and uploads
    # ------------------------------------------------------------------
    def _handle_submit_job(self) -> None:
        try:
            payload = self._read_json_body()
            job = self.server.jobs.submit(  # type: ignore[attr-defined]
                payload
            )
        except AppError as error:
            self._send_app_error(error)
            return
        except JobQueueFull as error:
            self._send_error_json(429, str(error), "queue_full")
            return

        self._send_json(202, {"ok": True, "job": job.to_dict()})

    def _handle_job_get(self, remainder: str) -> None:
        parts = [part for part in remainder.split("/") if part]

        if not parts:
            self._send_error_json(404, "Job not found.", "job_not_found")
            return

        job_id = parts[0]
        jobs = self.server.jobs  # type: ignore[attr-defined]

        try:
            if len(parts) == 1:
                self._send_json(200, {"ok": True, "job": jobs.get(job_id).to_dict()})
                return

            if len(parts) == 2 and parts[1] == "download":
                query = parse_qs(urlparse(self.path).query)
                archive_format = (query.get("format") or ["wav"])[0].lower()
                raw_bitrate = (query.get("bitrate") or [str(DEFAULT_BITRATE)])[0]

                archive, download_name = jobs.archive(
                    job_id,
                    format=archive_format,
                    bitrate=raw_bitrate,
                )

                self._send_bytes(
                    200,
                    archive,
                    "application/zip",
                    {
                        "Content-Disposition": (
                            f'attachment; filename="{download_name}"'
                        )
                    },
                )
                return

            if len(parts) == 3 and parts[1] == "audio":
                try:
                    item_path = jobs.item_file(job_id, parts[2])
                except FileNotFoundError:
                    self._send_error_json(
                        404,
                        "Audio not found.",
                        "audio_not_found",
                    )
                    return

                query = parse_qs(urlparse(self.path).query)
                item_format = (query.get("format") or ["wav"])[0].lower()

                if item_format == "mp3":
                    bitrate = validate_bitrate(
                        (query.get("bitrate") or [str(DEFAULT_BITRATE)])[0]
                    )
                    mp3_path, _ = cached_mp3(
                        item_path,
                        self.server.jobs.job_root / job_id / "mp3",  # type: ignore[attr-defined]
                        bitrate,
                    )
                    self._send_bytes(
                        200,
                        mp3_path.read_bytes(),
                        "audio/mpeg",
                        {
                            "Content-Disposition": (
                                f'attachment; filename="{mp3_path.name}"'
                            )
                        },
                    )
                    return

                if item_format != "wav":
                    self._send_error_json(
                        400,
                        "Unsupported audio format. Choose wav or mp3.",
                        "invalid_format",
                    )
                    return

                self._send_bytes(
                    200,
                    item_path.read_bytes(),
                    "audio/wav",
                    {
                        "Content-Disposition": (
                            f'attachment; filename="{item_path.name}"'
                        )
                    },
                )
                return

        except JobNotFound:
            self._send_error_json(404, "Job not found.", "job_not_found")
            return
        except ArchiveUnavailable as error:
            self._send_error_json(409, str(error), "no_results")
            return

        self._send_error_json(404, "Job not found.", "job_not_found")

    def _handle_upload(self) -> None:
        try:
            payload = self._read_json_body()
            files = validate_uploads(payload)
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(
            200,
            {
                "ok": True,
                "files": files,
                "count": len(files),
            },
        )

    def _handle_generate(self) -> None:
        """
        Shared implementation for ``POST /generate`` and
        ``POST /api/generate``.
        """

        try:
            payload = self._read_json_body()
            text = payload.get("text")

            if payload.get("pronunciation") and isinstance(text, str):
                entries = self.server.library.list_pronunciation()  # type: ignore[attr-defined]
                text, replacements = apply_dictionary(text, entries)
                if replacements:
                    print(
                        f"[web] applied {replacements} pronunciation rule(s)",
                        flush=True,
                    )

            result = self.app.synthesize(
                text,
                engine=payload.get("engine") or None,
                voice=payload.get("voice") or None,
                speed=payload.get("speed"),
            )
        except AppError as error:
            self._send_app_error(error)
            return

        record = self._record_studio_audio(result)

        self._send_json(
            200,
            {
                "ok": True,
                "text": result.text,
                "filename": result.filename,
                "audio_url": f"/audio/{result.filename}",
                "duration_seconds": result.duration_seconds,
                "sample_rate": result.sample_rate,
                "engine": result.engine,
                "voice": result.voice,
                "speed": result.speed,
                "padded": result.padded,
                "truncated": result.truncated,
                "library_id": record["id"] if record else None,
                "mp3_url": (
                    f"/api/library/{record['id']}/mp3" if record else None
                ),
            },
        )

    def do_PATCH(self) -> None:  # noqa: N802 (standard library naming)
        try:
            self._handle_patch()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:  # noqa: BLE001 - never leak a traceback
            self._log_internal_error()
            self._send_internal_error()

    def do_DELETE(self) -> None:  # noqa: N802 (standard library naming)
        try:
            self._handle_delete()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception:  # noqa: BLE001 - never leak a traceback
            self._log_internal_error()
            self._send_internal_error()

    # ------------------------------------------------------------------
    # Audio library
    # ------------------------------------------------------------------
    def _record_studio_audio(self, result: Any) -> None:
        """Record a successful Speech Studio generation (best effort)."""

        try:
            return self.server.library.record(  # type: ignore[attr-defined]
                text=result.text,
                source_filename=result.filename,
                source_path=result.path,
                engine=result.engine,
                voice=result.voice,
                speed=result.speed,
                sample_rate=result.sample_rate,
                duration_seconds=result.duration_seconds,
                source_type="studio",
            )
        except Exception as error:  # noqa: BLE001 - never fail the request
            print(f"[web] could not add audio to the library: {error}", flush=True)
            return None

    def _handle_library_get(self, path: str) -> None:
        if path == "/api/library":
            self._handle_library_list()
            return

        parts = [part for part in path[len("/api/library/"):].split("/") if part]

        if not parts:
            self._send_error_json(404, "Page not found.", "not_found")
            return

        record_id = parts[0]

        if len(parts) == 2 and parts[1] == "audio":
            self._handle_library_audio(record_id)
            return

        if len(parts) == 2 and parts[1] == "mp3":
            self._handle_library_mp3(record_id)
            return

        if len(parts) == 1:
            try:
                record = self.server.library.get(record_id)  # type: ignore[attr-defined]
            except AppError as error:
                self._send_app_error(error)
                return

            self._send_json(200, {"ok": True, "record": record})
            return

        self._send_error_json(404, "Page not found.", "not_found")

    def _handle_library_list(self) -> None:
        query = parse_qs(urlparse(self.path).query)

        def first(name: str, default: str | None = None) -> str | None:
            values = query.get(name)
            return values[0] if values else default

        text = (first("q") or "").strip() or None
        sort = first("sort") or "newest"
        engine = first("engine") or None
        source = first("source") or None

        try:
            page = int(first("page") or "1")
            limit = int(first("limit") or str(DEFAULT_PAGE_SIZE))
        except ValueError:
            self._send_error_json(
                400,
                "page and limit must be integers.",
                "invalid_parameter",
            )
            return

        page = max(1, page)
        limit = min(max(1, limit), 100)

        try:
            records, total = self.server.library.list_records(  # type: ignore[attr-defined]
                query=text,
                sort=sort,
                engine=engine,
                source=source,
                page=page,
                limit=limit,
            )
        except AppError as error:
            self._send_app_error(error)
            return

        pages = (total + limit - 1) // limit if total else 0

        self._send_json(
            200,
            {
                "ok": True,
                "records": records,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "pages": pages,
                },
            },
        )

    def _handle_library_audio(self, record_id: str) -> None:
        try:
            record, path = self.server.library.audio_path(record_id)  # type: ignore[attr-defined]
        except AppError as error:
            self._send_app_error(error)
            return

        query = parse_qs(urlparse(self.path).query)
        download = (query.get("download") or ["0"])[0] in ("1", "true")
        disposition = "attachment" if download else "inline"

        self._send_bytes(
            200,
            path.read_bytes(),
            "audio/wav",
            {
                "Content-Disposition": (
                    f'{disposition}; filename="{record["filename"]}"'
                )
            },
        )

    def _library_target(self) -> str | None:
        """Extract a record id from ``/api/library/<id>``."""

        path = self.path.split("?", 1)[0]

        if not path.startswith("/api/library/"):
            return None

        parts = [part for part in path[len("/api/library/"):].split("/") if part]

        return parts[0] if len(parts) == 1 else ""

    def _handle_patch(self) -> None:
        record_id = self._library_target()

        if record_id:
            self._handle_library_rename(record_id)
            return

        path = self.path.split("?", 1)[0]

        if path.startswith("/api/pronunciation/"):
            entry_id = path[len("/api/pronunciation/"):].strip("/")

            if entry_id:
                self._handle_pronunciation_update(entry_id)
                return

        if path.startswith("/api/presets/"):
            preset_id = path[len("/api/presets/"):].strip("/")

            if preset_id:
                self._handle_preset_update(preset_id)
                return

        self._send_error_json(404, "Page not found.", "not_found")

    def _handle_library_rename(self, record_id: str) -> None:
        try:
            payload = self._read_json_body()
            record = self.server.library.rename(  # type: ignore[attr-defined]
                record_id,
                payload.get("name"),
            )
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(200, {"ok": True, "record": record})

    def _handle_delete(self) -> None:
        record_id = self._library_target()

        if record_id:
            self._handle_library_delete(record_id)
            return

        path = self.path.split("?", 1)[0]

        if path.startswith("/api/pronunciation/"):
            entry_id = path[len("/api/pronunciation/"):].strip("/")

            if entry_id:
                self._handle_pronunciation_delete(entry_id)
                return

        if path.startswith("/api/presets/"):
            preset_id = path[len("/api/presets/"):].strip("/")

            if preset_id:
                self._handle_preset_delete(preset_id)
                return

        self._send_error_json(404, "Page not found.", "not_found")

    def _handle_library_delete(self, record_id: str) -> None:
        try:
            deleted = self.server.library.delete(record_id)  # type: ignore[attr-defined]
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(200, {"ok": True, "deleted": deleted})

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------
    def _handle_document_extract(self) -> None:
        try:
            payload = self._read_json_body()
        except AppError as error:
            self._send_app_error(error)
            return

        data = payload.get("data_base64")

        if not isinstance(data, str) or not data:
            self._send_error_json(
                400,
                "The document is empty.",
                "empty_document",
            )
            return

        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            self._send_error_json(
                400,
                "The uploaded document could not be decoded.",
                "invalid_document",
            )
            return

        try:
            document = extract_document(payload.get("name"), raw)
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(200, {"ok": True, "document": document})

    # ------------------------------------------------------------------
    # Pronunciation dictionary
    # ------------------------------------------------------------------
    def _handle_pronunciation_get(self, path: str) -> None:
        if path == "/api/pronunciation":
            query = parse_qs(urlparse(self.path).query)
            text = (query.get("q") or [""])[0].strip() or None

            try:
                entries = self.server.library.list_pronunciation(text)  # type: ignore[attr-defined]
            except AppError as error:
                self._send_app_error(error)
                return

            self._send_json(200, {"ok": True, "entries": entries})
            return

        entry_id = path[len("/api/pronunciation/"):].strip("/")

        if not entry_id:
            self._send_error_json(404, "Page not found.", "not_found")
            return

        try:
            entry = self.server.library.get_pronunciation(entry_id)  # type: ignore[attr-defined]
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(200, {"ok": True, "entry": entry})

    def _handle_pronunciation_create(self) -> None:
        try:
            payload = self._read_json_body()
            entry = self.server.library.add_pronunciation(  # type: ignore[attr-defined]
                payload.get("word"),
                payload.get("pronunciation"),
                payload.get("description"),
            )
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(201, {"ok": True, "entry": entry})

    def _handle_pronunciation_update(self, entry_id: str) -> None:
        try:
            payload = self._read_json_body()
            entry = self.server.library.update_pronunciation(  # type: ignore[attr-defined]
                entry_id,
                payload.get("word"),
                payload.get("pronunciation"),
                payload.get("description"),
            )
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(200, {"ok": True, "entry": entry})

    def _handle_pronunciation_delete(self, entry_id: str) -> None:
        try:
            deleted = self.server.library.delete_pronunciation(entry_id)  # type: ignore[attr-defined]
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(200, {"ok": True, "deleted": deleted})

    # ------------------------------------------------------------------
    # Voice presets
    # ------------------------------------------------------------------
    def _handle_presets_get(self, path: str) -> None:
        if path == "/api/presets":
            try:
                presets = self.server.library.list_presets()  # type: ignore[attr-defined]
            except AppError as error:
                self._send_app_error(error)
                return

            self._send_json(200, {"ok": True, "presets": presets})
            return

        preset_id = path[len("/api/presets/"):].strip("/")

        if not preset_id:
            self._send_error_json(404, "Page not found.", "not_found")
            return

        try:
            preset = self.server.library.get_preset(preset_id)  # type: ignore[attr-defined]
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(200, {"ok": True, "preset": preset})

    def _handle_preset_create(self) -> None:
        try:
            payload = self._read_json_body()
            preset = self.server.library.create_preset(  # type: ignore[attr-defined]
                payload.get("name"),
                payload.get("engine"),
                payload.get("voice"),
                payload.get("speed"),
                payload.get("description"),
            )
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(201, {"ok": True, "preset": preset})

    def _handle_preset_update(self, preset_id: str) -> None:
        try:
            payload = self._read_json_body()
            preset = self.server.library.update_preset(  # type: ignore[attr-defined]
                preset_id,
                payload.get("name"),
                payload.get("engine"),
                payload.get("voice"),
                payload.get("speed"),
                payload.get("description"),
            )
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(200, {"ok": True, "preset": preset})

    def _handle_preset_delete(self, preset_id: str) -> None:
        try:
            deleted = self.server.library.delete_preset(preset_id)  # type: ignore[attr-defined]
        except AppError as error:
            self._send_app_error(error)
            return

        self._send_json(200, {"ok": True, "deleted": deleted})

    # ------------------------------------------------------------------
    # MP3 export
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_download_name(name: str, fallback: str = "voiceforge-audio") -> str:
        cleaned = "".join(
            character
            if (
                character.isascii()
                and (character.isalnum() or character in " -_().")
            )
            else "_"
            for character in str(name or "")
        ).strip(" ._")

        return (cleaned[:60] or fallback)

    def _handle_library_mp3(self, record_id: str) -> None:
        query = parse_qs(self.query_string())
        raw_bitrate = (query.get("bitrate") or [str(DEFAULT_BITRATE)])[0]

        try:
            bitrate = validate_bitrate(raw_bitrate)
            record, wav_path = self.server.library.audio_path(record_id)  # type: ignore[attr-defined]
            cache_dir = self.server.library.audio_dir.parent / "mp3"  # type: ignore[attr-defined]
            mp3_path, _ = cached_mp3(wav_path, cache_dir, bitrate)
        except AppError as error:
            self._send_app_error(error)
            return

        download_name = self._safe_download_name(record["name"]) + ".mp3"

        self._send_bytes(
            200,
            mp3_path.read_bytes(),
            "audio/mpeg",
            {
                "Content-Disposition": (
                    f'attachment; filename="{download_name}"'
                ),
                "X-MP3-Bitrate": str(bitrate),
            },
        )

    def query_string(self) -> str:
        return self.path.split("?", 1)[1] if "?" in self.path else ""

    # Keep the console readable: one short line per request.
    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {format % args}", flush=True)


def create_server(
    app: VoiceForgeApp,
    host: str = "127.0.0.1",
    port: int = 7860,
    jobs: JobManager | None = None,
    library: Library | None = None,
) -> VoiceForgeServer:
    """Create (but do not start) the VoiceForge HTTP server."""

    return VoiceForgeServer((host, port), app, jobs=jobs, library=library)


def run(
    app: VoiceForgeApp,
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = False,
) -> None:
    """Serve the interface until interrupted."""

    server = create_server(app, host=host, port=port)
    url = f"http://{host}:{server.server_address[1]}"

    # Remove stale MP3 conversions. Only cached *.mp3 files are eligible;
    # WAV files and the library database are never touched.
    try:
        summary = cleanup_mp3_cache(Path("outputs/mp3"))

        if summary["removed"]:
            print(
                f"[cache] removed {len(summary['removed'])} stale MP3 file(s), "
                f"freed {summary['freed_bytes']} bytes"
            )
        else:
            print("[cache] MP3 cache is clean")
    except Exception as error:  # noqa: BLE001 - never block startup
        print(f"[cache] cleanup skipped: {error}")

    print(f"VoiceForge is running at {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VoiceForge.")
    finally:
        server.server_close()
