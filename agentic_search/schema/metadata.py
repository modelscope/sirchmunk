import enum
import json
import mimetypes
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from loguru import logger

from agentic_search.llm.openai import OpenAIChat
from agentic_search.schema.snapshot import SnapshotInfo
from agentic_search.utils.file_utils import md5_head


class FileType(enum.Enum):
    """
    Enumeration of supported file types for specialized schema extraction.
    """

    PLAINTEXT = "plaintext"
    CSV = "csv"
    JSON = "json"
    IMAGE = "image"
    PDF = "pdf"
    EXCEL = "excel"
    VIDEO = "video"
    OTHER = "other"
    DIRECTORY = "directory"


@dataclass
class FileInfo:
    """Base file metadata schema for any file or directory on disk.

    Args:
        file_or_url (Path): Absolute or relative path to the file/directory, or URL.
        type (FileType): Type of the file (e.g., text, image, pdf, directory).
        last_modified (datetime): Last modification time.
        create_time (datetime): Creation time (or metadata change time on Unix).
        do_snapshot (bool): Whether to capture a snapshot for this file.
        snapshot (SnapshotInfo): Snapshot information (if do_snapshot is True).

    Attributes (computed in __post_init__):
        size_bytes (int): File size in bytes.
        mime_type (str): MIME type inferred from file extension (e.g., "text/plain").
        extension (str): Lowercase file extension (e.g., ".txt").
        md5 (str): MD5 hash of the file. Empty string for directories. We use `md5_head` for quick fingerprinting.
        cache_key (str): Unique cache key based on MD5 and size for change detection.
    """

    file_or_url: Union[str, Path]
    last_modified: datetime
    create_time: datetime
    type: FileType = field(default=FileType.PLAINTEXT)
    do_snapshot: bool = field(default=True)
    snapshot: SnapshotInfo = field(default_factory=SnapshotInfo)

    size_bytes: int = field(init=False)
    mime_type: str = field(init=False)
    extension: str = field(init=False)
    md5: str = field(init=False)
    cache_key: str = field(init=False)

    def __post_init__(self) -> None:
        # TODO: add URLs support
        self.file_or_url = Path(self.file_or_url)
        self.extension = self.file_or_url.suffix.lower()
        self.size_bytes = self.file_or_url.stat().st_size
        self.mime_type = (
            mimetypes.guess_type(self.file_or_url)[0] or "application/octet-stream"
        )
        self.md5 = self.get_file_md5(file_path=self.file_or_url)
        self.cache_key = self.get_cache_key(file_or_url=self.file_or_url)

    def base_kwargs(self) -> Dict[str, Any]:
        """Return a dict of fields that can be safely passed to child dataclass __init__.

        Excludes fields with `init=False` (e.g., `extension`, `mime_type`), which are
        computed in __post_init__ and must not be passed during initialization.

        Returns:
            Dict[str, Any]: A dictionary containing only the init-accepted fields:
                {"path", "type", "size_bytes", "last_modified", "create_time", "do_snapshot"}.
        """
        # Get all fields declared in the dataclass (including inherited)
        init_fields = {f.name for f in self.__dataclass_fields__.values() if f.init}
        # Filter self.__dict__ to only include init-accepted fields
        return {k: v for k, v in self.__dict__.items() if k in init_fields}

    def to_dict(self):
        """Convert the FileInfo instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the FileInfo instance.
        """
        return {
            "path": str(self.file_or_url),
            "type": self.type.value,
            "size_bytes": self.size_bytes,
            "last_modified": self.last_modified.isoformat(),
            "create_time": self.create_time.isoformat(),
            "do_snapshot": self.do_snapshot,
            "snapshot": self.snapshot.to_dict(),
            "mime_type": self.mime_type,
            "extension": self.extension,
            "md5": self.md5,
            "cache_key": self.cache_key,
        }

    @staticmethod
    def from_dict(info: Dict[str, Any]):
        """Create a FileInfo instance from a dictionary.

        Args:
            info (Dict[str, Any]): A dictionary containing the fields of FileInfo.

        Returns:
            FileInfo: An instance of FileInfo populated with the provided data.
        """
        return FileInfo(
            file_or_url=Path(info["path"]),
            type=FileType(info["type"]),
            last_modified=datetime.fromisoformat(info["last_modified"]),
            create_time=datetime.fromisoformat(info["create_time"]),
            do_snapshot=info.get("do_snapshot", True),
        )

    @staticmethod
    def get_file_md5(file_path: Union[str, Path]) -> str:
        """
        Get the MD5 hash of a file if it exists.
        """
        file_path = Path(file_path)
        return md5_head(filepath=file_path) if file_path.is_file() else ""

    @staticmethod
    def get_cache_key(file_or_url: Union[str, Path]) -> str:
        """Generate a unique cache key for the file based on its path and MD5 hash.

        Returns:
            str: A unique cache key string.
        """
        md5: str = FileInfo.get_file_md5(file_path=file_or_url)
        size_bytes: int = Path(file_or_url).stat().st_size
        cache_key: str = f"{md5}_{str(size_bytes)}" if md5 else ""

        return cache_key

    @staticmethod
    def get_path_mtime(f_path: Union[str, Path], mtime: datetime) -> str:
        """
        Generate a unique identifier for a file based on its path and modification time for `unchanged` checking.

        Args:
            f_path (Union[str, Path]): The file path.
            mtime (datetime): The last modification time of the file.

        Returns:
            str: A unique identifier string in the format "path@ISO8601_mtime".
        """
        f_path: str = str(Path(f_path).resolve())
        return f"{f_path}@{mtime.isoformat()}"


@dataclass
class TextFileSchema(FileInfo):
    """Schema for plain-text-like files (e.g., .txt, .md, .log, .py).

    Args:
        encoding (Optional[str], optional): Detected character encoding (e.g., "utf-8"). Defaults to None.
        line_count (Optional[int], optional): Total number of lines — *not computed by default*. Defaults to None.
        first_lines_preview (List[str], optional): First few lines (up to 5) for quick inspection. Defaults to empty list.
    """

    encoding: Optional[str] = None
    line_count: Optional[int] = None
    first_lines_preview: List[str] = field(default_factory=list)


@dataclass
class CSVFileSchema(FileInfo):
    """Schema for CSV files.

    Args:
        delimiter (str, optional): Field delimiter detected (e.g., ",", ";"). Defaults to ",".
        has_header (bool, optional): Whether the first row appears to be a header. Defaults to True.
        columns (List[str], optional): List of column names. Empty if header not detected. Defaults to empty list.
        row_count (Optional[int], optional): Estimated or actual row count — *not computed by default*. Defaults to None.
        sample_rows (List[Dict[str, Any]], optional): Sample of up to 3 parsed rows as dictionaries. Defaults to empty list.
    """

    delimiter: str = field(default=",")
    has_header: bool = field(default=True)
    columns: List[str] = field(default_factory=list)
    row_count: Optional[int] = field(default=None)
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class JSONFileSchema(FileInfo):
    """Schema for JSON files.

    Args:
        root_type (Literal["object", "array", "scalar"], optional): Type of JSON root element. Defaults to "object".
        inferred_schema (Dict[str, Any], optional): Inferred JSON Schema (structural summary). Defaults to empty dict.
        is_valid_json (bool, optional): Whether the file is syntactically valid JSON. Defaults to True.
    """

    root_type: Literal["object", "array", "scalar"] = "object"
    inferred_schema: Dict[str, Any] = field(default_factory=dict)
    is_valid_json: bool = field(default=True)


@dataclass
class ImageFileSchema(FileInfo):
    """Schema for image files (PNG, JPEG, etc.).

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        mode (str): PIL color mode (e.g., "RGB", "RGBA", "L").
        format (str): Image format (e.g., "PNG", "JPEG").
        color_profile (Optional[str], optional): Color profile type if embedded (e.g., "icc"). Defaults to None.
    """

    width: int = field(default=0)
    height: int = field(default=0)
    mode: str = field(default=None)
    format: str = field(default=None)
    color_profile: Optional[str] = field(default=None)


@dataclass
class PDFFileSchema(FileInfo):
    """Schema for PDF documents.

    Args:
        page_count (int): Number of pages in the document.
        author (Optional[str], optional): Document author from metadata. Defaults to None.
        title (Optional[str], optional): Document title from metadata. Defaults to None.
        keywords (List[str], optional): Keywords from metadata. Defaults to empty list.
        is_encrypted (bool, optional): Whether the PDF is encrypted. Defaults to False.
    """

    page_count: int = field(default=0)
    author: Optional[str] = field(default=None)
    title: Optional[str] = field(default=None)
    keywords: List[str] = field(default_factory=list)
    is_encrypted: bool = field(default=False)


@dataclass
class ExcelFileSchema(FileInfo):
    """Schema for Excel workbooks (.xlsx, .xls).

    Args:
        sheet_names (List[str], optional): Names of all sheets in the workbook. Defaults to empty list.
        sheet_schemas (Dict[str, CSVFileSchema], optional): Mapping from sheet name to its CSV-like schema.
            Each sheet is treated as a tabular dataset. Defaults to empty dict.
    """

    sheet_names: List[str] = field(default_factory=list)
    sheet_schemas: Dict[str, CSVFileSchema] = field(default_factory=dict)


@dataclass
class VideoFileSchema(FileInfo):
    """Schema for video files (MP4, AVI, MOV, etc.).

    Extracts metadata via `ffprobe` (part of FFmpeg). Falls back to basic FileInfo if `ffprobe` is unavailable.

    Args:
        duration_sec (float): Duration of the video in seconds.
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.
        codec (str): Video codec name (e.g., "h264", "hevc", "vp9").
        framerate (float): Frames per second (e.g., 29.97, 60.0).
        bitrate_kbps (int): Average video bitrate in kbps.
        has_audio (bool): Whether the file contains an audio stream.
        audio_codec (Optional[str], optional): Audio codec name if present (e.g., "aac", "mp3"). Defaults to None.
        rotation (int, optional): Display rotation (0, 90, 180, 270) — from metadata tags. Defaults to 0.
    """

    duration_sec: float = field(default=0.0)
    width: int = field(default=0)
    height: int = field(default=0)
    codec: str = field(default="")
    framerate: float = field(default=0.0)
    bitrate_kbps: int = field(default=0)
    has_audio: bool = field(default=False)
    audio_codec: Optional[str] = field(default=None)
    rotation: int = field(default=0)


def build_file_schema(
    path: Union[str, Path],
    llm: Optional[OpenAIChat] = None,
) -> FileInfo:
    """
    Build a typed schema object for a given file path, based on its type.

    Supports: text, CSV, JSON, images, PDF, Excel, and video files.
    Falls back to basic FileInfo if parsing fails or type is unknown.

    Args:
        path (Path): Path to the file (must exist).
        llm (OpenAIChat): The llm client of OpenAI api.

    Returns:
        FileInfo: An instance of FileInfo or one of its subclasses with type-specific metadata.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    stat = path.stat()
    base_info = FileInfo(
        file_or_url=path,
        last_modified=datetime.fromtimestamp(stat.st_mtime),
        create_time=datetime.fromtimestamp(stat.st_ctime),
        do_snapshot=True,
    )

    if path.is_dir():
        base_info.type = FileType.DIRECTORY
        return base_info

    try:
        # --- Text-like files ---
        if base_info.extension in {
            ".txt",
            ".md",
            ".log",
            ".py",
            ".json",
            ".yml",
            ".yaml",
            ".xml",
        }:
            base_info.type = FileType.PLAINTEXT
            schema = _build_text_schema(
                file_info=base_info,
                llm=llm,
            )
            if base_info.extension == ".json":
                base_info.type = FileType.JSON
                schema = _build_json_schema(base_info, schema)
            return schema

        # --- CSV ---
        if base_info.extension == ".csv":
            base_info.type = FileType.CSV
            return _build_csv_schema(base_info)

        # --- Images ---
        if base_info.mime_type.startswith("image/"):
            base_info.type = FileType.IMAGE
            return _build_image_schema(base_info)

        # --- PDF ---
        if base_info.extension == ".pdf":
            base_info.type = FileType.PDF
            return _build_pdf_schema(base_info)

        # --- Excel ---
        if base_info.extension in {".xlsx", ".xls"}:
            base_info.type = FileType.EXCEL
            return _build_excel_schema(base_info)

        # --- Video ---
        if base_info.mime_type.startswith("video/") or base_info.extension in {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".webm",
            ".flv",
            ".wmv",
        }:
            base_info.type = FileType.VIDEO
            video_schema = _build_video_schema(base_info)
            if video_schema:
                return video_schema

    except Exception as e:
        logger.warning(f"Error building schema for {path}: {e}")

    # Fallback
    return base_info


def _build_text_schema(
    file_info: FileInfo,
    llm: Optional[OpenAIChat] = None,
) -> TextFileSchema:
    """Build TextFileSchema by sampling the beginning of the file."""
    try:
        with open(file_info.file_or_url, "rb") as f:
            raw = f.read(2048)
        encoding = _detect_encoding(raw)
        text = raw.decode(encoding, errors="replace")
        lines = text.splitlines()

        if file_info.do_snapshot:
            from agentic_search.schema.snapshot import TextSnapshot

            snapshot_info: SnapshotInfo = TextSnapshot(
                llm=llm,
            ).sampling(
                file_path=file_info.file_or_url,
            )
            file_info.snapshot = snapshot_info

        return TextFileSchema(
            **file_info.base_kwargs(),
            encoding=encoding,
            first_lines_preview=lines[:5],
        )
    except Exception as e:
        logger.warning(f"Error building text schema for {file_info.file_or_url}: {e}")
        return TextFileSchema(**file_info.base_kwargs())


def _build_json_schema(file_info: FileInfo, fallback: TextFileSchema) -> JSONFileSchema:
    """Attempt to parse and infer schema for JSON files using Genson."""
    try:
        import json

        from genson import SchemaBuilder

        data = json.loads(
            file_info.file_or_url.read_text(encoding=fallback.encoding or "utf-8")
        )
        builder = SchemaBuilder()
        builder.add_object(data)
        inferred = (
            builder.to_schema()
        )  # Returns dict, e.g., {"type": "object", "properties": {...}}

        return JSONFileSchema(
            **file_info.base_kwargs(),
            root_type=_json_root_type(data),
            inferred_schema=inferred,
            is_valid_json=True,
        )
    except Exception as e:
        logger.warning(f"Error building JSON schema for {file_info.file_or_url}: {e}")
        return JSONFileSchema(**file_info.base_kwargs(), is_valid_json=False)


def _build_csv_schema(file_info: FileInfo) -> CSVFileSchema:
    """Build CSV schema using csv.Sniffer and sample parsing."""
    import csv

    with open(
        file_info.file_or_url, newline="", encoding="utf-8", errors="ignore"
    ) as f:
        sample = f.read(4096)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        has_header = sniffer.has_header(sample)
        f.seek(0)
        reader = csv.DictReader(f, dialect=dialect)
        columns = list(reader.fieldnames) if reader.fieldnames else []
        sample_rows = [row for _, row in zip(range(3), reader)]
        return CSVFileSchema(
            **file_info.base_kwargs(),
            delimiter=dialect.delimiter,
            has_header=has_header,
            columns=columns,
            sample_rows=sample_rows,
        )


def _build_image_schema(file_info: FileInfo) -> ImageFileSchema:
    """Build image schema using PIL (lightweight metadata read)."""
    from PIL import Image

    with Image.open(file_info.file_or_url) as img:
        return ImageFileSchema(
            **file_info.base_kwargs(),
            width=img.width,
            height=img.height,
            mode=img.mode,
            format=img.format or "",
            color_profile="icc" if "icc_profile" in img.info else None,
        )


def _build_pdf_schema(file_info: FileInfo) -> PDFFileSchema:
    """Build PDF schema using pypdf (streaming, no full load)."""
    import pypdf

    with open(file_info.file_or_url, "rb") as f:
        reader = pypdf.PdfReader(f)
        meta = reader.metadata or {}
        return PDFFileSchema(
            **file_info.base_kwargs(),
            page_count=len(reader.pages),
            author=meta.get("/Author"),
            title=meta.get("/Title"),
            keywords=_parse_pdf_keywords(meta.get("/Keywords")),
            is_encrypted=reader.is_encrypted,
        )


def _build_excel_schema(file_info: FileInfo) -> ExcelFileSchema:
    """Build Excel schema using pandas (reads only headers + few rows)."""
    import pandas as pd

    sheets = pd.read_excel(file_info.file_or_url, sheet_name=None, nrows=5)
    sheet_schemas = {}

    base_kwargs = {
        "path": file_info.file_or_url,
        "last_modified": file_info.last_modified,
        "create_time": file_info.create_time,
        "is_dir": False,  # sheets are not dirs
    }

    for name, df in sheets.items():
        cols = df.columns.tolist()
        sample = df.head(3).to_dict(orient="records")
        sheet_schemas[name] = CSVFileSchema(
            **base_kwargs,
            delimiter=",",
            has_header=True,
            columns=cols,
            sample_rows=sample,
        )

    return ExcelFileSchema(
        **file_info.base_kwargs(),
        sheet_names=list(sheets.keys()),
        sheet_schemas=sheet_schemas,
    )


def _build_video_schema(file_info: FileInfo) -> Optional[VideoFileSchema]:
    """
    Build video schema by calling `ffprobe` to extract metadata.

    Args:
        file_info (FileInfo): Base file info for the video file.

    Returns:
        VideoFileSchema if ffprobe succeeds; None otherwise (caller should fall back).
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file_info.file_or_url),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        format_info = data.get("format", {})
        streams = data.get("streams", [])

        # Find best video stream
        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), None
        )
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), None
        )

        if not video_stream:
            return None

        # Extract key fields with safe defaults
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        codec = video_stream.get("codec_name", "unknown")
        duration = float(format_info.get("duration", 0.0))
        bitrate = int(int(format_info.get("bit_rate", 0)) // 1000)  # bps → kbps

        # Framerate: may be "30/1" or "2997/100"
        fps_str = video_stream.get("avg_frame_rate", "0/1")
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            framerate = num / den if den != 0 else 0.0
        else:
            framerate = float(fps_str or 0.0)

        # Rotation from metadata tags
        rotation = 0
        tags = video_stream.get("tags", {})
        rotate_tag = tags.get("rotate") or tags.get("ROTATE")
        if rotate_tag and re.match(r"^-?\d+$", rotate_tag):
            rotation = int(rotate_tag) % 360

        return VideoFileSchema(
            **file_info.base_kwargs(),
            duration_sec=duration,
            width=width,
            height=height,
            codec=codec,
            framerate=framerate,
            bitrate_kbps=bitrate,
            has_audio=audio_stream is not None,
            audio_codec=audio_stream.get("codec_name") if audio_stream else None,
            rotation=rotation,
        )

    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
        OSError,
    ) as e:
        logger.warning(f"Error building video schema for {file_info.file_or_url}: {e}")
        return None


def _detect_encoding(raw: bytes) -> str:
    """Detect text encoding from raw bytes using charset_normalizer (fallback to utf-8)."""
    try:
        import charset_normalizer

        result = charset_normalizer.from_bytes(raw).best()
        return result.encoding if result else "utf-8"
    except ImportError:
        return "utf-8"


def _json_root_type(obj) -> Literal["object", "array", "scalar"]:
    """Determine JSON root type for schema labeling."""
    if isinstance(obj, dict):
        return "object"
    elif isinstance(obj, list):
        return "array"
    else:
        return "scalar"


def _parse_pdf_keywords(raw: Optional[str]) -> List[str]:
    """Parse PDF keywords string into a clean list."""
    if not raw:
        return []
    # Normalize: split by common delimiters and strip
    return [k.strip() for k in re.split(r"[,;|]", raw) if k.strip()]
