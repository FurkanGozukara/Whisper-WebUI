import argparse
import hashlib
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests


SCRIPT_DIR = Path(__file__).resolve().parent
WEBUI_DIR = SCRIPT_DIR / "Whisper-WebUI"
DEFAULT_DIARIZATION_DIR = WEBUI_DIR / "models" / "Diarization"

PIPELINE_REPO_ID = "MonsterMMORPG/Wan_GGUF"
PIPELINE_SUBFOLDER = "Speaker_Diarization_3_1"
PIPELINE_CONFIG_PATH = f"{PIPELINE_SUBFOLDER}/config.yaml"

MODEL_ALIASES = {"all", "diarization", "speaker-diarization", "speaker_diarization"}

IGNORED_PIPELINE_PREFIXES = (
    f"{PIPELINE_SUBFOLDER}/.github/",
    f"{PIPELINE_SUBFOLDER}/reproducible_research/",
)

CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT = 300
DEFAULT_RETRIES = 3


@dataclass(frozen=True)
class RepoFile:
    repo_id: str
    remote_path: str
    local_path: Path
    size: Optional[int] = None
    sha256: Optional[str] = None


@dataclass(frozen=True)
class DependencySource:
    requested_repo_id: str
    source_repo_id: str
    source_subfolder: Optional[str]
    local_dir_name: str

    @property
    def source_label(self) -> str:
        if self.source_subfolder:
            return f"{self.source_repo_id}/{self.source_subfolder}"
        return self.source_repo_id


class DownloadError(RuntimeError):
    pass


DEPENDENCY_SOURCE_OVERRIDES: Dict[str, DependencySource] = {
    "pyannote/segmentation-3.0": DependencySource(
        requested_repo_id="pyannote/segmentation-3.0",
        source_repo_id=PIPELINE_REPO_ID,
        source_subfolder="pyannote_segmentation3",
        local_dir_name="pyannote_segmentation3",
    ),
}


def normalize_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return path_str
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def resolve_target_dirs(download_dir: Optional[str]) -> Tuple[Path, Path]:
    requested = Path(normalize_path(download_dir) or DEFAULT_DIARIZATION_DIR).resolve()
    if requested.name == PIPELINE_SUBFOLDER:
        return requested.parent, requested
    return requested, requested / PIPELINE_SUBFOLDER


def get_token(explicit_token: Optional[str]) -> Optional[str]:
    return (
        explicit_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def build_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {"User-Agent": "WhisperWebUI-DiarizationDownloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def hf_file_url(repo_id: str, remote_path: str) -> str:
    quoted_path = quote(remote_path, safe="/")
    return f"https://huggingface.co/{repo_id}/resolve/main/{quoted_path}"


def hf_tree_url(repo_id: str, path: Optional[str] = None) -> str:
    base_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    if path:
        return f"{base_url}/{quote(path.strip('/'), safe='/')}"
    return base_url


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: int,
    retries: int,
    **kwargs,
) -> requests.Response:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            if response.status_code in {401, 403}:
                return response
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 ** (attempt - 1), 8))
    raise DownloadError(f"Request failed after {retries} attempts: {url}\n{last_error}") from last_error


def list_repo_tree(
    session: requests.Session,
    repo_id: str,
    token: Optional[str],
    *,
    path: Optional[str] = None,
    timeout: int,
    retries: int,
) -> List[dict]:
    entries: List[dict] = []
    next_url: Optional[str] = hf_tree_url(repo_id, path)
    params: Optional[Dict[str, str]] = {"recursive": "1"}

    while next_url:
        response = request_with_retry(
            session,
            "GET",
            next_url,
            timeout=timeout,
            retries=retries,
            headers=build_headers(token),
            params=params,
        )
        params = None
        entries.extend(response.json())

        link_header = response.headers.get("Link") or ""
        next_match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        next_url = next_match.group(1) if next_match else None

    return entries


def get_repo_file_map(
    session: requests.Session,
    repo_id: str,
    token: Optional[str],
    *,
    path: Optional[str] = None,
    timeout: int,
    retries: int,
) -> Dict[str, dict]:
    entries = list_repo_tree(
        session,
        repo_id,
        token,
        path=path,
        timeout=timeout,
        retries=retries,
    )
    return {
        entry["path"]: entry
        for entry in entries
        if entry.get("type") == "file"
    }


def read_remote_text(
    session: requests.Session,
    repo_id: str,
    remote_path: str,
    token: Optional[str],
    *,
    timeout: int,
    retries: int,
) -> str:
    response = request_with_retry(
        session,
        "GET",
        hf_file_url(repo_id, remote_path),
        timeout=timeout,
        retries=retries,
        headers=build_headers(token),
    )
    if response.status_code in {401, 403}:
        raise DownloadError(
            f"Access denied while reading {repo_id}/{remote_path}. "
            "If this is a gated repo, pass --token or set HF_TOKEN."
        )
    text = response.text
    response.close()
    return text


def extract_dependency_repo_ids(config_text: str) -> Dict[str, str]:
    repo_ids: Dict[str, str] = {}
    pattern = re.compile(
        r"^\s*(?P<key>[A-Za-z0-9_]+):\s*[\"']?(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']?\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(config_text):
        key = match.group("key")
        repo_id = match.group("repo")
        if repo_id != PIPELINE_REPO_ID:
            repo_ids[key] = repo_id
    return repo_ids


def yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def rewrite_pipeline_config(
    config_text: str,
    dependency_sources: Dict[str, DependencySource],
    pipeline_dir: Path,
) -> str:
    rewritten = config_text

    for key, dependency_source in dependency_sources.items():
        local_weights_path = (
            pipeline_dir / dependency_source.local_dir_name / "pytorch_model.bin"
        ).resolve()
        pattern = re.compile(
            rf"^(?P<indent>\s*){re.escape(key)}:\s*[\"']?{re.escape(dependency_source.requested_repo_id)}[\"']?\s*$",
            re.MULTILINE,
        )

        def _replace(match: re.Match) -> str:
            indent = match.group("indent")
            return (
                f"{indent}{key}:\n"
                f"{indent}  checkpoint: {yaml_quote(local_weights_path.as_posix())}"
            )

        rewritten, replacements = pattern.subn(_replace, rewritten, count=1)
        if replacements == 0:
            raise DownloadError(
                f"Could not rewrite local checkpoint path for '{key}: {dependency_source.requested_repo_id}' in pipeline config."
            )

    return rewritten


def should_skip_pipeline_file(remote_path: str) -> bool:
    return any(remote_path.startswith(prefix) for prefix in IGNORED_PIPELINE_PREFIXES)


def repo_file_sha256(entry: dict) -> Optional[str]:
    lfs = entry.get("lfs") or {}
    oid = lfs.get("oid")
    if not oid:
        return None
    if len(oid) == 64 and all(ch in "0123456789abcdef" for ch in oid.lower()):
        return oid.lower()
    return None


def resolve_dependency_source(requested_repo_id: str) -> DependencySource:
    override = DEPENDENCY_SOURCE_OVERRIDES.get(requested_repo_id)
    if override is not None:
        return override
    return DependencySource(
        requested_repo_id=requested_repo_id,
        source_repo_id=requested_repo_id,
        source_subfolder=None,
        local_dir_name=requested_repo_id.split("/")[-1],
    )


def build_download_plan(
    session: requests.Session,
    token: Optional[str],
    pipeline_dir: Path,
    *,
    timeout: int,
    retries: int,
) -> Tuple[List[RepoFile], str, Dict[str, DependencySource]]:
    pipeline_file_map = get_repo_file_map(
        session,
        PIPELINE_REPO_ID,
        token,
        path=PIPELINE_SUBFOLDER,
        timeout=timeout,
        retries=retries,
    )
    if PIPELINE_CONFIG_PATH not in pipeline_file_map:
        raise DownloadError(f"Missing required pipeline file in repo: {PIPELINE_CONFIG_PATH}")

    config_text = read_remote_text(
        session,
        PIPELINE_REPO_ID,
        PIPELINE_CONFIG_PATH,
        token,
        timeout=timeout,
        retries=retries,
    )
    dependency_repo_ids = extract_dependency_repo_ids(config_text)
    if not dependency_repo_ids:
        raise DownloadError(
            "No dependency repos were found in the remote diarization config. "
            "Upstream layout may have changed."
        )
    dependency_sources = {
        key: resolve_dependency_source(repo_id)
        for key, repo_id in dependency_repo_ids.items()
    }

    files: List[RepoFile] = []

    for remote_path, entry in sorted(pipeline_file_map.items()):
        if not remote_path.startswith(f"{PIPELINE_SUBFOLDER}/"):
            continue
        if should_skip_pipeline_file(remote_path):
            continue
        local_relative = Path(remote_path).relative_to(PIPELINE_SUBFOLDER)
        files.append(
            RepoFile(
                repo_id=PIPELINE_REPO_ID,
                remote_path=remote_path,
                local_path=pipeline_dir / local_relative,
                size=entry.get("size"),
                sha256=repo_file_sha256(entry),
            )
        )

    unique_sources: Dict[Tuple[str, Optional[str], str], DependencySource] = {}
    for dependency_source in dependency_sources.values():
        unique_sources[
            (
                dependency_source.source_repo_id,
                dependency_source.source_subfolder,
                dependency_source.local_dir_name,
            )
        ] = dependency_source

    for dependency_source in unique_sources.values():
        repo_file_map = get_repo_file_map(
            session,
            dependency_source.source_repo_id,
            token,
            path=dependency_source.source_subfolder,
            timeout=timeout,
            retries=retries,
        )

        source_subfolder = dependency_source.source_subfolder
        if source_subfolder:
            repo_file_map = {
                path[len(source_subfolder) + 1:]: entry
                for path, entry in repo_file_map.items()
                if path.startswith(f"{source_subfolder}/")
            }
            remote_prefix = f"{source_subfolder}/"
        else:
            remote_prefix = ""

        if "pytorch_model.bin" not in repo_file_map:
            raise DownloadError(
                f"Dependency source does not expose pytorch_model.bin: {dependency_source.source_label}"
            )

        for relative_remote_path, entry in sorted(repo_file_map.items()):
            local_relative = Path(dependency_source.local_dir_name) / relative_remote_path
            files.append(
                RepoFile(
                    repo_id=dependency_source.source_repo_id,
                    remote_path=f"{remote_prefix}{relative_remote_path}",
                    local_path=pipeline_dir / local_relative,
                    size=entry.get("size"),
                    sha256=repo_file_sha256(entry),
                )
            )

    return files, config_text, dependency_sources


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_file_matches(file: RepoFile, force: bool) -> bool:
    if force or not file.local_path.is_file():
        return False
    if file.size is not None and file.local_path.stat().st_size != file.size:
        return False
    if file.sha256:
        return sha256_file(file.local_path) == file.sha256
    return True


def download_file(
    session: requests.Session,
    file: RepoFile,
    token: Optional[str],
    *,
    timeout: int,
    retries: int,
    force: bool,
) -> str:
    if existing_file_matches(file, force):
        return "skipped"

    file.local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file.local_path.with_suffix(file.local_path.suffix + ".part")

    if tmp_path.exists():
        tmp_path.unlink()

    response = request_with_retry(
        session,
        "GET",
        hf_file_url(file.repo_id, file.remote_path),
        timeout=timeout,
        retries=retries,
        headers=build_headers(token),
        stream=True,
        allow_redirects=True,
    )

    if response.status_code in {401, 403}:
        response.close()
        raise DownloadError(
            f"Access denied while downloading {file.repo_id}/{file.remote_path}. "
            "Accept the gated model terms on Hugging Face and pass --token or set HF_TOKEN."
        )

    try:
        expected_size = file.size
        content_length = response.headers.get("Content-Length")
        if expected_size is None and content_length and content_length.isdigit():
            expected_size = int(content_length)

        downloaded = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)

        if expected_size is not None and downloaded != expected_size:
            raise DownloadError(
                f"Downloaded size mismatch for {file.remote_path}: expected {expected_size}, got {downloaded}"
            )

        if file.sha256:
            actual_sha256 = sha256_file(tmp_path)
            if actual_sha256 != file.sha256:
                raise DownloadError(
                    f"SHA256 mismatch for {file.remote_path}: expected {file.sha256}, got {actual_sha256}"
                )
    finally:
        response.close()

    tmp_path.replace(file.local_path)
    return "downloaded"


def write_rewritten_pipeline_config(
    pipeline_dir: Path,
    original_config_text: str,
    dependency_sources: Dict[str, DependencySource],
) -> None:
    config_path = pipeline_dir / "config.yaml"
    rewritten = rewrite_pipeline_config(
        original_config_text,
        dependency_sources,
        pipeline_dir,
    )
    config_path.write_text(rewritten, encoding="utf-8")


def download_models(
    model_id: Optional[str] = None,
    download_dir: Optional[str] = None,
    *,
    token: Optional[str] = None,
    force: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
) -> None:
    normalized_model_id = (model_id or "diarization").strip().lower()
    if normalized_model_id not in MODEL_ALIASES:
        raise DownloadError(
            "This script now only downloads whisper diarization assets. "
            f"Unsupported model selection: {model_id}"
        )

    diarization_dir, pipeline_dir = resolve_target_dirs(download_dir)
    resolved_token = get_token(token)

    diarization_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    print("Model: Whisper Speaker Diarization")
    print(f"Target directory: {pipeline_dir}")
    if resolved_token:
        print("Token: provided")

    session = requests.Session()
    session.headers.update({"User-Agent": "WhisperWebUI-DiarizationDownloader/1.0"})

    files, config_text, dependency_sources = build_download_plan(
        session,
        resolved_token,
        pipeline_dir,
        timeout=timeout,
        retries=retries,
    )

    print("Dependencies:")
    for key, dependency_source in sorted(dependency_sources.items()):
        if dependency_source.requested_repo_id == dependency_source.source_label:
            print(f"  {key}: {dependency_source.source_label}")
        else:
            print(
                f"  {key}: {dependency_source.requested_repo_id} -> {dependency_source.source_label}"
            )

    downloaded = 0
    skipped = 0

    for index, file in enumerate(files, start=1):
        display_path = file.local_path.relative_to(pipeline_dir)
        print(f"[{index}/{len(files)}] {display_path}")
        result = download_file(
            session,
            file,
            resolved_token,
            timeout=timeout,
            retries=retries,
            force=force,
        )
        if result == "downloaded":
            downloaded += 1
        else:
            skipped += 1

    write_rewritten_pipeline_config(pipeline_dir, config_text, dependency_sources)

    print()
    print("Offline diarization bundle is ready.")
    print(f"Pipeline directory: {pipeline_dir}")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped: {skipped}")
    print("config.yaml was rewritten to absolute local checkpoint paths.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download whisper diarization assets into Whisper-WebUI/models/Diarization "
            "without relying on Hugging Face cache layout."
        )
    )
    parser.add_argument(
        "--model",
        default="diarization",
        help=(
            "Backward-compatible option. Only diarization is supported now. "
            "Accepted values: diarization, speaker-diarization, all."
        ),
    )
    parser.add_argument(
        "--dir",
        type=str,
        help=(
            "Base diarization directory or exact pipeline directory. "
            f"Default: {DEFAULT_DIARIZATION_DIR}"
        ),
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face token. Falls back to HF_TOKEN/HUGGINGFACE_TOKEN if omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if matching local files already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-request timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Number of retries per request. Default: {DEFAULT_RETRIES}",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        download_models(
            model_id=args.model,
            download_dir=args.dir,
            token=args.token,
            force=args.force,
            timeout=args.timeout,
            retries=args.retries,
        )
    except DownloadError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[ERROR] Download cancelled by user.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
