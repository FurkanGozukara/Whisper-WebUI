import os
from ruamel.yaml import YAML
from gradio.utils import NamedString

from modules.utils.paths import DEFAULT_PARAMETERS_CONFIG_PATH

AUDIO_EXTENSION = ['.mp3', '.wav', '.wma', '.aac', '.flac', '.ogg', '.m4a', '.aiff', '.alac', '.opus', '.webm', '.ac3',
                   '.amr', '.au', '.mid', '.midi', '.mka']

VIDEO_EXTENSION = ['.mp4', '.mkv', '.flv', '.avi', '.mov', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg', '.3gp',
                   '.f4v', '.ogv', '.vob', '.mts', '.m2ts', '.divx', '.mxf', '.rm', '.rmvb', '.ts']

# .webm is both an audio and a video extension; listed once, so a batch does not queue each .webm twice
MEDIA_EXTENSION = list(dict.fromkeys(VIDEO_EXTENSION + AUDIO_EXTENSION))
FALLBACK_ENCODINGS = ['cp949', 'euc-kr']


def load_yaml(path: str = DEFAULT_PARAMETERS_CONFIG_PATH, use_fallback: bool = True):
    yaml = YAML(typ="safe")
    yaml.preserve_quotes = True
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return yaml.load(file)
    except UnicodeDecodeError:
        if not use_fallback:
            raise

        print(f"UTF-8 decoding failed for {path}. Trying fallback encodings...")

        try:
            with open(path, 'rb') as file:
                raw_bytes = file.read()
        except IOError as e:
            raise RuntimeError(f"Failed to read file {path} as binary.") from e

        for encoding in FALLBACK_ENCODINGS:
            try:
                content = raw_bytes.decode(encoding)
                config = yaml.load(content)
                print(f"Successfully loaded {path} with '{encoding}'. Consider converting the file to UTF-8.")
                return config
            except Exception as inner_e:  # Catches both UnicodeDecodeError and YAMLError
                print(f"   -> Loading with '{encoding}' failed: {inner_e}")
                continue

        raise RuntimeError(f"All attempted encodings ({['utf-8'] + FALLBACK_ENCODINGS}) failed to load {path}.")


def save_yaml(data: dict, path: str = DEFAULT_PARAMETERS_CONFIG_PATH):
    yaml = YAML(typ="safe")
    yaml.map_indent = 2
    yaml.sequence_indent = 4
    yaml.sequence_dash_offset = 2
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    yaml.sort_base_mapping_type_on_output = False

    with open(path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file)
    return path


def _is_media_file(file_name: str) -> bool:
    return os.path.splitext(file_name)[1].lower() in MEDIA_EXTENSION


def normalize_folder_path(path):
    """A folder typed or pasted into the UI: surrounding spaces and quotes removed ("Copy as path" in Explorer
    adds quotes, which made the batch fail with WinError 123 or find no files)."""
    if not path:
        return path
    path = str(path).strip()
    while len(path) >= 2 and path[0] == path[-1] and path[0] in "\"'":
        path = path[1:-1].strip()
    return path


def get_media_files(folder_path, include_sub_directory=False):
    """Media files of a folder, each once and in name order. Extensions are matched without case on every
    system: fnmatch is case-sensitive on Linux, so GOPR0001.MP4 was skipped there."""
    media_files = []

    if include_sub_directory:
        for root, dirs, files in os.walk(folder_path):
            dirs.sort()
            media_files.extend(
                os.path.join(root, file) for file in sorted(files)
                if _is_media_file(file) and os.path.isfile(os.path.join(root, file))
            )
    else:
        media_files = [
            os.path.join(folder_path, file) for file in sorted(os.listdir(folder_path))
            if _is_media_file(file) and os.path.isfile(os.path.join(folder_path, file))
        ]

    return media_files


def format_gradio_files(files: list):
    if not files:
        return files

    gradio_files = []
    for file in files:
        gradio_files.append(NamedString(file))
    return gradio_files


def is_video(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    return extension in VIDEO_EXTENSION


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        subtitle_content = f.read()
    return subtitle_content
