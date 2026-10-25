from typing import Any


def open_media_input(input_file: Any) -> Any:
    """Open original media with PyAV, preserving tolerant metadata handling.

    PyAV 19 removed metadata_errors and always uses UTF-8 with surrogateescape.
    Older releases need metadata_errors="ignore" for files with malformed tags.
    Select the API before opening: no failed-open retry, conversion or temp file.
    Keep PyAV lazy so importing the UI does not load the decoder libraries.
    """
    import av

    if int(av.__version__.partition(".")[0]) >= 19:
        return av.open(input_file, mode="r")
    return av.open(input_file, mode="r", metadata_errors="ignore")
