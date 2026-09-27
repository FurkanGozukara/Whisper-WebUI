import sys

from tqdm import tqdm


class DownloadProgressTqdm(tqdm):
    """Download bar for CMD. Hugging Face hides its own bars when stderr is not a terminal, and the job
    worker's stderr is a pipe, so without this a first-use model download printed no progress."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("file", sys.stderr)
        kwargs.setdefault("dynamic_ncols", True)
        super().__init__(*args, **kwargs)
