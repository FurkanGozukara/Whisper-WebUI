#!/bin/bash

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
# The distribution keeps its platform-specific dependency pins next to Whisper-WebUI.
if [ ! -f ../requirements_whisper.txt ] || [ ! -f ../uv_build_constraints.txt ]; then
    echo "The distribution requirements are missing. Extract the complete installer archive first." >&2
    exit 1
fi

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv venv || exit 1
fi

source venv/bin/activate || exit 1
python -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))' || {
    echo "Python 3.12 is required. Run the distribution installer to rebuild this environment." >&2
    exit 1
}

export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy
export UV_COMPILE_BYTECODE=1
python -m pip install -U pip uv || exit 1
uv pip install -r ../requirements_whisper.txt --index-strategy unsafe-best-match --build-constraints ../uv_build_constraints.txt || {
    echo ""
    echo "Requirements installation failed. Please remove the venv folder and run the script again."
    deactivate
    exit 1
}
uv pip install git+https://github.com/NVIDIA/NeMo.git --index-strategy unsafe-best-match --build-constraints ../uv_build_constraints.txt || exit 1
python ../DownloadModels.py || exit 1
echo "Requirements installed successfully."

deactivate
