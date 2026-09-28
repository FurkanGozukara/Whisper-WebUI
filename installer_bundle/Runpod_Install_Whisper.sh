#!/usr/bin/env bash

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1

apt update --yes
apt install xz-utils curl --yes

FFMPEG_URL="https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main/ffmpeg-n9.0-latest-linux64-gpl-9.0.tar.xz"
echo "Installing FFmpeg n9.0 as the system default..."
FFMPEG_TMP="$(mktemp -d)"
if ! curl -fL -o "$FFMPEG_TMP/ffmpeg.tar.xz" "$FFMPEG_URL" \
    || ! tar -xJf "$FFMPEG_TMP/ffmpeg.tar.xz" -C "$FFMPEG_TMP" \
    || ! install -m 755 "$FFMPEG_TMP"/ffmpeg-*/bin/ffmpeg "$FFMPEG_TMP"/ffmpeg-*/bin/ffprobe /usr/local/bin/ \
    || ! ln -sf /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/bin/; then
    echo "Failed to install FFmpeg. Exiting..."
    exit 1
fi
rm -rf "$FFMPEG_TMP"
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "FFmpeg or ffprobe installation could not be verified. Exiting..."
    exit 1
fi
echo "Using $(ffmpeg -version | head -n 1)"

export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy
# Compile the libraries to bytecode during install; otherwise the first app start does it and looks frozen
export UV_COMPILE_BYTECODE=1
export UV_PYTHON_PREFERENCE=only-managed

if [ ! -d Whisper-WebUI/.git ]; then
    git clone https://github.com/FurkanGozukara/Whisper-WebUI || exit 1
fi

cd Whisper-WebUI || exit 1

git reset --hard || exit 1

git pull || exit 1

echo "Provisioning the latest stable Python 3.12 with uv..."

VENV_DIR="venv"
VENV_PYTHON="$VENV_DIR/bin/python"
PYTHON_REQUEST="3.12"
PYTHON_VERSION_CMD='import sys; print(".".join(map(str, sys.version_info[:3])))'

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    if command -v curl >/dev/null 2>&1; then
        if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
            echo "Failed to install uv. Exiting..."
            exit 1
        fi
    elif command -v wget >/dev/null 2>&1; then
        if ! wget -qO- https://astral.sh/uv/install.sh | sh; then
            echo "Failed to install uv. Exiting..."
            exit 1
        fi
    else
        echo "Neither curl nor wget is available to install uv. Exiting..."
        exit 1
    fi
fi

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to install uv. Exiting..."
    exit 1
fi
if ! uv python install "$PYTHON_REQUEST"; then
    echo "Failed to download Python $PYTHON_REQUEST. Exiting..."
    exit 1
fi

TARGET_PYTHON="$(uv python find "$PYTHON_REQUEST" 2>/dev/null)"
if [ -z "$TARGET_PYTHON" ] || [ ! -x "$TARGET_PYTHON" ]; then
    echo "Could not locate the managed Python $PYTHON_REQUEST interpreter. Exiting..."
    exit 1
fi

PYTHON_FULL_VERSION="$("$TARGET_PYTHON" -c "$PYTHON_VERSION_CMD")"
echo "Using Python $PYTHON_FULL_VERSION from $TARGET_PYTHON"

if [ -x "$VENV_PYTHON" ] && [ "$("$VENV_PYTHON" -c "$PYTHON_VERSION_CMD" 2>/dev/null)" = "$PYTHON_FULL_VERSION" ]; then
    echo "Reusing existing Python $PYTHON_FULL_VERSION virtual environment."
else
    if [ -e "$VENV_DIR" ]; then
        echo "Existing virtual environment is not on Python $PYTHON_FULL_VERSION; recreating it..."
        rm -rf "$VENV_DIR"
    else
        echo "No existing virtual environment was found; creating one..."
    fi

    if ! uv venv --python "$TARGET_PYTHON" --seed "$VENV_DIR"; then
        echo "Failed to create the Python $PYTHON_FULL_VERSION virtual environment. Exiting..."
        exit 1
    fi
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Python 3.12 virtual environment setup failed. Exiting..."
    exit 1
fi

source ./venv/bin/activate || exit 1

python -m pip install --upgrade pip || exit 1

pip install uv || exit 1

uv pip install wheel || exit 1

echo "Installing requirements"

cd ..

uv pip install -r requirements_whisper.txt --index-strategy unsafe-best-match --build-constraints uv_build_constraints.txt || exit 1

uv pip install git+https://github.com/NVIDIA/NeMo.git --index-strategy unsafe-best-match --build-constraints uv_build_constraints.txt || exit 1

python DownloadModels.py || exit 1

# Show completion message
echo "Virtual environment made and installed properly"

# Keep the terminal open
read -p "Press Enter to continue..."
