#!/bin/bash

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
if [ ! -x venv/bin/python ]; then
    echo "Virtual environment not found. Run the installer first." >&2
    exit 1
fi
source venv/bin/activate || exit 1
exec python app.py "$@"
