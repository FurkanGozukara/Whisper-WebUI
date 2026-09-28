# Installer archive source files

These files are maintained copies of the installer bundle distributed alongside
Whisper-WebUI. They include the Windows/Linux launcher and installer fixes from
the verification work, plus the offline model downloader. The dependency files
(`requirements_whisper.txt`, `uv_build_constraints.txt`) come with the installer
download and are not kept in this repository.

For a release archive, copy this directory's files one level above the
`Whisper-WebUI` checkout, preserving this layout:

```text
release/
  Windows_Install_Update.bat
  Windows_Start_app.bat
  Runpod_Install_Whisper.sh
  Massed_Compute_Install.sh
  DownloadModels.py
  requirements_whisper.txt      (from the installer download)
  uv_build_constraints.txt      (from the installer download)
  Whisper-WebUI/
    Install.bat
    Install.sh
    app.py
```

The outer installers can create the `Whisper-WebUI` checkout when it is absent.
Run them from the assembled release directory. The repository's `Install.bat`
and `Install.sh` intentionally consume the adjacent distribution dependency
files. Keep these maintained copies synchronized when preparing future archives.

Shell syntax and relocated launch paths were checked on Linux. A fresh installer
run and native Windows execution remain unverified; see
[compatibility verification](../docs/compatibility-verification.md).
