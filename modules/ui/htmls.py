"""Shared UI styling: theme, per-button colours, page CSS and client-side helpers.

The look follows the IndexTTS Premium app: the stock Gradio ``Origin`` theme,
dark by default with an instant light/dark switch, and every action button in
its own hue so a page never reads as a wall of identical controls.  Everything
except the button palette is written against the theme's CSS variables, so the
light and dark modes stay in sync automatically.
"""

import gradio as gr

__all__ = [
    "BUTTON_COLORS",
    "BUTTON_ICONS",
    "CSS",
    "HEAD",
    "NLLB_VRAM_TABLE",
    "TOGGLE_SECTIONS_JS",
    "TOGGLE_THEME_JS",
    "app_theme",
    "btn",
]


def app_theme(name=None):
    """Return the stock Origin theme, or the Gradio theme named on the command line."""
    if not name or str(name).strip().lower() == "origin":
        return gr.themes.Origin()
    return name


# Every action button gets its own hue as a (deep, mid, bright) triple.  The
# gradient runs deep -> mid -> bright so the face has depth, the glow is built
# from the mid tone, and white text stays readable on all three stops.
BUTTON_HUES = {
    "emerald": ("#065f46", "#059669", "#34d399"),
    "green":   ("#166534", "#16a34a", "#4ade80"),
    "lime":    ("#3f6212", "#65a30d", "#a3e635"),
    "teal":    ("#115e59", "#0d9488", "#2dd4bf"),
    "cyan":    ("#155e75", "#0891b2", "#22d3ee"),
    "sky":     ("#075985", "#0284c7", "#38bdf8"),
    "blue":    ("#1e40af", "#2563eb", "#60a5fa"),
    "indigo":  ("#3730a3", "#4f46e5", "#818cf8"),
    "violet":  ("#5b21b6", "#7c3aed", "#a78bfa"),
    "purple":  ("#6b21a8", "#9333ea", "#c084fc"),
    "fuchsia": ("#86198f", "#c026d3", "#e879f9"),
    "pink":    ("#9d174d", "#db2777", "#f9a8d4"),
    "rose":    ("#9f1239", "#e11d48", "#fda4af"),
    "red":     ("#991b1b", "#dc2626", "#f87171"),
    "orange":  ("#9a3412", "#ea580c", "#fb923c"),
    "amber":   ("#92400e", "#d97706", "#fbbf24"),
    "gold":    ("#854d0e", "#ca8a04", "#fde047"),
    "bronze":  ("#5c3a21", "#8b5a2b", "#d4a373"),
    "coral":   ("#9a3b2e", "#e0573e", "#ffa08a"),
    "slate":   ("#334155", "#475569", "#94a3b8"),
    "gray":    ("#3f3f46", "#52525b", "#a1a1aa"),
}
BUTTON_COLORS = tuple(BUTTON_HUES)

# Button icons are drawn by CSS instead of being written into the label, so the
# translated labels in configs/translation.yaml keep matching their keys.
# Values are CSS escapes: U+FE0F asks for the colour emoji presentation.
BUTTON_ICONS = {
    "save": r"\1F4BE",
    "reset": r"\21BA",
    "delete": r"\1F5D1\FE0F",
    "download": r"\2B07\FE0F",
    "folder": r"\1F4C2",
    "load": r"\1F4E5",
    "generate": r"\1F4DD",
    "cancel": r"\26D4",
    "translate": r"\1F310",
    "music": r"\1F3B5",
    "sections": r"\21D5",
    "theme": r"\1F317",
}


def btn(color, *extra, icon=None):
    """Build the ``elem_classes`` list for a coloured action button.

    ``color`` picks one of :data:`BUTTON_COLORS` and ``icon`` one of
    :data:`BUTTON_ICONS`.  Every button has the same height and type size; only
    the hue changes.
    """
    if color not in BUTTON_HUES:
        raise ValueError(f"Unknown button colour: {color}")
    classes = ["ax", f"ax-{color}", *extra]
    if icon is not None:
        if icon not in BUTTON_ICONS:
            raise ValueError(f"Unknown button icon: {icon}")
        classes.append(f"ax-icon-{icon}")
    return classes


def _rgb(value):
    digits = value.lstrip("#")
    return tuple(int(digits[index: index + 2], 16) for index in (0, 2, 4))


def _button_palette_css():
    """Generate the gradient, border and glow rules for every button hue."""
    rules = []
    for name, (deep, mid, bright) in BUTTON_HUES.items():
        red, green, blue = _rgb(mid)
        bright_rgb = ", ".join(str(part) for part in _rgb(bright))
        rules.append(
            f"""
button.ax-{name} {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 55%, {bright} 100%) !important;
  border-color: rgba({bright_rgb}, .72) !important;
  box-shadow: 0 8px 20px rgba({red}, {green}, {blue}, .30), inset 0 1px 0 rgba(255, 255, 255, .20) !important;
}}
button.ax-{name}:hover:not(:disabled) {{
  border-color: rgba({bright_rgb}, .98) !important;
  box-shadow: 0 12px 26px rgba({red}, {green}, {blue}, .44), inset 0 1px 0 rgba(255, 255, 255, .28) !important;
}}
body:not(.dark) button.ax-{name} {{
  background: linear-gradient(135deg, {deep} 0%, {mid} 66%, {bright} 100%) !important;
  border-color: {mid} !important;
  box-shadow: 0 7px 17px rgba({red}, {green}, {blue}, .26), inset 0 1px 0 rgba(255, 255, 255, .26) !important;
}}
body:not(.dark) button.ax-{name}:hover:not(:disabled) {{
  box-shadow: 0 11px 24px rgba({red}, {green}, {blue}, .38), inset 0 1px 0 rgba(255, 255, 255, .32) !important;
}}"""
        )
    return "\n".join(rules)


def _button_icon_css():
    return "\n".join(
        f'button.ax-icon-{name}::before {{ content: "{glyph}"; }}'
        for name, glyph in BUTTON_ICONS.items()
    )


_BASE_CSS = r"""
/* --------------------------------------------------------------------- *
 * Accent tokens for everything that is not a button.  Only these two
 * blocks know about light and dark; the rest of the sheet reads them.
 * --------------------------------------------------------------------- */
:root {
  --ax-ok: #047857;
  --ax-warn: #b45309;
  --ax-error: #be123c;
  --ax-accent: #0f766e;
}
:root.dark, :root .dark {
  --ax-ok: #10b981;
  --ax-warn: #f59e0b;
  --ax-error: #fb7185;
  --ax-accent: #14b8a6;
}

/* --------------------------------------------------------------------- *
 * Action buttons.  Every one is the same height, weight and type size so
 * rows of controls share a baseline; only the hue changes, and the hue is
 * generated per colour further down.  Motion is limited to hover and press:
 * nothing animates while the page is idle, so long jobs stay smooth.
 * --------------------------------------------------------------------- */
button.ax {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--size-2);
  min-height: 44px;
  padding: var(--size-2) var(--size-4) !important;
  border-width: 1px !important;
  border-style: solid !important;
  border-radius: var(--radius-lg) !important;
  color: #f8fafc !important;
  font-size: var(--text-md) !important;
  font-weight: 650 !important;
  line-height: 1.25 !important;
  text-align: center;
  text-shadow: 0 1px 2px rgba(2, 6, 23, .45) !important;
  transition: transform 140ms ease, filter 140ms ease, box-shadow 140ms ease;
}
button.ax:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.06); }
button.ax:active:not(:disabled) { transform: translateY(1px); filter: brightness(.96); }
button.ax:focus-visible { outline: 2px solid #bae6fd; outline-offset: 2px; }
button.ax:disabled { filter: grayscale(.45) opacity(.62); transform: none; box-shadow: none !important; cursor: not-allowed; }
button.ax.ax-lg { min-height: 54px; font-size: var(--text-lg) !important; letter-spacing: .02em; }
button.ax[class*="ax-icon-"]::before { font-size: 1.05em; line-height: 1; text-shadow: none; }
/* Inside a row a button would otherwise stretch to the tallest neighbour. */
.row > button.ax { align-self: center; }

/* --------------------------------------------------------------------- *
 * Page furniture
 * --------------------------------------------------------------------- */
.app-header {
  align-items: center;
  flex-wrap: nowrap;
  gap: var(--size-4);
  padding-bottom: var(--size-3);
  border-bottom: 1px solid var(--border-color-primary);
}
.app-header > :first-child { flex: 1 1 auto; min-width: 0; }
.app-header h1 { margin: 0 !important; line-height: 1.2; }
.app-header p { margin: var(--size-1) 0 0 !important; color: var(--body-text-color-subdued); }
.app-header a { text-decoration: none; }
.app-header a:hover { text-decoration: underline; }
/* Gradio's own row rule is scoped, so the header strip has to out-specify it. */
.row.header-actions {
  flex: 0 0 auto !important;
  width: auto !important;
  min-width: 0 !important;
  flex-wrap: nowrap;
  justify-content: flex-end;
  gap: var(--size-2);
}
.header-actions button.ax { flex: 0 0 auto; white-space: nowrap; }
/* A field with its action button: line the button up with the input box, not
   with the label above it. */
.row.input-action-row { align-items: flex-end; }
.row.input-action-row > button.ax { align-self: flex-end !important; margin-bottom: 11px; }
.preset-status { min-height: var(--size-6); font-size: var(--text-sm); color: var(--body-text-color-subdued); }
.side-actions { gap: var(--size-3) !important; }
/* Help text inside a gr.Group: without a block background the group's border
   colour shows through and the panel reads as a lighter slab. */
.column.group-panel {
  padding: var(--block-padding) !important;
  background: var(--block-background-fill);
}
.column.group-panel .prose { color: var(--body-text-color); }

/* --------------------------------------------------------------------- *
 * Uploaded media preview.  The browser plays the uploaded file itself; no
 * copy or conversion is made, so a large MKV shows up immediately.
 * --------------------------------------------------------------------- */
.upload-preview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
  gap: var(--size-3);
  margin-top: var(--size-2);
}
.upload-preview-card {
  min-width: 0;
  padding: var(--size-3);
  border: 1px solid var(--border-color-primary);
  border-left: 4px solid var(--ax-accent);
  border-radius: var(--radius-lg);
  background: var(--background-fill-secondary);
}
.upload-preview-card.is-audio { border-left-color: #8b5cf6; }
.upload-preview-meta {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: var(--size-3);
  margin-bottom: var(--size-2);
  flex-wrap: wrap;
}
.upload-preview-name { font-weight: 700; color: var(--body-text-color); overflow-wrap: anywhere; }
.upload-preview-type {
  font-size: var(--text-sm);
  font-weight: 600;
  color: var(--body-text-color-subdued);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.upload-preview-card video,
.upload-preview-card audio { display: block; width: 100%; }
.upload-preview-card video { max-height: 320px; border-radius: var(--radius-md); background: #000; }
.upload-preview-note { margin-top: var(--size-2); font-size: var(--text-sm); color: var(--ax-warn); }
.upload-preview-card.preview-unavailable video,
.upload-preview-card.preview-unavailable audio { display: none; }

/* --------------------------------------------------------------------- *
 * Transcript boxes and microphone recorders
 * --------------------------------------------------------------------- */
.live-transcription-box textarea {
  min-height: 180px !important;
  height: 180px !important;
  max-height: 180px !important;
  overflow-y: auto !important;
  resize: none !important;
}

.mic-recorder-frame {
  position: relative;
  overflow: hidden;
  min-height: 320px;
  padding: 18px !important;
  border: 2px solid var(--border-color-accent) !important;
  border-radius: 28px !important;
  background:
    radial-gradient(circle at top right, var(--color-accent-soft), transparent 38%),
    linear-gradient(180deg, var(--block-background-fill), var(--background-fill-secondary)) !important;
  box-shadow: var(--shadow-drop-lg);
}

.mic-recorder-frame::after {
  position: absolute;
  top: 14px;
  right: 14px;
  z-index: 2;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid var(--border-color-accent);
  background: var(--color-accent-soft);
  color: var(--color-accent);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.08em;
  box-shadow: var(--shadow-drop);
}

#live-mic-recorder::after { content: "LIVE MIC"; }
#record-mic-recorder::after { content: "RECORD THEN GENERATE"; }

.mic-recorder-frame > .wrap,
.mic-recorder-frame .full-container,
.mic-recorder-frame .input-container,
.mic-recorder-frame .input-wrapper,
.mic-recorder-frame .component-wrapper {
  min-height: 250px !important;
}

.mic-recorder-frame .input-wrapper,
.mic-recorder-frame .recording-overlay {
  border-radius: 22px !important;
  background: var(--block-background-fill) !important;
}

.mic-recorder-frame .recording-content,
.mic-recorder-frame .minimal-audio-recorder,
.mic-recorder-frame .minimal-audio-player {
  min-height: 180px !important;
}

.mic-recorder-frame [data-testid="microphone-waveform"],
.mic-recorder-frame [data-testid="recording-waveform"],
.mic-recorder-frame .microphone,
.mic-recorder-frame .waveform-wrapper {
  min-height: 140px !important;
}

.mic-recorder-frame .waveform-wrapper {
  border-radius: 18px !important;
  background: var(--background-fill-secondary) !important;
  padding: 10px !important;
  border: 1px solid var(--border-color-primary) !important;
}

.mic-recorder-frame .record-button,
.mic-recorder-frame .stop-button,
.mic-recorder-frame .stop-button-paused,
.mic-recorder-frame .pause-button,
.mic-recorder-frame .resume-button,
.mic-recorder-frame .duration,
.mic-recorder-frame .timestamp,
.mic-recorder-frame .mic-select,
.mic-recorder-frame .device-select-large {
  min-height: 72px !important;
  font-size: 18px !important;
  font-weight: 700 !important;
}

.mic-recorder-frame .record-button,
.mic-recorder-frame .stop-button,
.mic-recorder-frame .stop-button-paused,
.mic-recorder-frame .pause-button,
.mic-recorder-frame .resume-button {
  min-width: 160px !important;
  padding: 0 20px !important;
  border-radius: 18px !important;
}

.mic-recorder-frame .record-button {
  border: 2px solid var(--border-color-accent) !important;
  background: var(--block-background-fill) !important;
}

.mic-recorder-frame .record-button::before,
.mic-recorder-frame .stop-button::before,
.mic-recorder-frame .stop-button-paused::before {
  height: 18px !important;
  width: 18px !important;
  margin-right: 14px !important;
}

.mic-recorder-frame .stop-button,
.mic-recorder-frame .stop-button-paused {
  border: 2px solid var(--border-color-primary) !important;
  background: var(--button-secondary-background-fill) !important;
}

.mic-recorder-frame .pause-button,
.mic-recorder-frame .resume-button,
.mic-recorder-frame .duration,
.mic-recorder-frame .timestamp,
.mic-recorder-frame .mic-select,
.mic-recorder-frame .device-select-large {
  border: 1px solid var(--border-color-primary) !important;
  background: var(--background-fill-secondary) !important;
  color: var(--body-text-color) !important;
}

.mic-recorder-frame .mic-select,
.mic-recorder-frame .device-select-large {
  min-width: 240px !important;
  max-width: min(100%, 420px) !important;
}

.mic-recorder-frame .recording-overlay {
  border: 2px solid var(--border-color-accent) !important;
}

/* NLLB VRAM table: scoped, so it cannot restyle the file lists or any other table. */
#md_nllb_vram_table table { width: 100%; border-collapse: collapse; }
#md_nllb_vram_table th,
#md_nllb_vram_table td { padding: var(--size-2); border: 1px solid var(--border-color-primary); text-align: left; }
#md_nllb_vram_table th { background: var(--background-fill-secondary); }
#md_nllb_vram_table summary { cursor: pointer; font-weight: 600; }

@media (max-width: 900px) {
  .app-header { flex-wrap: wrap; }
  .row.header-actions {
    flex: 1 1 100% !important;
    width: 100% !important;
    flex-wrap: wrap;
    justify-content: flex-start;
  }
  .header-actions button.ax {
    flex: 1 1 180px;
    width: auto !important;
    min-width: 0 !important;
    white-space: normal;
  }

  .mic-recorder-frame {
    min-height: 280px;
  }

  .mic-recorder-frame .record-button,
  .mic-recorder-frame .stop-button,
  .mic-recorder-frame .stop-button-paused,
  .mic-recorder-frame .pause-button,
  .mic-recorder-frame .resume-button,
  .mic-recorder-frame .duration,
  .mic-recorder-frame .timestamp,
  .mic-recorder-frame .mic-select,
  .mic-recorder-frame .device-select-large {
    min-height: 64px !important;
    min-width: 132px !important;
    font-size: 16px !important;
  }
}

@media (prefers-reduced-motion: reduce) {
  button.ax { transition: none; }
  button.ax:hover:not(:disabled), button.ax:active:not(:disabled) { transform: none; }
}
"""

CSS = _BASE_CSS + _button_palette_css() + "\n" + _button_icon_css() + "\n"


# Dark is the default the first time the app is opened.  The choice is stored in
# localStorage and mirrored into the ``__theme`` query parameter so a reload or a
# bookmark restores it before Gradio paints the page.
_THEME_HEAD = """
<meta name="color-scheme" content="dark light">
<script>
(function () {
  var KEY = "whisperwebui.theme";
  function resolve() {
    var url = new URL(window.location.href);
    var param = url.searchParams.get("__theme");
    var stored = null;
    try { stored = window.localStorage.getItem(KEY); } catch (e) {}
    var mode = param || stored || "dark";
    if (mode !== "light") { mode = "dark"; }
    try { window.localStorage.setItem(KEY, mode); } catch (e) {}
    if (param !== mode) {
      url.searchParams.set("__theme", mode);
      window.history.replaceState(null, "", url.toString());
    }
    return mode;
  }
  function paint(mode) {
    if (!document.body) { return false; }
    document.body.classList.toggle("dark", mode === "dark");
    return true;
  }
  var mode = resolve();
  if (!paint(mode)) {
    document.addEventListener("DOMContentLoaded", function () { paint(mode); });
  }
})();
</script>
"""

_HELPERS_HEAD = """
<script>
(() => {
  const transcriptionSelector = ".live-transcription-box textarea";
  const micSelectSelector = '.mic-recorder-frame select[aria-label="Select input device"]';
  const previewCardSelector = ".upload-preview-card";
  let syncingMicDevices = false;
  let patchedGetUserMedia = false;

  // Transcript boxes follow new text while the reader is at the bottom.  Only a
  // change of the text length touches layout, so an idle page does no work, and
  // scrolling up to read earlier lines is not undone by the next segment.
  const followState = new WeakMap();

  const trackTranscript = (textarea) => {
    let state = followState.get(textarea);
    if (!state) {
      state = { length: -1, follow: true };
      followState.set(textarea, state);
      textarea.addEventListener("scroll", () => {
        state.follow = textarea.scrollHeight - textarea.scrollTop - textarea.clientHeight < 32;
      }, { passive: true });
    }
    return state;
  };

  const followTranscripts = () => {
    document.querySelectorAll(transcriptionSelector).forEach((textarea) => {
      const state = trackTranscript(textarea);
      const length = textarea.value.length;
      if (length === state.length) return;
      if (length < state.length) state.follow = true;
      state.length = length;
      if (state.follow) textarea.scrollTop = textarea.scrollHeight;
    });
  };

  const getMicLabel = (device, index) => {
    const label = (device?.label || "").trim();
    if (label) return label;
    return index === 0 ? "Browser default microphone" : `Microphone ${index + 1}`;
  };

  const syncMicDeviceSelects = async () => {
    if (syncingMicDevices) return;
    if (!document.querySelector(micSelectSelector)) return;
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.enumerateDevices !== "function") return;

    syncingMicDevices = true;

    try {
      const devices = (await navigator.mediaDevices.enumerateDevices())
        .filter((device) => device.kind === "audioinput" && device.deviceId);

      if (!devices.length) return;

      document.querySelectorAll(micSelectSelector).forEach((select) => {
        const currentValue = select.value;
        const currentText = select.options[select.selectedIndex]?.textContent || "";
        const existingOptions = Array.from(select.options);
        const needsRefresh =
          existingOptions.length !== devices.length ||
          /no microphone/i.test(currentText) ||
          existingOptions.some((option, index) => {
            const device = devices[index];
            return !device || option.value !== device.deviceId || option.textContent !== getMicLabel(device, index);
          });

        if (!needsRefresh) return;

        select.innerHTML = "";

        devices.forEach((device, index) => {
          const option = document.createElement("option");
          option.value = device.deviceId;
          option.textContent = getMicLabel(device, index);
          select.appendChild(option);
        });

        const nextValue = devices.some((device) => device.deviceId === currentValue)
          ? currentValue
          : devices[0].deviceId;

        if (nextValue) {
          select.value = nextValue;
        }

        select.disabled = false;
        select.dispatchEvent(new Event("input", { bubbles: true }));
        select.dispatchEvent(new Event("change", { bubbles: true }));
      });
    } catch (error) {
      // Keep the built-in fallback text if device enumeration still fails.
    } finally {
      syncingMicDevices = false;
    }
  };

  const scheduleMicRefresh = () => {
    window.setTimeout(syncMicDeviceSelects, 150);
    window.setTimeout(syncMicDeviceSelects, 1000);
    window.setTimeout(syncMicDeviceSelects, 2500);
  };

  const isVisibleElement = (element) => {
    if (!(element instanceof Element)) return false;
    return !!(element.offsetWidth || element.offsetHeight || element.getClientRects().length);
  };

  const readPreferredMicDeviceId = (root) => {
    if (root instanceof Element) {
      const scopedSelect = root.querySelector(micSelectSelector);
      if (scopedSelect instanceof HTMLSelectElement && scopedSelect.value) {
        return scopedSelect.value;
      }
    }

    const visibleSelects = Array.from(document.querySelectorAll(micSelectSelector))
      .filter((select) => isVisibleElement(select));

    for (const select of visibleSelects) {
      if (select instanceof HTMLSelectElement && select.value) {
        return select.value;
      }
    }

    return "";
  };

  const setPreferredMicDeviceId = (deviceId) => {
    window.__preferredMicDeviceId = deviceId || "";
  };

  const buildMicConstraints = (constraints, preferredDeviceId) => {
    if (!preferredDeviceId) return constraints;

    const original = constraints && typeof constraints === "object" ? constraints : {};
    const next = { ...original };
    const originalAudio = next.audio;

    if (originalAudio && typeof originalAudio === "object" && !Array.isArray(originalAudio)) {
      next.audio = {
        ...originalAudio,
        deviceId: { exact: preferredDeviceId },
      };
    } else {
      next.audio = { deviceId: { exact: preferredDeviceId } };
    }

    return next;
  };

  const ensurePreferredMicPatch = () => {
    if (patchedGetUserMedia) return;
    if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") return;

    const originalGetUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);

    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const preferredDeviceId = window.__preferredMicDeviceId || readPreferredMicDeviceId();
      const nextConstraints = buildMicConstraints(constraints, preferredDeviceId);
      window.__lastPreferredMicDeviceId = preferredDeviceId || "";
      window.__lastPreferredMicConstraints = nextConstraints;

      try {
        return await originalGetUserMedia(nextConstraints);
      } catch (error) {
        if (
          preferredDeviceId &&
          nextConstraints !== constraints &&
          error &&
          (error.name === "OverconstrainedError" || error.name === "NotFoundError")
        ) {
          return originalGetUserMedia(constraints);
        }
        throw error;
      }
    };

    patchedGetUserMedia = true;
  };

  // The upload preview plays the original file.  When the browser cannot open
  // the container (AVI, WMV, FLV...) or decode the video codec, say so on the
  // card instead of leaving an empty player.
  const notePreview = (media, message, hideMedia) => {
    const card = media.closest(previewCardSelector);
    if (!card) return;
    let note = card.querySelector(".upload-preview-note");
    if (!note) {
      note = document.createElement("div");
      note.className = "upload-preview-note";
      card.appendChild(note);
    }
    note.textContent = message;
    card.classList.toggle("preview-unavailable", Boolean(hideMedia));
  };

  const watchPreviewMedia = () => {
    document.addEventListener("error", (event) => {
      const media = event.target;
      if (!(media instanceof HTMLMediaElement) || !media.closest(previewCardSelector)) return;
      notePreview(
        media,
        "This browser cannot play this file format, so there is no preview. Transcription is not affected.",
        true,
      );
    }, true);

    document.addEventListener("loadedmetadata", (event) => {
      const media = event.target;
      if (!(media instanceof HTMLVideoElement) || !media.closest(previewCardSelector)) return;
      if (!media.videoWidth) {
        notePreview(
          media,
          "This browser cannot decode the video codec, so only the audio plays here. Transcription is not affected.",
          false,
        );
      }
    }, true);
  };

  const init = () => {
    ensurePreferredMicPatch();
    watchPreviewMedia();
    followTranscripts();
    syncMicDeviceSelects();
    window.setInterval(followTranscripts, 250);
    window.setInterval(syncMicDeviceSelects, 2000);

    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest(".mic-recorder-frame .record-button")) {
        setPreferredMicDeviceId(readPreferredMicDeviceId(target.closest(".mic-recorder-frame")));
        scheduleMicRefresh();
      }
    });

    document.addEventListener("change", (event) => {
      const target = event.target;
      if (target instanceof HTMLSelectElement && target.matches(micSelectSelector)) {
        setPreferredMicDeviceId(target.value);
      }
    });

    if (navigator.mediaDevices && typeof navigator.mediaDevices.addEventListener === "function") {
      navigator.mediaDevices.addEventListener("devicechange", syncMicDeviceSelects);
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
</script>
"""

HEAD = _THEME_HEAD + _HELPERS_HEAD


# Switching themes only swaps the ``dark`` class Gradio itself keys off, so it is
# instant: no reload and no round trip to the server.
TOGGLE_THEME_JS = """
() => {
  const dark = !document.body.classList.contains("dark");
  document.body.classList.toggle("dark", dark);
  const mode = dark ? "dark" : "light";
  try { window.localStorage.setItem("whisperwebui.theme", mode); } catch (e) {}
  const url = new URL(window.location.href);
  url.searchParams.set("__theme", mode);
  window.history.replaceState(null, "", url.toString());
}
"""


# Expand or collapse every accordion on the tab that is currently on screen.
TOGGLE_SECTIONS_JS = """
async () => {
  const visible = (element) =>
    Boolean(element && (element.offsetWidth || element.offsetHeight || element.getClientRects().length));
  const tabs = document.querySelector("#main-tabs");
  const panel = tabs
    ? Array.from(tabs.querySelectorAll(":scope > .tabitem")).find(visible)
    : null;
  const scope = panel || document;
  const heads = () => Array.from(scope.querySelectorAll("button.label-wrap")).filter(visible);
  const first = heads();
  if (!first.length) { return; }
  const expand = first.some((head) => !head.classList.contains("open"));
  // A nested accordion only reaches the DOM once its parent is open, so keep
  // sweeping until nothing is left in the wrong state.
  for (let pass = 0; pass < 6; pass++) {
    const pending = heads().filter(
      (head) => head.classList.contains("open") !== expand
    );
    if (!pending.length) { break; }
    pending.forEach((head) => head.click());
    await new Promise((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(resolve))
    );
  }
}
"""


NLLB_VRAM_TABLE = """
<details>
  <summary>VRAM usage for each model</summary>
  <table>
    <thead>
      <tr>
        <th>Model name</th>
        <th>Required VRAM</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>nllb-200-3.3B</td>
        <td>~16GB</td>
      </tr>
      <tr>
        <td>nllb-200-1.3B</td>
        <td>~8GB</td>
      </tr>
      <tr>
        <td>nllb-200-distilled-600M</td>
        <td>~4GB</td>
      </tr>
    </tbody>
  </table>
  <p><strong>Note:</strong> Be mindful of your VRAM! The table above provides an approximate VRAM usage for each model.</p>
</details>
"""
