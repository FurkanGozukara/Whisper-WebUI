CSS = """
/* Readable font for all text */
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', Arial, sans-serif !important;
}

/* Layout spacing */
.markdown {
    margin-bottom: 0;
    padding-bottom: 0;
}
.tabs {
    margin-top: 0;
    padding-top: 0;
}

/* Button styling */
.bmc-button {
    padding: 2px 5px;
    border-radius: 5px;
    background-color: #FF813F;
    color: white;
    box-shadow: 0px 1px 2px rgba(0, 0, 0, 0.3);
    text-decoration: none;
    display: inline-block;
    font-size: 20px;
    margin: 2px;
    cursor: pointer;
    -webkit-transition: background-color 0.3s ease;
    -ms-transition: background-color 0.3s ease;
    transition: background-color 0.3s ease;
}
.bmc-button:hover,
.bmc-button:active,
.bmc-button:focus {
    background-color: #FF5633;
}

/* Project link styling */
#md_project a {
  text-decoration: none;
}
#md_project a:hover {
  text-decoration: underline;
}

.upload-preview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 14px;
    margin-top: 10px;
}

.upload-preview-card {
    padding: 14px;
    border-radius: 18px;
    border: 1px solid rgba(15, 23, 42, 0.08);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.95));
    box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
}

.upload-preview-meta {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    margin-bottom: 10px;
    flex-wrap: wrap;
}

.upload-preview-name {
    font-weight: 700;
    color: #0f172a;
    word-break: break-word;
}

.upload-preview-type {
    font-size: 12px;
    font-weight: 600;
    color: #475569;
}

.upload-preview-card video,
.upload-preview-card audio {
    width: 100%;
    display: block;
}

.upload-preview-card video {
    max-height: 300px;
    border-radius: 14px;
    background: #020617;
}

#top-download-output-button {
    min-height: 0 !important;
}

:is(button.download-output-button, .download-output-button button) {
    color: #fff7d1 !important;
    text-shadow:
        0 0 10px rgba(255, 243, 176, 0.45),
        0 1px 0 rgba(107, 71, 10, 0.92);
    background:
        radial-gradient(circle at 18% 0%, rgba(255, 245, 180, 0.26), transparent 34%),
        linear-gradient(180deg, #ffe78a 0%, #f7c94a 18%, #d89b11 58%, #8f6208 100%) !important;
    border-color: rgba(141, 98, 8, 0.72) !important;
    box-shadow:
        0 0 0 1px rgba(255, 232, 147, 0.14),
        0 14px 30px rgba(216, 155, 17, 0.28),
        0 0 26px rgba(247, 201, 74, 0.18),
        0 1px 0 rgba(255, 250, 219, 0.4) inset,
        0 -3px 0 rgba(115, 75, 7, 0.3) inset !important;
    animation: premium-download-glow 2.8s ease-in-out infinite;
}

:is(button.download-output-button, .download-output-button button):hover {
    box-shadow:
        0 0 0 1px rgba(255, 240, 171, 0.18),
        0 18px 36px rgba(216, 155, 17, 0.34),
        0 0 34px rgba(247, 201, 74, 0.24),
        0 1px 0 rgba(255, 252, 230, 0.44) inset,
        0 -3px 0 rgba(115, 75, 7, 0.34) inset !important;
}

@keyframes premium-download-glow {
    0%, 100% {
        box-shadow:
            0 14px 30px rgba(216, 155, 17, 0.28),
            0 0 26px rgba(247, 201, 74, 0.18),
            0 1px 0 rgba(255, 250, 219, 0.4) inset,
            0 -3px 0 rgba(115, 75, 7, 0.3) inset;
    }
    50% {
        box-shadow:
            0 18px 36px rgba(216, 155, 17, 0.34),
            0 0 34px rgba(247, 201, 74, 0.24),
            0 1px 0 rgba(255, 252, 230, 0.44) inset,
            0 -3px 0 rgba(115, 75, 7, 0.34) inset;
    }
}

:is(button.action-button, .action-button button) {
    position: relative;
    overflow: hidden;
    min-height: 52px;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    color: #fdf8ff !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
    text-shadow: 0 1px 0 rgba(15, 23, 42, 0.28);
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}

:is(button.action-button, .action-button button)::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 15%, rgba(255, 255, 255, 0.28) 36%, transparent 58%);
    transform: translateX(-160%);
    transition: transform 0.55s ease;
    pointer-events: none;
}

:is(button.action-button, .action-button button):hover {
    transform: translateY(-1px);
    filter: saturate(1.06) brightness(1.03);
}

:is(button.action-button, .action-button button):hover::before {
    transform: translateX(160%);
}

:is(button.action-button, .action-button button):active {
    transform: translateY(1px);
}

:is(button.action-button, .action-button button):focus-visible {
    outline: 2px solid rgba(255, 255, 255, 0.82);
    outline-offset: 2px;
}

:is(button.generate-subtitle-button, .generate-subtitle-button button) {
    color: #ffe7eb !important;
    text-shadow:
        0 0 8px rgba(255, 222, 228, 0.65),
        0 0 18px rgba(255, 131, 157, 0.48),
        0 1px 0 rgba(107, 13, 35, 0.9);
    background:
        radial-gradient(circle at 18% 0%, rgba(255, 197, 210, 0.36), transparent 34%),
        linear-gradient(180deg, #ff9aae 0%, #ff6383 16%, #d91f4d 55%, #7f102c 100%) !important;
    border-color: rgba(135, 17, 48, 0.72) !important;
    box-shadow:
        0 0 0 1px rgba(255, 180, 196, 0.12),
        0 16px 34px rgba(217, 31, 77, 0.34),
        0 0 26px rgba(255, 77, 116, 0.26),
        0 1px 0 rgba(255, 234, 239, 0.34) inset,
        0 -3px 0 rgba(95, 9, 31, 0.34) inset !important;
    animation: premium-button-glow 2.8s ease-in-out infinite;
}

:is(button.generate-subtitle-button, .generate-subtitle-button button)::before {
    animation: premium-button-sheen 3.6s ease-in-out infinite;
}

:is(button.generate-subtitle-button, .generate-subtitle-button button):hover {
    box-shadow:
        0 0 0 1px rgba(255, 188, 202, 0.18),
        0 20px 38px rgba(217, 31, 77, 0.42),
        0 0 34px rgba(255, 77, 116, 0.34),
        0 1px 0 rgba(255, 238, 242, 0.38) inset,
        0 -3px 0 rgba(95, 9, 31, 0.38) inset !important;
}

:is(button.generate-subtitle-button, .generate-subtitle-button button):active {
    animation-play-state: paused;
}

:is(button.open-outputs-folder-button, .open-outputs-folder-button button) {
    color: #edfff6 !important;
    text-shadow:
        0 0 8px rgba(220, 255, 242, 0.42),
        0 1px 0 rgba(6, 72, 52, 0.85);
    background:
        radial-gradient(circle at 18% 0%, rgba(214, 255, 241, 0.22), transparent 34%),
        linear-gradient(180deg, #a7f3d6 0%, #58d8ab 18%, #10946f 58%, #0a5c48 100%) !important;
    border-color: rgba(8, 100, 82, 0.68) !important;
    box-shadow:
        0 0 0 1px rgba(183, 255, 226, 0.1),
        0 14px 28px rgba(15, 159, 127, 0.24),
        0 1px 0 rgba(229, 255, 248, 0.34) inset,
        0 -3px 0 rgba(5, 73, 60, 0.28) inset !important;
}

:is(button.open-outputs-folder-button, .open-outputs-folder-button button):hover {
    box-shadow:
        0 0 0 1px rgba(198, 255, 231, 0.14),
        0 18px 34px rgba(15, 159, 127, 0.3),
        0 1px 0 rgba(236, 255, 250, 0.38) inset,
        0 -3px 0 rgba(5, 73, 60, 0.32) inset !important;
}

@keyframes premium-button-glow {
    0%, 100% {
        box-shadow:
            0 16px 34px rgba(217, 31, 77, 0.34),
            0 0 26px rgba(255, 77, 116, 0.26),
            0 1px 0 rgba(255, 234, 239, 0.34) inset,
            0 -3px 0 rgba(95, 9, 31, 0.34) inset;
    }
    50% {
        box-shadow:
            0 20px 40px rgba(217, 31, 77, 0.42),
            0 0 34px rgba(255, 77, 116, 0.34),
            0 1px 0 rgba(255, 238, 242, 0.38) inset,
            0 -3px 0 rgba(95, 9, 31, 0.38) inset;
    }
}

@keyframes premium-button-sheen {
    0%, 100% {
        transform: translateX(-150%);
    }
    45%, 55% {
        transform: translateX(150%);
    }
}

@media (prefers-reduced-motion: reduce) {
    :is(button.generate-subtitle-button, .generate-subtitle-button button),
    :is(button.generate-subtitle-button, .generate-subtitle-button button)::before,
    :is(button.download-output-button, .download-output-button button) {
        animation: none !important;
    }
}

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

#live-mic-recorder::after {
    content: "LIVE MIC";
}

#record-mic-recorder::after {
    content: "RECORD THEN GENERATE";
}

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

@media (max-width: 900px) {
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
"""

HEAD = """
<script>
(() => {
  const transcriptionSelector = ".live-transcription-box textarea";
  const micSelectSelector = '.mic-recorder-frame select[aria-label="Select input device"]';
  let syncingMicDevices = false;
  let patchedGetUserMedia = false;

  const scrollToBottom = (textarea) => {
    if (!textarea) return;
    textarea.scrollTop = textarea.scrollHeight;
  };

  const syncAll = () => {
    document.querySelectorAll(transcriptionSelector).forEach(scrollToBottom);
  };

  const getMicLabel = (device, index) => {
    const label = (device?.label || "").trim();
    if (label) return label;
    return index === 0 ? "Browser default microphone" : `Microphone ${index + 1}`;
  };

  const syncMicDeviceSelects = async () => {
    if (syncingMicDevices) return;
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
    } else if (originalAudio) {
      next.audio = { deviceId: { exact: preferredDeviceId } };
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

  const init = () => {
    ensurePreferredMicPatch();
    syncAll();
    syncMicDeviceSelects();
    window.setInterval(syncAll, 200);
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

NLLB_VRAM_TABLE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    table {
      border-collapse: collapse;
      width: 100%;
    }
    th, td {
      border: 1px solid #dddddd;
      text-align: left;
      padding: 8px;
    }
    th {
      background-color: #f2f2f2;
    }
  </style>
</head>
<body>

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

</body>
</html>
"""
