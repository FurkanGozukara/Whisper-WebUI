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
"""

HEAD = """
<script>
(() => {
  const selector = ".live-transcription-box textarea";

  const scrollToBottom = (textarea) => {
    if (!textarea) return;
    textarea.scrollTop = textarea.scrollHeight;
  };

  const syncAll = () => {
    document.querySelectorAll(selector).forEach(scrollToBottom);
  };

  const init = () => {
    syncAll();
    window.setInterval(syncAll, 200);
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
