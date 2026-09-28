"""Browser PCM recorder that retains the complete recording independently of previews."""

from pathlib import Path

import gradio as gr


def live_microphone():
    return gr.HTML(
        value=None,
        label="Live microphone preview",
        elem_id="live-mic-recorder",
        elem_classes=["mic-recorder-frame", "mic-recorder-live"],
        html_template="""
        <div class="live-recorder">
          <label>Microphone device <select aria-label="Live Mic microphone device">
            <option value="">Default microphone</option>
          </select></label>
          <div class="live-controls">
            <button type="button" data-action="record">Live Mic Record</button>
            <button type="button" data-action="stop" disabled>Live Mic Stop</button>
            <button type="button" data-action="retry" hidden>Retry saving recording</button>
          </div>
          <div class="live-level" role="meter" aria-label="Microphone level" aria-valuemin="0"
               aria-valuemax="100" aria-valuenow="0"><span></span></div>
          <p data-status role="status">Ready to record.</p>
          <a data-download hidden download="live-recording.wav">Download recorded audio</a>
        </div>
        """,
        css_template="""
          .live-recorder { padding: 16px; }
          label { display: grid; gap: 6px; }
          select { width: 100%; padding: 8px; color: var(--body-text-color);
                   background: var(--input-background-fill); border: 1px solid var(--border-color-primary); }
          .live-controls { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
          button { padding: 8px 12px; border-radius: 6px; cursor: pointer;
                   border: 1px solid var(--button-secondary-border-color);
                   background: var(--button-secondary-background-fill); color: var(--button-secondary-text-color); }
          button:disabled { opacity: .5; cursor: default; }
          [hidden] { display: none !important; }
          .live-level { height: 8px; border-radius: 4px; overflow: hidden; background: var(--border-color-primary); }
          .live-level span { display: block; height: 100%; width: 0; background: #22a06b; }
          p { margin: 10px 0 0; font-variant-numeric: tabular-nums; }
        """,
        js_on_load=Path(__file__).with_suffix(".js").read_text(encoding="utf-8"),
    )
