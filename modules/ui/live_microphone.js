// The full PCM recording stays in this browser until Stop successfully uploads it.
// Preview uploads contain complete rolling windows, so a slow server may skip a
// preview without dropping audio from either the next preview or the final file.
const recordButton = element.querySelector('[data-action="record"]');
const stopButton = element.querySelector('[data-action="stop"]');
const retryButton = element.querySelector('[data-action="retry"]');
const deviceSelect = element.querySelector('select');
const status = element.querySelector('[data-status]');
const downloadLink = element.querySelector('[data-download]');
const meter = element.querySelector('[role="meter"]');
const meterFill = meter.querySelector('span');
let context, mediaStream, source, worklet, timer, recording = false, disposed = false;
let chunks = [], sampleCount = 0, sampleRate = 16000, recordingId = '';
let previewUpload = null, finalFile = null, downloadURL = null, flushFinished = null;

const processorSource = `
class LivePCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(4096);
    this.used = 0;
    this.active = true;
    this.port.onmessage = ({data}) => {
      if (data === 'flush') {
        this.active = false;
        this.flush();
        this.port.postMessage({flushed: true});
      }
    };
  }
  flush() {
    if (this.used) {
      const block = this.buffer.slice(0, this.used);
      this.port.postMessage({samples: block}, [block.buffer]);
      this.used = 0;
    }
  }
  process(inputs) {
    const channels = inputs[0];
    if (!this.active || !channels || !channels.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let sample = 0;
      for (const channel of channels) sample += channel[i];
      this.buffer[this.used++] = sample / channels.length;
      if (this.used === this.buffer.length) this.flush();
    }
    return true;
  }
}
registerProcessor('live-pcm-recorder', LivePCMRecorder);
`;

function wavFile(recentSeconds = null) {
  const count = recentSeconds === null ? sampleCount : Math.min(sampleCount, Math.round(recentSeconds * sampleRate));
  const buffer = new ArrayBuffer(44 + count * 2);
  const view = new DataView(buffer);
  function textAt(offset, text) { for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i)); }
  textAt(0, 'RIFF'); view.setUint32(4, 36 + count * 2, true); textAt(8, 'WAVE');
  textAt(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  textAt(36, 'data'); view.setUint32(40, count * 2, true);
  let skip = sampleCount - count, offset = 44;
  for (const chunk of chunks) {
    if (skip >= chunk.length) { skip -= chunk.length; continue; }
    for (let i = skip; i < chunk.length; i++) {
      const value = Math.max(-1, Math.min(1, chunk[i]));
      view.setInt16(offset, Math.round(value * (value < 0 ? 32768 : 32767)), true);
      offset += 2;
    }
    skip = 0;
  }
  return new File([buffer], recentSeconds === null ? 'live-recording.wav' : 'live-preview.wav', {type: 'audio/wav'});
}

async function refreshDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const selected = deviceSelect.value;
    const devices = await navigator.mediaDevices.enumerateDevices();
    deviceSelect.replaceChildren(new Option('Default microphone', ''));
    for (const device of devices.filter(item => item.kind === 'audioinput')) {
      if (device.deviceId) deviceSelect.add(new Option(device.label || `Microphone ${deviceSelect.length}`, device.deviceId));
    }
    deviceSelect.value = selected;
  } catch (error) { console.debug('Microphone device list unavailable:', error); }
}

async function sendPreview() {
  if (!recording || previewUpload || !sampleCount) return;
  const totalSamples = sampleCount;
  const id = recordingId;
  const pending = (async () => {
    try {
      const uploaded = await upload(wavFile(15));
      if (!recording || recordingId !== id || disposed) return;
      props.value = {...uploaded, capture_mode: 'preview_window', sample_rate: sampleRate,
        total_samples: totalSamples, recording_id: id};
      trigger('input');
    } catch (error) {
      if (recording) status.textContent = `Recording ${(sampleCount / sampleRate).toFixed(1)}s. Preview upload failed; the full recording is retained. ${error.message || error}`;
    }
  })();
  previewUpload = pending;
  try { await pending; } finally { if (previewUpload === pending) previewUpload = null; }
}

async function releaseMicrophone() {
  clearInterval(timer);
  mediaStream?.getTracks().forEach(track => track.stop());
  source?.disconnect();
  worklet?.disconnect();
  if (context && context.state !== 'closed') await context.close();
  mediaStream = source = worklet = context = null;
  meterFill.style.width = '0%';
  meter.setAttribute('aria-valuenow', '0');
}

async function saveRecording() {
  retryButton.hidden = true;
  recordButton.disabled = true;
  status.textContent = `Saving complete recording (${(sampleCount / sampleRate).toFixed(1)}s)…`;
  try {
    const uploaded = await upload(finalFile);
    if (disposed) return;
    props.value = {...uploaded, capture_mode: 'complete_recording', sample_rate: sampleRate,
      total_samples: sampleCount, recording_id: recordingId};
    trigger('stop_recording');
    status.textContent = `Saved ${(sampleCount / sampleRate).toFixed(1)}s. Audio uploaded for subtitle generation.`;
    chunks = [];
    finalFile = null;
  } catch (error) {
    status.textContent = `Could not save the recording: ${error.message || error}. Retry saving or download the audio below.`;
    retryButton.hidden = false;
  } finally {
    recordButton.disabled = false;
    deviceSelect.disabled = false;
  }
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  clearInterval(timer);
  stopButton.disabled = true;
  status.textContent = 'Finishing recording…';
  // Acknowledgement comes after the processor's last PCM block on the same port.
  await new Promise(resolve => {
    // A disconnected device or failed audio processor must not trap the Stop button.
    const timeout = setTimeout(() => {flushFinished = null; resolve();}, 2000);
    flushFinished = () => {clearTimeout(timeout); resolve();};
    worklet.port.postMessage('flush');
  });
  await releaseMicrophone();
  // A stalled preview upload must not block saving/downloading the complete recording.
  // Its completion checks recording/id before publishing, so it cannot replace this file.
  if (!sampleCount) {
    status.textContent = 'No audio was captured. Check microphone permissions and the selected device.';
    recordButton.disabled = deviceSelect.disabled = false;
    return;
  }
  finalFile = wavFile();
  if (downloadURL) URL.revokeObjectURL(downloadURL);
  downloadURL = URL.createObjectURL(finalFile);
  downloadLink.href = downloadURL;
  downloadLink.hidden = false;
  await saveRecording();
}

async function startRecording() {
  if (recording || disposed) return;
  recordButton.disabled = deviceSelect.disabled = true;
  retryButton.hidden = true;
  status.textContent = 'Opening microphone…';
  try {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      throw new Error('Microphone recording requires Chrome on localhost or an HTTPS connection.');
    }
    const selectedDevice = deviceSelect.value;
    mediaStream = await navigator.mediaDevices.getUserMedia({audio: {
      deviceId: selectedDevice ? {exact: selectedDevice} : undefined,
      channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false
    }});
    if (disposed) {await releaseMicrophone(); return;}
    context = new AudioContext({sampleRate: 16000});
    sampleRate = context.sampleRate;
    const processorURL = URL.createObjectURL(new Blob([processorSource], {type: 'text/javascript'}));
    try { await context.audioWorklet.addModule(processorURL); }
    finally { URL.revokeObjectURL(processorURL); }
    chunks = []; sampleCount = 0; finalFile = null; previewUpload = null;
    recordingId = crypto.randomUUID();
    worklet = new AudioWorkletNode(context, 'live-pcm-recorder');
    worklet.port.onmessage = ({data}) => {
      if (data.samples) {
        chunks.push(data.samples);
        sampleCount += data.samples.length;
        let peak = 0;
        for (const sample of data.samples) peak = Math.max(peak, Math.abs(sample));
        const level = Math.min(100, Math.round(peak * 100));
        meterFill.style.width = `${level}%`;
        meter.setAttribute('aria-valuenow', String(level));
        if (recording) status.textContent = `Recording ${(sampleCount / sampleRate).toFixed(1)}s. Stop to save the complete audio and generate subtitles.`;
      }
      if (data.flushed && flushFinished) { flushFinished(); flushFinished = null; }
    };
    source = context.createMediaStreamSource(mediaStream);
    recording = true;
    props.value = null;
    trigger('start_recording');
    source.connect(worklet);
    worklet.connect(context.destination); // Worklet emits silence; keep its audio clock running.
    await context.resume();
    stopButton.disabled = false;
    downloadLink.hidden = true;
    timer = setInterval(sendPreview, 2000);
    mediaStream.getTracks().forEach(track => track.addEventListener('ended', () => { if (recording) stopRecording(); }));
    await refreshDevices();
    status.textContent = 'Recording 0.0s. Stop to save the complete audio and generate subtitles.';
  } catch (error) {
    recording = false;
    await releaseMicrophone();
    status.textContent = `Microphone unavailable: ${error.message || error}. Allow microphone access and check the selected device.`;
    recordButton.disabled = deviceSelect.disabled = false;
    stopButton.disabled = true;
  }
}

recordButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', stopRecording);
retryButton.addEventListener('click', saveRecording);
navigator.mediaDevices?.addEventListener('devicechange', refreshDevices);
window.addEventListener('pagehide', () => {
  disposed = true;
  recording = false;
  releaseMicrophone();
  navigator.mediaDevices?.removeEventListener('devicechange', refreshDevices);
  if (downloadURL) URL.revokeObjectURL(downloadURL);
}, {once: true});
refreshDevices();
