import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const script = readFileSync(new URL('../modules/ui/live_microphone.js', import.meta.url), 'utf8');

function recorderHarness(upload = async () => ({path: '/cache/audio.wav', url: '/audio.wav'})) {
  const nodes = new Map();
  function node(selector) {
    if (!nodes.has(selector)) nodes.set(selector, {
      value: '', style: {}, disabled: false, hidden: false, textContent: '',
      querySelector: node, addEventListener() {}, setAttribute() {}, replaceChildren() {}, add() {},
    });
    return nodes.get(selector);
  }
  const events = [], props = {value: null}, tracks = [{stop() {this.stopped = true;}, addEventListener() {}}];
  class Worklet {
    constructor() {
      this.port = {postMessage: () => queueMicrotask(() => this.port.onmessage({data: {flushed: true}}))};
    }
    connect() {} disconnect() {}
  }
  class Context {
    sampleRate = 16000;
    state = 'running';
    audioWorklet = {addModule: async () => {}};
    createMediaStreamSource() {return {connect() {}, disconnect() {}};}
    async resume() {} async close() {this.state = 'closed';}
  }
  const navigator = {mediaDevices: {
    async enumerateDevices() {return [];}, async getUserMedia() {return {getTracks: () => tracks};},
    addEventListener() {}, removeEventListener() {},
  }};
  const sandbox = vm.createContext({
    element: {querySelector: node}, props, upload,
    trigger: event => events.push({event, value: props.value}),
    navigator, window: {AudioWorkletNode: Worklet, addEventListener() {}},
    AudioWorkletNode: Worklet, AudioContext: Context, Option: class {},
    crypto: {randomUUID: () => 'recording-123'},
    File, Blob, URL, console, setInterval: () => 1, clearInterval() {}, setTimeout, clearTimeout, queueMicrotask,
  });
  const api = vm.runInContext(`(function() {${script}\n return {
    startRecording, stopRecording, sendPreview, saveRecording, wavFile, processorSource,
    addSamples: samples => {chunks.push(samples); sampleCount += samples.length;},
    count: () => sampleCount, status, retryButton, recordButton,
  };})()`, sandbox);
  return {api, events, props, tracks, navigator};
}

test('AudioWorklet retains every frame across block boundaries and flushes its final partial block', () => {
  const {api} = recorderHarness();
  let Processor;
  const received = [];
  vm.runInNewContext(api.processorSource, {
    AudioWorkletProcessor: class {port = {postMessage: value => received.push(value)};},
    registerProcessor(name, cls) {Processor = cls;},
  });
  const processor = new Processor();
  for (let i = 0; i < 417; i++) processor.process([[new Float32Array(128).fill(.25), new Float32Array(128).fill(.75)]]);
  processor.port.onmessage({data: 'flush'});
  const blocks = received.filter(item => item.samples).map(item => item.samples);
  assert.equal(blocks.reduce((n, block) => n + block.length, 0), 417 * 128);
  assert(blocks.every(block => block.every(value => value === .5)));
  assert.equal(received.at(-1).flushed, true);
  const afterFlush = received.length;
  processor.process([[new Float32Array(128)]]);
  assert.equal(received.length, afterFlush);
});

test('final WAV preserves full duration and preview WAV contains the most recent 15 seconds', async () => {
  const {api} = recorderHarness();
  api.addSamples(new Float32Array(16000 * 22).fill(-.25));
  api.addSamples(new Float32Array(16000 * 15).fill(.5));
  const full = new DataView(await api.wavFile().arrayBuffer());
  const preview = new DataView(await api.wavFile(15).arrayBuffer());
  assert.equal(full.getUint32(24, true), 16000);
  assert.equal(full.getUint32(40, true), 37 * 16000 * 2);
  assert.equal(full.getInt16(44, true), -8192);
  assert.equal(full.getInt16(full.byteLength - 2, true), 16384);
  assert.equal(preview.getUint32(40, true), 15 * 16000 * 2);
  assert.equal(preview.getInt16(44, true), 16384);
});

test('a stalled preview cannot block, overwrite or truncate the final upload on Stop', {timeout: 3000}, async () => {
  let finishPreview;
  const uploads = [];
  const {api, events, tracks} = recorderHarness(async file => {
    uploads.push(file);
    if (uploads.length === 1) return new Promise(resolve => {finishPreview = resolve;});
    return {path: '/cache/final.wav', url: '/final.wav'};
  });
  await api.startRecording();
  api.addSamples(new Float32Array(16000 * 22).fill(.5));
  const preview = api.sendPreview();
  api.addSamples(new Float32Array(16000 * 15).fill(.25));
  await api.stopRecording();
  assert.equal(events.at(-1).event, 'stop_recording');
  finishPreview({path: '/cache/preview.wav', url: '/preview.wav'});
  await preview;
  assert.deepEqual(events.map(item => item.event), ['start_recording', 'stop_recording']);
  assert.equal(events.at(-1).value.capture_mode, 'complete_recording');
  assert.equal(events.at(-1).value.total_samples, 37 * 16000);
  assert.equal(uploads[1].size, 44 + 37 * 16000 * 2);
  assert.equal(tracks[0].stopped, true);
});

test('failed final uploads keep complete audio available for retry', async () => {
  let calls = 0;
  const uploads = [];
  const {api, events} = recorderHarness(async file => {
    uploads.push(file);
    if (++calls === 1) throw Error('connection lost');
    return {path: '/cache/final.wav'};
  });
  await api.startRecording();
  api.addSamples(new Float32Array(16000 * 2));
  await api.stopRecording();
  assert.equal(api.retryButton.hidden, false);
  assert.match(api.status.textContent, /connection lost/);
  assert.equal(events.length, 1);
  await api.saveRecording();
  assert.equal(uploads[0], uploads[1]);
  assert.equal(events.at(-1).event, 'stop_recording');
  assert.equal(api.retryButton.hidden, true);
});

test('permission failures leave a visible error and allow another recording attempt', async () => {
  const {api, navigator, events} = recorderHarness();
  navigator.mediaDevices.getUserMedia = async () => {throw Error('Permission denied');};
  await api.startRecording();
  assert.match(api.status.textContent, /Permission denied/);
  assert.equal(api.recordButton.disabled, false);
  assert.equal(events.length, 0);
});
