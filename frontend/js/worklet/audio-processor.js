/**
 * AudioWorkletProcessor that captures mic audio, downsamples to 22050 Hz,
 * and posts Float32Array chunks (~100ms each) to the main thread.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._inputSampleRate = sampleRate; // provided by AudioWorklet global
    this._targetRate = 22050;
    this._chunkSize = 2205; // ~100ms at 22050 Hz
    this._resampleRatio = this._targetRate / this._inputSampleRate;
    this._resampleAccumulator = 0;
    this._lastSample = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0];

    for (let i = 0; i < samples.length; i++) {
      this._resampleAccumulator += this._resampleRatio;
      while (this._resampleAccumulator >= 1.0) {
        // Linear interpolation between previous and current sample
        const frac = 1.0 - (this._resampleAccumulator - 1.0);
        const interpolated = this._lastSample + frac * (samples[i] - this._lastSample);
        this._buffer.push(interpolated);
        this._resampleAccumulator -= 1.0;

        if (this._buffer.length >= this._chunkSize) {
          const chunk = new Float32Array(this._buffer);
          this.port.postMessage({ type: 'chunk', data: chunk }, [chunk.buffer]);
          this._buffer = [];
        }
      }
      this._lastSample = samples[i];
    }

    return true;
  }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
