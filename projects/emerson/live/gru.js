// Emerson GRU world model — browser/Node inference, mathematically identical
// to the PyTorch GatedRecurrentWorldModel (nn.GRU input=4, hidden=256, one
// layer, plus a Linear head). Weights arrive in the exhibit JSON format:
// {tensors: {name: {shape, data_b64}}} with float32 little-endian payloads.
'use strict';

function decodeTensor(entry) {
  const bin = typeof window !== 'undefined'
    ? atob(entry.data_b64)
    : Buffer.from(entry.data_b64, 'base64').toString('binary');
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

class EmersonGRU {
  constructor(weightsDoc) {
    const t = weightsDoc.tensors;
    this.Wih = decodeTensor(t['gru.weight_ih_l0']); // (3H, 4) rows [r; z; n]
    this.Whh = decodeTensor(t['gru.weight_hh_l0']); // (3H, H)
    this.bih = decodeTensor(t['gru.bias_ih_l0']);   // (3H)
    this.bhh = decodeTensor(t['gru.bias_hh_l0']);   // (3H)
    this.headW = decodeTensor(t['head.weight']);    // (1, H)
    this.headB = decodeTensor(t['head.bias']);      // (1)
    this.H = this.headW.length;
    this.h = new Float64Array(this.H);
  }

  reset() { this.h.fill(0); }

  // obsNorm: position/10 in [0,1]; act: -1 | 0 | 1. Returns predicted next obsNorm.
  step(obsNorm, act) {
    const H = this.H, h = this.h;
    const x = [obsNorm, act === -1 ? 1 : 0, act === 0 ? 1 : 0, act === 1 ? 1 : 0];
    const gi = new Float64Array(3 * H), gh = new Float64Array(3 * H);
    for (let r = 0; r < 3 * H; r++) {
      let s = this.bih[r];
      const row = r * 4;
      for (let c = 0; c < 4; c++) s += this.Wih[row + c] * x[c];
      gi[r] = s;
    }
    for (let r = 0; r < 3 * H; r++) {
      let s = this.bhh[r];
      const row = r * H;
      for (let c = 0; c < H; c++) s += this.Whh[row + c] * h[c];
      gh[r] = s;
    }
    let pred = this.headB[0];
    for (let i = 0; i < H; i++) {
      const rGate = 1 / (1 + Math.exp(-(gi[i] + gh[i])));
      const zGate = 1 / (1 + Math.exp(-(gi[H + i] + gh[H + i])));
      const nGate = Math.tanh(gi[2 * H + i] + rGate * gh[2 * H + i]);
      const hn = (1 - zGate) * nGate + zGate * h[i];
      h[i] = hn;
      pred += this.headW[i] * hn;
    }
    return pred;
  }
}

if (typeof module !== 'undefined') module.exports = { EmersonGRU, decodeTensor };
