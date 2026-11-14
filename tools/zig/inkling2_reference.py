# Development KV rounding differs from deployed vLLM; see the family README for this oracle's limitations.
"""Independent CPU reference for the Inkling2 text target; see the family ABI.txt."""
import argparse
import json
import math
import pathlib
import struct
import numpy as np


def bf(x):
    x = np.asarray(x, dtype=np.float32).copy()
    bits = x.view(np.uint32)
    bits += np.uint32(0x7fff)+(bits >> 16 & 1)
    bits &= np.uint32(0xffff0000)
    return x.astype(np.float64)


def fp8_table():
    result = []
    for c in range(256):
        e, m = (c & 127) >> 3, c & 7
        x = math.nan if c & 127 == 127 else math.ldexp(m, -9) if e == 0 else math.ldexp(1+m/8, e-7)
        result.append(-x if c & 128 else x)
    return np.array(result, dtype=np.float64)


F8 = fp8_table()
F4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def nearest(x, positive):
    a = np.abs(x)
    hi = np.minimum(np.searchsorted(positive, a), len(positive)-1)
    lo = np.maximum(hi-1, 0)
    dl, dh = a-positive[lo], positive[hi]-a
    take_hi = (dh < dl) | ((dh == dl) & ((hi & 1) == 0))
    return np.copysign(positive[np.where(take_hi, hi, lo)], x).astype(np.float32)


def nv_quant(x, normalized=False):
    x = np.asarray(x, dtype=np.float32)
    amax = np.max(np.abs(x))
    if amax == 0:
        return (x.copy().astype(np.float64),np.float32(0)) if normalized else x.copy().astype(np.float64)
    enc = np.minimum(np.float32(1536)/amax, np.finfo(np.float32).max)
    dec = np.float32(1)/enc
    outer = amax*np.float32(1/1536)
    groups = x.reshape(-1, 16)
    bmax = np.max(np.abs(groups), axis=1)
    candidates, normalized_candidates, errors = [], [], []
    for factor in [np.float32(1/6), np.float32(.25)]:
        scale = nearest(np.minimum(bmax*(enc*factor), np.float32(448)), F8[:127].astype(np.float32))
        with np.errstate(divide='ignore'):
            inverse = np.minimum(np.float32(1)/(scale*dec), np.finfo(np.float32).max)
        codes = nearest(groups*inverse[:, None], F4[:8].astype(np.float32))
        normed = codes*scale[:, None]
        y = normed*outer
        error = np.abs(y-groups)
        for _ in range(4):
            error = error[:, 0::2]+error[:, 1::2]
        candidates.append(y)
        normalized_candidates.append(normed)
        errors.append(error[:, 0])
    picked = normalized_candidates if normalized else candidates
    value = np.where((errors[1] < errors[0])[:, None], picked[1], picked[0]).reshape(-1).astype(np.float64)
    return (value,amax) if normalized else value


def mx_quant(x):
    groups = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    raw = np.max(np.abs(groups), axis=1)*np.float32(1/448)
    exponent = ((raw.view(np.uint32)+0x7fffff) & 0x7f800000) >> 23
    exponent = np.where(raw <= np.float32(2**-127), 0, np.minimum(exponent, 254)).astype(np.uint32)
    scale = np.maximum((exponent << 23).view(np.float32), np.float32(2**-127))
    inverse = np.where(exponent == 0, np.float32(2**127), np.float32(1)/scale)
    q = nearest(np.clip(groups*inverse[:, None], -448, 448), F8[:127].astype(np.float32))
    return (q*scale[:, None]).reshape(-1).astype(np.float64)


class Checkpoint:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.index = json.loads((self.path/'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}
        self.current = 0

    def tensor(self, name, expert=None):
        file = self.index[name]
        if file not in self.headers:
            with (self.path/file).open('rb') as f:
                size = struct.unpack('<Q', f.read(8))[0]
                self.headers[file] = (8+size, json.loads(f.read(size)))
        offset, table = self.headers[file]
        t = table[name]
        dtype = {'BF16': '<u2', 'F32': '<f4', 'U8': 'u1', 'F8_E4M3': 'u1'}[t['dtype']]
        arr = np.memmap(self.path/file, mode='r', dtype=dtype, offset=offset+t['data_offsets'][0], shape=tuple(t['shape']))
        if expert is not None:
            arr = arr[expert]
        if t['dtype'] == 'BF16':
            return (arr.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return arr

    def layer(self, tail, expert=None):
        return self.tensor(f'model.llm.layers.{self.current}.'+tail, expert)

    def expert(self, name, e, normalized=False):
        packed = self.layer('mlp.experts.'+name+'_weight', e)
        if self.current >= 40:
            vals = F8[packed]
            scale = self.layer('mlp.experts.'+name+'_weight_scale', e)
            assert np.all(scale != 255)
            scale = np.ldexp(np.ones(scale.shape), scale.astype(np.int32)-127)
            vals.reshape(vals.shape[0], -1, 32)[:] *= scale[:, :, None]
            return vals
        N, packed_k = packed.shape
        vals = np.empty((N, packed_k*2), dtype=np.float64)
        vals[:, 0::2] = F4[packed & 15]
        vals[:, 1::2] = F4[packed >> 4]
        scales = F8[self.layer('mlp.experts.'+name+'_weight_scale', e)]
        amax = float(self.layer('mlp.experts.'+name+'_weight_amax', e))
        denom = float(self.layer('mlp.experts.'+name+'_weight_outer_scale_denominator')[0])
        vals.reshape(N, -1, 16)[:] *= scales[:, :, None]
        if not normalized:
            vals *= amax/denom
        return vals

    def embedding(self, token):
        return self.rows('model.llm.embed.weight', token, 1)[0]

    def rows(self, name, first, count):
        file = self.index[name]
        if file not in self.headers:
            with (self.path/file).open('rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                self.headers[file] = (8+n, json.loads(f.read(n)))
        offset, table = self.headers[file]
        entry = table[name]
        assert entry['dtype'] == 'BF16' and len(entry['shape']) == 2
        columns = entry['shape'][1]
        assert first+count <= entry['shape'][0]
        raw = np.memmap(self.path/file, mode='r', dtype='<u2',
                        offset=offset+entry['data_offsets'][0]+first*columns*2, shape=(count, columns))
        return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    def head(self, h, trace=None):
        mix = residual(h[0], h[1], self.tensor('model.llm.attn_res.output_query'))
        x = rms(mix, self.tensor('model.llm.norm.weight'))
        if trace:
            trace('final_mix', mix);trace('final_norm', x)
        out = np.empty(200058, dtype=np.float64)
        for first in range(0, 200058, 4096):
            count = min(4096, 200058-first)
            z = (self.rows('model.llm.unembed.weight', first, count) @ x).astype(np.float32)
            out[first:first+count] = np.float32(30)*np.tanh(z/np.float32(30))
        return out


def rms(x, w=None):
    y = x/np.sqrt(np.mean(x*x)+1e-6)
    return bf(y if w is None else w*y)


def residual(states, partial, query):
    sources = np.stack([*states, partial])
    normalized = sources/np.sqrt(np.mean(sources*sources, axis=1, keepdims=True)+1e-6)
    if query.ndim == 1:
        logits = normalized @ query/math.sqrt(partial.size)
        p = np.exp(logits-logits.max());p /= p.sum()
        return p @ normalized
    logits = normalized @ query.T/math.sqrt(partial.size)
    p = np.exp(logits-logits.max(axis=0, keepdims=True));p /= p.sum(axis=0, keepdims=True)
    return p.T @ normalized


class Layer:
    def __init__(self, ck, index):
        self.ck = ck
        self.index = index
        self.hist = {k: [] for k in ['k', 'v', 'a', 'm']}
        self.keys, self.values = [], []
        self.trace = None

    def emit(self, tag, x):
        if self.trace:
            self.trace(self.index, tag, x)

    def linear(self, tail, x):
        return bf(self.ck.layer(tail) @ x)

    def conv(self, key, x, tail, dynamic_input=None):
        if dynamic_input is None:
            w = self.ck.layer(tail).reshape(x.size, 4)
        else:
            w = self.ck.layer(tail+'static_w').T
            dyn = bf(self.ck.layer(tail+'weight_proj') @ dynamic_input).reshape(8, 4)
            w = w+np.repeat(dyn, 128, axis=0)
        previous = self.hist[key]
        z = x.copy()
        for lag, row in enumerate([x]+previous[:3]):
            z += w[:, 3-lag if dynamic_input is None else lag]*row
        previous.insert(0, x.copy())
        del previous[3:]
        z *= float(np.float32(1/math.sqrt(5) if dynamic_input is None else 1/3))
        return bf(z) if key == 'a' else z

    def forward(self, h, states):
        c = self.ck
        l = self.index;c.current = l
        self.emit('entry', h)
        self.emit('stack', np.stack(states) if states else np.zeros((0, 4608)))
        glob = l % 6 == 5
        if l == 0:
            x = rms(h, c.layer('attn_norm.weight'));ki = vi = x
        else:
            query = c.tensor(f'model.llm.attn_res.attn_queries.{l-1}')
            mixed = residual(states, h, query)
            if glob:
                x = rms(mixed[0], c.layer('q_input_norm.weight'))
                ki = rms(mixed[1], c.layer('k_input_norm.weight'))
                vi = rms(mixed[2], c.layer('v_input_norm.weight'))
            else:
                x = rms(mixed, c.layer('attn_norm.weight'));ki = vi = x
        self.emit('norm_q', x)
        if glob:
            self.emit('norm_k', ki);self.emit('norm_v', vi)
        kp = self.linear('attn.wk_dv.weight', ki)
        vp = self.linear('attn.wv_dv.weight', vi)
        k = self.conv('k', kp, 'attn.k_sconv.' if glob else 'attn.k_sconv.weight', ki if glob else None).reshape(8, 128)
        v = self.conv('v', vp, 'attn.v_sconv.' if glob else 'attn.v_sconv.weight', vi if glob else None).reshape(8, 128)
        k = bf(k*np.minimum(1, float(np.float32(2*math.sqrt(128)))/(np.sqrt((k*k).sum(1))+1e-6))[:, None])
        v = bf(v/np.sqrt(np.mean(v*v, axis=1, keepdims=True)+1e-6))
        self.keys.append(k);self.values.append(v)
        q = self.linear('attn.wq_du.weight', x).reshape(48, 128)*c.layer('attn.q_l2_clip.scale')
        r = self.linear('attn.wr_du.weight', x).reshape(48, 16)*c.layer('attn.rel_l2_clip.scale')
        q = bf(q*np.minimum(1, float(np.float32(2*math.sqrt(128)))/(np.sqrt((q*q).sum(1))+1e-6))[:, None])
        r = bf(r*np.minimum(1, 8/(np.sqrt((r*r).sum(1))+1e-6))[:, None])
        pos = len(self.keys)-1
        if glob:
            tau = float(np.float32(1+np.float32(.1)*np.log(np.float32(max(1, (pos+1)/128000)))))
            q = bf(q*tau);r = bf(r*tau)
        profile = r @ c.layer('attn.rel_logits_proj.proj')
        sink = self.linear('attn.attn_sink_proj.weight', x)
        gate = self.linear('attn.gate_attn.weight', x).reshape(48, 128)
        att = np.zeros((48, 128))
        K, V = np.stack(self.keys), np.stack(self.values)
        lo = 0 if glob else max(0, pos-1023)
        for head in range(48):
            scores = K[lo:, head//6] @ q[head]/math.sqrt(128)
            dist = pos-np.arange(lo, pos+1);mask = dist < 1024
            scores[mask] += profile[head, dist[mask]]
            mx = max(sink[head], scores.max())
            p = np.exp(scores-mx)
            p /= p.sum()+math.exp(sink[head]-mx)
            att[head] = bf(bf(p @ V[lo:, head//6])/(1+np.exp(-gate[head])))
        ow=c.layer('attn.wo_ud.weight');flat=att.reshape(-1)
        a=bf(sum(bf(ow[:,rank*1536:(rank+1)*1536] @ flat[rank*1536:(rank+1)*1536]) for rank in range(4)))
        self.emit('attn_out', a)
        ac = self.conv('a', a, 'attn_sconv.weight')
        self.emit('attn_conv', ac)
        if l % 6 == 0:
            states.append(h.copy());mid = ac
        else:
            mid = h+ac
        self.emit('mid', mid)
        mix = residual(states, mid, c.tensor(f'model.llm.attn_res.mlp_queries.{l}'))
        mlp = rms(mix, c.layer('mlp_norm.weight'))
        self.emit('mlp_in', mlp)
        logits = c.layer('mlp.gate.weight') @ mlp
        scores = 1/(1+np.exp(-logits))
        biased = scores+c.layer('mlp.gate.bias')
        picked = np.argsort(-biased, kind='stable')[:8]
        gamma = 8*scores[picked]/(scores[picked].sum()+1e-20)
        self.emit('route_logits', logits);self.emit('route_ids', picked);self.emit('gamma', gamma)
        quant = mx_quant if l >= 40 else nv_quant
        xq, x_amax = nv_quant(mlp,True) if l<40 else (mx_quant(mlp),np.float32(1))
        self.emit('x_quant', (xq.astype(np.float32)*np.float32(x_amax*np.float32(1/1536))).astype(np.float64) if l<40 else xq)
        partials = np.zeros((4, 4608))
        for e, weight in zip(picked, gamma):
            scale1=np.float32(1) if l>=40 else np.float32(np.float32(x_amax*np.float32(c.layer('mlp.experts.w13_weight_amax',int(e))))*np.float32(1/(np.float32(1536)*np.float32(c.layer('mlp.experts.w13_weight_outer_scale_denominator')[0]))))
            gu = bf((c.expert('w13',int(e),True) @ xq).astype(np.float32)*scale1).reshape(2304, 2)
            g, u = gu[:, 0], gu[:, 1]
            act = bf(((4*np.tanh(g/4))/(1+np.exp(-g)))*(25*np.tanh(u/25))*weight)
            aq,a_amax=nv_quant(act,True) if l<40 else (mx_quant(act),np.float32(1))
            scale2=np.float32(1) if l>=40 else np.float32(np.float32(a_amax*np.float32(c.layer('mlp.experts.w2_weight_amax',int(e))))*np.float32(1/(np.float32(1536)*np.float32(c.layer('mlp.experts.w2_weight_outer_scale_denominator')[0]))))
            down = bf((c.expert('w2',int(e),True) @ aq).astype(np.float32)*scale2)
            partials[e//96] += down
        moe = bf(np.sum(bf(partials), axis=0))
        self.emit('moe_out', moe)
        mc = self.conv('m', moe, 'mlp_sconv.weight')
        self.emit('mlp_conv', mc)
        out = bf(mid+mc)
        self.emit('exit', out)
        return out, picked.tolist(), gamma.tolist()


def main():
    p = argparse.ArgumentParser();p.add_argument('model');p.add_argument('native_outputs')
    p.add_argument('--layers', type=int, default=1);p.add_argument('--reference-only', action='store_true')
    p.add_argument('--trace-dir');args = p.parse_args()
    assert 1 <= args.layers <= 48
    root = pathlib.Path(args.native_outputs)
    ids = json.loads((root/'ids.json').read_text())
    ck = Checkpoint(args.model);layers = [Layer(ck, i) for i in range(args.layers)];results = []
    for row, token in enumerate(ids):
        def trace(layer, tag, x):
            if not args.trace_dir:
                return
            folder = pathlib.Path(args.trace_dir);folder.mkdir(parents=True, exist_ok=True)
            dtype, suffix = ('<i4', 'i32') if tag == 'route_ids' else ('<f4', 'f32')
            np.asarray(x).astype(dtype).tofile(folder/f'r{row}-l{layer}-{tag}.{suffix}')
        for l in layers:
            l.trace = trace if args.trace_dir else None
        want = rms(ck.embedding(token), ck.tensor('model.llm.embed_norm.weight'))
        trace(0, 'embedding', want)
        states = []
        for l in layers:
            want, experts, gamma = l.forward(want, states)
            if args.layers > 1 and (l.index+1)%6 == 0:
                print(json.dumps(dict(row=row, layer=l.index+1, hidden_l2=float(np.linalg.norm(want)))), flush=True)
        if args.layers == 48:
            want = ck.head((states, want), lambda tag, x:trace(48, tag, x))
            trace(48, 'logits', want[:50048])
        if args.reference_only:
            target = root/f'row{row}-reference-{"logits" if args.layers == 48 else "hidden"}.f32'
            want.astype('<f4').tofile(target)
            record = dict(row=row, token=token, layers=args.layers, finite=bool(np.isfinite(want).all()),
                          outputs=want.size, top1=int(want.argmax()) if args.layers == 48 else None)
            results.append(record);print(json.dumps(record), flush=True);continue
        got = np.fromfile(root/f'row{row}-hidden.f32', dtype='<f4').astype(np.float64)
        assert got.shape == want.shape and np.isfinite(want).all() and np.isfinite(got).all()
        delta = got-want
        record = dict(row=row, token=token, cosine=float(got@want/(np.linalg.norm(got)*np.linalg.norm(want))),
                      relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(want)), max_abs=float(np.abs(delta).max()),
                      exact_elements=int(np.count_nonzero(got == want)), elements=4608, experts=experts, gamma=gamma)
        results.append(record);print(json.dumps(record), flush=True)
        want.astype('<f4').tofile(root/f'row{row}-reference-hidden.f32')
    summary = dict(scope=f'CPU FP64 {args.layers}-layer reference with checkpoint activation quantization, not serving qualification',
                   numpy_version=np.__version__, results=results)
    (root/'reference-result.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
