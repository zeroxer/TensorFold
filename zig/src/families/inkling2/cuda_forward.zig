//! Experimental one-token target graph. Layer math is owned here; collectives use TensorFold's TP group.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const kernels = @import("cuda_kernels.zig");
const state = @import("cuda_state.zig");
const trace = @import("cuda_trace.zig");
const lane = @import("cuda_lane.zig");
const batch_mod = @import("cuda_batch.zig");
const history_mod = @import("cuda_history.zig");

pub const Forward = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    k: *const kernels.Kernels,
    w: *const weights.Weights,
    b: *state.Scratch,
    group: *const cuda.tp_group.Group,
    watch: ?trace.Watch = null,
    math: ?*const lane.Math = null,
    features: u64 = 0,
    row: u32 = 0,

    fn emit(f: *const Forward, l: u32, tag: trace.Tag, ptr: u64, count: usize) !void {
        if (f.watch) |w| try w.call(w.ctx, f.row, l, tag, ptr, count);
    }
    fn tap(f: *const Forward, l: u32, src: u64) !void {
        if (f.features == 0) return;
        for ([_]u32{ 5, 11, 17, 23, 29, 35, 41, 48 }, 0..) |id, tap_slot| if (id == l)
            try f.copy(f.features + (f.row * 8 + tap_slot) * 4608 * 4, src, 4608 * 4);
    }

    fn copy(f: *const Forward, dst: u64, src: u64, bytes: usize) !void {
        try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, f.stream.handle), "inkling2 copy");
    }
    fn matrix(f: *const Forward, w: weights.Weight, x: u64, y: u64, bf16: bool) !void {
        if (w.rank != 2) return error.BadMatrixRank;
        if (f.math) |m| if (w.lane) |packed_weight| return m.mm(f.stream, packed_weight, x, 1, y, bf16);
        try f.k.dense(f.stream, x, w.ptr, y, w.dims[0], w.dims[1], bf16);
    }
    fn linear(f: *const Forward, l: u32, name: []const u8, x: u64, y: u64, bf16: bool) !void {
        try f.matrix(try f.w.layer(l, name), x, y, bf16);
    }
    fn norm(f: *const Forward, l: u32, name: []const u8, x: u64, y: u64) !void {
        try f.k.rms(f.stream, x, (try f.w.layer(l, name)).ptr, y, f.w.c.hidden, f.w.c.rms_eps);
    }
    fn slot(f: *const Forward) u64 {
        return f.b.gathered + f.w.p.rank * f.w.c.hidden * 4;
    }
    fn gather(f: *const Forward, out: u64, bf16: bool) !void {
        if (f.group.world != 4 or f.group.rank != f.w.p.rank) return error.BadRankGroup;
        try f.group.allGather(f.stream, f.slot(), f.b.gathered, f.w.c.hidden * 4);
        try f.k.merge(f.stream, f.b.gathered, out, f.w.c.hidden, 4, bf16);
    }

    /// Appends one token. A partial-stack run returns its layer output, never final model logits.
    pub fn token(f: *Forward, s: *state.State, id: u32) !void {
        f.row = s.pos;
        const c = f.w.c;
        const p = f.w.p;
        const b = f.b;
        if (id >= c.unpadded_vocab or s.pos >= s.capacity or s.layers.len != f.w.layers) return error.BadTokenState;
        try f.k.embed(f.stream, (try f.w.get("model.llm.embed.weight")).ptr, f.slot(), id, p.vocab.start, p.valid_vocab.count, c.hidden);
        try f.gather(b.h, false);
        try f.k.rms(f.stream, b.h, (try f.w.get("model.llm.embed_norm.weight")).ptr, b.x, c.hidden, c.rms_eps);
        try f.copy(b.h, b.x, c.hidden * 4);
        try f.emit(0, .embedding, b.h, c.hidden);
        var depth: usize = 0;
        for (s.layers, 0..) |ls, i| try f.layer(s, ls, @intCast(i), &depth);
        s.pos += 1;
        if (f.w.layers == c.layers) {
            try f.k.residual(f.stream, b.stack, b.h, (try f.w.get("model.llm.attn_res.output_query")).ptr, b.mix, c.hidden, depth, 1, c.rms_eps);
            try f.emit(c.layers, .final_mix, b.mix, c.hidden);
            try f.k.rms(f.stream, b.mix, (try f.w.get("model.llm.norm.weight")).ptr, b.x, c.hidden, c.rms_eps);
            try f.emit(c.layers, .final_norm, b.x, c.hidden);
            try f.tap(48, b.x);
            try f.matrix(try f.w.get("model.llm.unembed.weight"), b.x, b.logits, false);
            try f.k.softcap(f.stream, b.logits, b.capped, p.vocab.count, p.valid_vocab.count, c.logits_cap);
            try f.emit(c.layers, .logits, b.capped, p.valid_vocab.count);
        }
    }

    /// A diagnostic or batched graph can run one actual layer with explicit committed block depth.
    pub fn layer(f: *Forward, s: *const state.State, ls: state.Layer, l: u32, depth: *usize) !void {
        const c = f.w.c;
        const b = f.b;
        f.row = s.pos;
        const local = c.local(l);
        try f.tap(l, b.h);
        try f.emit(l, .entry, b.h, c.hidden);
        try f.emit(l, .stack, b.stack, depth.* * c.hidden);
        if (l == 0) {
            try f.norm(l, "attn_norm.weight", b.h, b.norm);
        } else {
            try f.k.residual(f.stream, b.stack, b.h, (try f.w.query("attn", l - 1)).ptr, b.mix, c.hidden, depth.*, if (local) 1 else 3, c.rms_eps);
            if (local) {
                try f.norm(l, "attn_norm.weight", b.mix, b.norm);
            } else {
                try f.norm(l, "q_input_norm.weight", b.mix, b.norm);
                try f.norm(l, "k_input_norm.weight", b.mix + c.hidden * 4, b.norm + c.hidden * 4);
                try f.norm(l, "v_input_norm.weight", b.mix + c.hidden * 8, b.norm + c.hidden * 8);
            }
        }
        try f.emit(l, .norm_q, b.norm, c.hidden);
        if (!local) {
            try f.emit(l, .norm_k, b.norm + c.hidden * 4, c.hidden);
            try f.emit(l, .norm_v, b.norm + c.hidden * 8, c.hidden);
        }
        try f.attention(s, ls, l, local);
        try f.emit(l, .attn_out, b.out, c.hidden);
        try f.k.conv(f.stream, b.out, (try f.w.layer(l, "attn_sconv.weight")).ptr, ls.hist_a, 0, b.conv, c.hidden, 0, false, true);
        try f.emit(l, .attn_conv, b.conv, c.hidden);
        const boundary = l % c.residual_block == 0;
        if (boundary) {
            try f.copy(b.stack + depth.* * c.hidden * 4, b.h, c.hidden * 4);
            depth.* += 1;
        }
        try f.k.add(f.stream, b.h, b.conv, b.mid, c.hidden, boundary, false);
        try f.emit(l, .mid, b.mid, c.hidden);
        try f.k.residual(f.stream, b.stack, b.mid, (try f.w.query("mlp", l)).ptr, b.mix, c.hidden, depth.*, 1, c.rms_eps);
        try f.norm(l, "mlp_norm.weight", b.mix, b.x);
        try f.emit(l, .mlp_in, b.x, c.hidden);
        try f.moe(l);
        try f.emit(l, .moe_out, b.out, c.hidden);
        try f.k.conv(f.stream, b.out, (try f.w.layer(l, "mlp_sconv.weight")).ptr, ls.hist_m, 0, b.conv, c.hidden, 0, false, false);
        try f.emit(l, .mlp_conv, b.conv, c.hidden);
        try f.k.add(f.stream, b.mid, b.conv, b.h, c.hidden, false, true);
        try f.emit(l, .exit, b.h, c.hidden);
    }

    fn attention(f: *const Forward, s: *const state.State, ls: state.Layer, l: u32, local: bool) !void {
        const c = f.w.c;
        const b = f.b;
        const ki = if (local) b.norm else b.norm + c.hidden * 4;
        const vi = if (local) b.norm else b.norm + c.hidden * 8;
        try f.linear(l, "attn.wq_du.weight", b.norm, b.q, true);
        try f.linear(l, "attn.wr_du.weight", b.norm, b.r, true);
        try f.linear(l, "attn.wk_dv.weight", ki, b.k, true);
        try f.linear(l, "attn.wv_dv.weight", vi, b.v, true);
        try f.linear(l, "attn.gate_attn.weight", b.norm, b.gate, true);
        try f.linear(l, "attn.attn_sink_proj.weight", b.norm, b.sink, true);
        if (local) {
            try f.k.conv(f.stream, b.k, (try f.w.layer(l, "attn.k_sconv.weight")).ptr, ls.hist_k, 0, b.kc, 256, 0, false, false);
            try f.k.conv(f.stream, b.v, (try f.w.layer(l, "attn.v_sconv.weight")).ptr, ls.hist_v, 0, b.vc, 256, 0, false, false);
        } else {
            try f.linear(l, "attn.k_sconv.weight_proj", ki, b.dyn_k, true);
            try f.linear(l, "attn.v_sconv.weight_proj", vi, b.dyn_v, true);
            try f.k.conv(f.stream, b.k, (try f.w.layer(l, "attn.k_sconv.static_w")).ptr, ls.hist_k, b.dyn_k, b.kc, 256, 128, true, false);
            try f.k.conv(f.stream, b.v, (try f.w.layer(l, "attn.v_sconv.static_w")).ptr, ls.hist_v, b.dyn_v, b.vc, 256, 128, true, false);
        }
        const tau: f32 = if (local) 1 else 1 + c.log_alpha * @log(@max(1, @as(f32, @floatFromInt(s.pos + 1)) / @as(f32, @floatFromInt(c.log_floor))));
        try f.k.qr(f.stream, b.q, b.r, (try f.w.layer(l, "attn.q_l2_clip.scale")).ptr, (try f.w.layer(l, "attn.rel_l2_clip.scale")).ptr, b.qo, b.ro, 12, 128, 16, c.qk_clip, c.rel_clip, tau, c.rms_eps);
        try f.k.kv(f.stream, b.kc, b.vc, ls.k, ls.v, 128, 2, s.pos, s.capacity, c.qk_clip, c.rms_eps);
        try f.k.relative(f.stream, b.ro, (try f.w.layer(l, "attn.rel_logits_proj.proj")).ptr, b.profile);
        try f.k.scores(f.stream, b.qo, ls.k, b.profile, b.scores, s.capacity, s.pos, local);
        try f.k.attend(f.stream, b.scores, b.sink, ls.v, b.gate, b.attn, s.capacity, s.pos, local);
        try f.linear(l, "attn.wo_ud.weight", b.attn, f.slot(), true);
        try f.gather(b.out, true);
    }

    fn expert(f: *const Forward, l: u32, local: u32, up: bool, x: u64, y: u64) !void {
        const mw = try f.w.layer(l, if (up) "mlp.experts.w13_weight" else "mlp.experts.w2_weight");
        const ms = try f.w.layer(l, if (up) "mlp.experts.w13_weight_scale" else "mlp.experts.w2_weight_scale");
        const mx = f.w.c.expertFormat(l) == .mxfp8;
        var amax: u64 = 0;
        var denom: u64 = 0;
        if (!mx) {
            amax = (try f.w.layer(l, if (up) "mlp.experts.w13_weight_amax" else "mlp.experts.w2_weight_amax")).ptr + local * 4;
            denom = (try f.w.layer(l, if (up) "mlp.experts.w13_weight_outer_scale_denominator" else "mlp.experts.w2_weight_outer_scale_denominator")).ptr;
        }
        const K: usize = if (up) 4608 else 2304;
        if (f.math) |m| {
            var packed_weight = mw.lane orelse return error.MissingExpertLayout;
            packed_weight.codes = mw.ptr + local * (mw.bytes / mw.dims[0]);
            packed_weight.scales = ms.ptr + local * (ms.bytes / ms.dims[0]);
            try m.mmBf16(f.stream, packed_weight, if (up) f.b.x_norm else f.b.a_norm, 1, y, false);
            try m.go(f.stream, .quant_output, .{ .x = 18 }, .{ y, if (up) f.b.x_amax else f.b.a_amax, amax, denom, @as(c_int, 4608), @as(c_int, 1), @as(c_int, @intFromBool(!mx)) });
            return;
        }
        try f.k.qmv(f.stream, x, mw.ptr + local * (mw.bytes / mw.dims[0]), ms.ptr + local * (ms.bytes / ms.dims[0]), amax, denom, y, 4608, K, mx, true);
    }

    fn moe(f: *const Forward, l: u32) !void {
        const c = f.w.c;
        const b = f.b;
        const mx = c.expertFormat(l) == .mxfp8;
        try f.linear(l, "mlp.gate.weight", b.x, b.router, false);
        try f.k.route(f.stream, b.router, (try f.w.layer(l, "mlp.gate.bias")).ptr, b.ids, b.gammas, 384, 8, c.route_scale);
        if (f.math != null) try f.k.quantNorm(f.stream, b.x, b.xq, b.x_norm, b.x_amax, 4608, mx, 1) else try f.k.quant(f.stream, b.x, b.xq, 4608, mx);
        try f.emit(l, .route_logits, b.router, c.experts);
        try f.emit(l, .route_ids, b.ids, c.topk);
        try f.emit(l, .gamma, b.gammas, c.topk);
        try f.emit(l, .x_quant, b.xq, c.hidden);
        try f.d.check(f.d.api.cuMemsetD8Async(b.expert_parts, 0, 8 * c.hidden * 4, f.stream.handle), "inkling2 expert zeros");
        try f.stream.synchronize();
        var ids: [8]i32 = undefined;
        var gammas: [8]f32 = undefined;
        try cuda.memory.download(f.d, b.ids, std.mem.asBytes(&ids));
        try cuda.memory.download(f.d, b.gammas, std.mem.asBytes(&gammas));
        for (ids, gammas, 0..) |id, gamma, pick| {
            if (id < 0 or id >= c.experts or !std.math.isFinite(gamma)) return error.InvalidRouterOutput;
            if (!f.w.p.experts.contains(@intCast(id))) continue;
            const local = @as(u32, @intCast(id)) - f.w.p.experts.start;
            try f.expert(l, local, true, b.xq, b.gu);
            try f.k.situ(f.stream, b.gu, b.act, c.intermediate, gamma);
            if (f.math != null) try f.k.quantNorm(f.stream, b.act, b.aq, b.a_norm, b.a_amax, c.intermediate, mx, 1) else try f.k.quant(f.stream, b.act, b.aq, c.intermediate, mx);
            try f.expert(l, local, false, b.aq, b.expert_parts + pick * c.hidden * 4);
        }
        try f.k.merge(f.stream, b.expert_parts, f.slot(), c.hidden, 8, true);
        try f.gather(b.out, true);
    }

    /// Head rows to compute; every mode still writes final-normalization features for every input row.
    pub const Tail = enum { none, last, all };

    fn tapSlot(l: u32) ?usize {
        for ([_]u32{ 5, 11, 17, 23, 29, 35, 41, 48 }, 0..) |id, tap_index| if (id == l) return tap_index;
        return null;
    }
    fn tapRow(f: *const Forward, l: u32, row: u32, src: u64) !void {
        if (f.features == 0) return;
        const tap_index = tapSlot(l) orelse return;
        try f.copy(f.features + (row * 8 + @as(u32, @intCast(tap_index))) * 4608 * 4, src, 4608 * 4);
    }
    fn laneWeight(f: *const Forward, l: u32, name: []const u8) !cuda.qmmf.Weight {
        return (try f.w.layer(l, name)).lane orelse error.BatchNeedsPackedWeights;
    }
    fn gatherRows(f: *Forward, wb: *batch_mod.Batch, n: usize, out: u64, bf16: bool) !void {
        if (f.group.world != 4 or f.group.rank != f.w.p.rank) return error.BadRankGroup;
        try f.group.allGather(f.stream, wb.gathered + f.w.p.rank * n * f.w.c.hidden * 4, wb.gathered, n * f.w.c.hidden * 4);
        try f.k.mergeRanksRows(f.stream, wb.gathered, out, f.w.c.hidden, n, bf16);
    }

    /// Layers-first pass over up to eight positions, checked against token() by the byte-parity probe.
    pub fn window(f: *Forward, s: *state.State, wb: *batch_mod.Batch, hist: *const history_mod.History, ids: []const u32, tail: Tail) !void {
        const c = f.w.c;
        const p = f.w.p;
        const n = ids.len;
        const base = s.pos;
        if (n == 0 or n > 8 or base + n > s.capacity or s.layers.len != f.w.layers) return error.BadWindowState;
        for (ids) |id| if (id >= c.unpadded_vocab) return error.BadTokenState;
        const m = f.math orelse return error.BatchNeedsPackedWeights;
        var taus: [8]f32 = undefined;
        for (&taus, 0..) |*t, i|
            t.* = 1 + c.log_alpha * @log(@max(1, @as(f32, @floatFromInt(base + i + 1)) / @as(f32, @floatFromInt(c.log_floor))));
        const taus_dev: cuda.DeviceBuffer = .{ .d = f.d, .ptr = wb.taus, .len = 8 * 4, .borrowed = true };
        try taus_dev.upload(0, std.mem.asBytes(&taus));
        const rank_slot = wb.gathered + p.rank * n * c.hidden * 4;
        const table = (try f.w.get("model.llm.embed.weight")).ptr;
        for (ids, 0..) |id, i|
            try f.k.embed(f.stream, table, rank_slot + i * c.hidden * 4, id, p.vocab.start, p.valid_vocab.count, c.hidden);
        try f.gatherRows(wb, n, wb.h, false);
        try f.k.rmsRows(f.stream, wb.h, c.hidden, (try f.w.get("model.llm.embed_norm.weight")).ptr, wb.x, c.hidden, c.hidden, n, c.rms_eps);
        try f.copy(wb.h, wb.x, n * c.hidden * 4);
        var depth: usize = 0;
        for (s.layers, 0..) |ls, i| try f.windowLayer(s, wb, hist, ls, @intCast(i), base, @intCast(n), &depth);
        s.pos += @intCast(n);
        if (f.w.layers != c.layers) return;
        try f.k.residualRows(f.stream, wb.stack, wb.h, (try f.w.get("model.llm.attn_res.output_query")).ptr, wb.mix1, c.hidden, depth, 1, c.rms_eps, n);
        try f.k.rmsRows(f.stream, wb.mix1, c.hidden, (try f.w.get("model.llm.norm.weight")).ptr, wb.x, c.hidden, c.hidden, n, c.rms_eps);
        for (0..n) |i| try f.tapRow(48, base + @as(u32, @intCast(i)), wb.x + i * c.hidden * 4);
        if (tail == .none) return;
        const unembed = (try f.w.get("model.llm.unembed.weight")).lane orelse return error.BatchNeedsPackedWeights;
        if (tail == .last) {
            try f.copy(wb.xrow, wb.x + (n - 1) * c.hidden * 4, c.hidden * 4);
            try m.mm(f.stream, unembed, wb.xrow, 1, wb.logits, false);
            try f.k.softcapRows(f.stream, wb.logits, wb.capped, p.vocab.count, p.valid_vocab.count, c.logits_cap, 1);
        } else {
            try m.mm(f.stream, unembed, wb.x, n, wb.logits, false);
            try f.k.softcapRows(f.stream, wb.logits, wb.capped, p.vocab.count, p.valid_vocab.count, c.logits_cap, n);
        }
    }

    fn windowLayer(f: *Forward, s: *const state.State, wb: *batch_mod.Batch, hist: *const history_mod.History, ls: state.Layer, l: u32, base: u32, n: usize, depth: *usize) !void {
        const c = f.w.c;
        const local = c.local(l);
        if (l % c.residual_block == 5) for (0..n) |i|
            try f.tapRow(l, base + @as(u32, @intCast(i)), wb.h + i * c.hidden * 4);
        if (l == 0) {
            try f.k.rmsRows(f.stream, wb.h, c.hidden, (try f.w.layer(l, "attn_norm.weight")).ptr, wb.norm1, c.hidden, c.hidden, n, c.rms_eps);
        } else if (local) {
            try f.k.residualRows(f.stream, wb.stack, wb.h, (try f.w.query("attn", l - 1)).ptr, wb.mix1, c.hidden, depth.*, 1, c.rms_eps, n);
            try f.k.rmsRows(f.stream, wb.mix1, c.hidden, (try f.w.layer(l, "attn_norm.weight")).ptr, wb.norm1, c.hidden, c.hidden, n, c.rms_eps);
        } else {
            try f.k.residualRows(f.stream, wb.stack, wb.h, (try f.w.query("attn", l - 1)).ptr, wb.mix3, c.hidden, depth.*, 3, c.rms_eps, n);
            const cols = [_]usize{ 0, c.hidden, 2 * c.hidden };
            const names = [_][]const u8{ "q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight" };
            for (cols, names) |col, name|
                try f.k.rmsRows(f.stream, wb.mix3 + col * 4, 3 * c.hidden, (try f.w.layer(l, name)).ptr, wb.norm3 + col * 4, 3 * c.hidden, c.hidden, n, c.rms_eps);
        }
        try f.windowAttention(s, wb, hist, ls, l, base, n, local);
        try f.convSnap(s, wb, hist, ls, l, .a, wb.out, "attn_sconv.weight", wb.conv, true, n, c.hidden);
        const boundary = l % c.residual_block == 0;
        if (boundary) {
            for (0..n) |i|
                try f.copy(wb.stack + (i * 8 + depth.*) * c.hidden * 4, wb.h + i * c.hidden * 4, c.hidden * 4);
            depth.* += 1;
        }
        try f.k.addRows(f.stream, wb.h, wb.conv, wb.mid, c.hidden, n, boundary, false);
        try f.k.residualRows(f.stream, wb.stack, wb.mid, (try f.w.query("mlp", l)).ptr, wb.mix1, c.hidden, depth.*, 1, c.rms_eps, n);
        try f.k.rmsRows(f.stream, wb.mix1, c.hidden, (try f.w.layer(l, "mlp_norm.weight")).ptr, wb.x, c.hidden, c.hidden, n, c.rms_eps);
        try f.windowMoe(wb, l, n);
        try f.convSnap(s, wb, hist, ls, l, .m, wb.out, "mlp_sconv.weight", wb.conv, false, n, c.hidden);
        try f.k.addRows(f.stream, wb.mid, wb.conv, wb.h, c.hidden, n, false, true);
    }

    const ConvSlot = enum { k, v, a, m };

    /// The static attention/mlp short convolution over a window, with each row's prefix history
    fn convSnap(f: *Forward, s: *const state.State, wb: *batch_mod.Batch, hist: *const history_mod.History, ls: state.Layer, l: u32, which: ConvSlot, x: u64, name: []const u8, y: u64, rounded: bool, n: usize, h: usize) !void {
        _ = wb;
        const off: u64 = switch (which) {
            .k => 0,
            .v => @intCast(ls.hist_v - ls.hist_k),
            .a => @intCast(ls.hist_a - ls.hist_k),
            .m => @intCast(ls.hist_m - ls.hist_k),
        };
        const snap = hist.storage.ptr + hist.stride + @as(u64, l) * s.history_bytes + off;
        try f.k.convWindow(f.stream, x, (try f.w.layer(l, name)).ptr, ls.hist_k + off, 0, y, h, 0, false, rounded, n, snap, hist.stride / 4);
    }

    fn windowAttention(f: *Forward, s: *const state.State, wb: *batch_mod.Batch, hist: *const history_mod.History, ls: state.Layer, l: u32, base: u32, n: usize, local: bool) !void {
        const c = f.w.c;
        const m = f.math.?;
        const qi = if (local) wb.norm1 else wb.norm3;
        const ki = if (local) wb.norm1 else wb.norm3 + c.hidden * 4;
        const vi = if (local) wb.norm1 else wb.norm3 + 2 * c.hidden * 4;
        const kv_stride: usize = if (local) c.hidden else 3 * c.hidden;
        try m.mmStrided(f.stream, try f.laneWeight(l, "attn.wq_du.weight"), qi, kv_stride, n, wb.q, true);
        try m.mmStrided(f.stream, try f.laneWeight(l, "attn.wr_du.weight"), qi, kv_stride, n, wb.r, true);
        try m.mmStrided(f.stream, try f.laneWeight(l, "attn.wk_dv.weight"), ki, kv_stride, n, wb.k, true);
        try m.mmStrided(f.stream, try f.laneWeight(l, "attn.wv_dv.weight"), vi, kv_stride, n, wb.v, true);
        try m.mmStrided(f.stream, try f.laneWeight(l, "attn.gate_attn.weight"), qi, kv_stride, n, wb.gate, true);
        try m.mmStrided(f.stream, try f.laneWeight(l, "attn.attn_sink_proj.weight"), qi, kv_stride, n, wb.sink, true);
        if (local) {
            try f.convSnap(s, wb, hist, ls, l, .k, wb.k, "attn.k_sconv.weight", wb.kc, false, n, 256);
            try f.convSnap(s, wb, hist, ls, l, .v, wb.v, "attn.v_sconv.weight", wb.vc, false, n, 256);
        } else {
            try m.mmStrided(f.stream, try f.laneWeight(l, "attn.k_sconv.weight_proj"), ki, kv_stride, n, wb.dyn_k, true);
            try m.mmStrided(f.stream, try f.laneWeight(l, "attn.v_sconv.weight_proj"), vi, kv_stride, n, wb.dyn_v, true);
            try f.dynConv(s, hist, ls, l, .k, wb.k, "attn.k_sconv.static_w", wb.dyn_k, wb.kc, n);
            try f.dynConv(s, hist, ls, l, .v, wb.v, "attn.v_sconv.static_w", wb.dyn_v, wb.vc, n);
        }
        const taus = if (local) wb.taus_one else wb.taus;
        try f.k.qrRows(f.stream, wb.q, wb.r, (try f.w.layer(l, "attn.q_l2_clip.scale")).ptr, (try f.w.layer(l, "attn.rel_l2_clip.scale")).ptr, wb.qo, wb.ro, 12, 128, 16, taus, c.qk_clip, c.rel_clip, c.rms_eps, n);
        try f.k.kvRows(f.stream, wb.kc, wb.vc, ls.k, ls.v, 128, 2, base, s.capacity, c.qk_clip, c.rms_eps, n);
        try f.k.relativeRows(f.stream, wb.ro, (try f.w.layer(l, "attn.rel_logits_proj.proj")).ptr, wb.profile, 12, 16, 1024, n);
        try f.k.scoresRows(f.stream, wb.qo, ls.k, wb.profile, wb.scores, s.capacity, base, n, local);
        try f.k.attendRows(f.stream, wb.scores, wb.sink, ls.v, wb.gate, wb.attn, s.capacity, base, n, local);
        try m.mm(f.stream, try f.laneWeight(l, "attn.wo_ud.weight"), wb.attn, n, wb.gathered + f.w.p.rank * n * c.hidden * 4, true);
        try f.gatherRows(wb, n, wb.out, true);
    }

    fn dynConv(f: *Forward, s: *const state.State, hist: *const history_mod.History, ls: state.Layer, l: u32, which: ConvSlot, x: u64, name: []const u8, dyn: u64, y: u64, n: usize) !void {
        const off: u64 = switch (which) {
            .k => 0,
            .v => @intCast(ls.hist_v - ls.hist_k),
            .a, .m => unreachable,
        };
        const snap = hist.storage.ptr + hist.stride + @as(u64, l) * s.history_bytes + off;
        try f.k.convWindow(f.stream, x, (try f.w.layer(l, name)).ptr, ls.hist_k + off, dyn, y, 256, 128, true, false, n, snap, hist.stride / 4);
    }

    fn windowMoe(f: *Forward, wb: *batch_mod.Batch, l: u32, n: usize) !void {
        const c = f.w.c;
        const m = f.math.?;
        const mx = c.expertFormat(l) == .mxfp8;
        try m.mm(f.stream, try f.laneWeight(l, "mlp.gate.weight"), wb.x, n, wb.router, false);
        try f.k.routeRows(f.stream, wb.router, (try f.w.layer(l, "mlp.gate.bias")).ptr, wb.ids, wb.gammas, 384, 8, c.route_scale, n);
        try f.k.quantNorm(f.stream, wb.x, 0, wb.x_norm, wb.x_amax, 4608, mx, n);
        try f.d.check(f.d.api.cuMemsetD8Async(wb.expert_parts, 0, n * 8 * c.hidden * 4, f.stream.handle), "inkling2 expert zeros");
        // One router readback per layer; collective order stays identical regardless of owned experts.
        try f.stream.synchronize();
        try cuda.memory.download(f.d, wb.ids, std.mem.asBytes(&wb.host_ids));
        try cuda.memory.download(f.d, wb.gammas, std.mem.asBytes(&wb.host_gammas));
        @memset(&wb.counts, 0);
        for (wb.host_ids[0..n], 0..) |row_ids, row| for (row_ids, 0..) |id, pick| {
            if (id < 0 or id >= c.experts or !std.math.isFinite(wb.host_gammas[row][pick])) return error.InvalidRouterOutput;
        };
        for (wb.host_ids[0..n], 0..) |row_ids, row| for (row_ids, 0..) |id, pick| {
            if (!f.w.p.experts.contains(@intCast(id))) continue;
            const local: usize = @intCast(id - @as(i32, @intCast(f.w.p.experts.start)));
            if (wb.counts[local] >= 8) return error.InvalidRouterOutput;
            wb.groups[local][wb.counts[local]] = .{ .row = @intCast(row), .pick = @intCast(pick) };
            wb.counts[local] += 1;
        };
        const meta: cuda.DeviceBuffer = .{ .d = f.d, .ptr = wb.g_meta, .len = 96 * 24 * 4, .borrowed = true };
        for (0..f.w.p.experts.count) |local| {
            const members = wb.groups[local][0..wb.counts[local]];
            for (members, 0..) |mem, g| {
                wb.meta[local][g] = mem.row;
                wb.meta[local][8 + g] = mem.pick;
                wb.meta[local][16 + g] = @bitCast(wb.host_gammas[mem.row][mem.pick]);
            }
        }
        try meta.uploadAsync(0, std.mem.asBytes(&wb.meta), f.stream.handle);
        for (0..f.w.p.experts.count) |local| {
            const members = wb.groups[local][0..wb.counts[local]];
            if (members.len == 0) continue;
            const group_meta = wb.g_meta + local * 24 * 4;
            var amax: u64 = 0;
            var denom: u64 = 0;
            if (!mx) {
                amax = (try f.w.layer(l, "mlp.experts.w13_weight_amax")).ptr + local * 4;
                denom = (try f.w.layer(l, "mlp.experts.w13_weight_outer_scale_denominator")).ptr;
            }
            try m.gatherU16(f.stream, wb.x_norm, wb.g_in16, group_meta, 4608, members.len);
            try m.gatherF32(f.stream, wb.x_amax, wb.g_amax, group_meta, 1, members.len);
            try m.mmBf16(f.stream, try f.expertLane(l, @intCast(local), true), wb.g_in16, members.len, wb.gu, false);
            try m.go(f.stream, .quant_output, .{ .x = 18, .y = @intCast(members.len) }, .{ wb.gu, wb.g_amax, amax, denom, @as(c_int, 4608), @as(c_int, @intCast(members.len)), @as(c_int, @intFromBool(!mx)) });
            try f.k.situRows(f.stream, wb.gu, wb.act, c.intermediate, group_meta + 64, group_meta, members.len);
            try m.gatherF32(f.stream, wb.act, wb.act_g, group_meta, c.intermediate, members.len);
            try f.k.quantNorm(f.stream, wb.act_g, 0, wb.a_norm, wb.g_a_amax, c.intermediate, mx, members.len);
            if (!mx) {
                amax = (try f.w.layer(l, "mlp.experts.w2_weight_amax")).ptr + local * 4;
                denom = (try f.w.layer(l, "mlp.experts.w2_weight_outer_scale_denominator")).ptr;
            }
            try m.mmBf16(f.stream, try f.expertLane(l, @intCast(local), false), wb.a_norm, members.len, wb.g_out, false);
            try m.go(f.stream, .quant_output, .{ .x = 18, .y = @intCast(members.len) }, .{ wb.g_out, wb.g_a_amax, amax, denom, @as(c_int, 4608), @as(c_int, @intCast(members.len)), @as(c_int, @intFromBool(!mx)) });
            try m.scatterF32(f.stream, wb.g_out, wb.expert_parts, group_meta, group_meta + 32, 8, c.hidden, members.len);
        }
        try f.k.mergeRows(f.stream, wb.expert_parts, wb.gathered + f.w.p.rank * n * c.hidden * 4, c.hidden, 8, n, true);
        try f.gatherRows(wb, n, wb.out, true);
    }

    fn expertLane(f: *const Forward, l: u32, local: u32, up: bool) !cuda.qmmf.Weight {
        const mw = try f.w.layer(l, if (up) "mlp.experts.w13_weight" else "mlp.experts.w2_weight");
        const ms = try f.w.layer(l, if (up) "mlp.experts.w13_weight_scale" else "mlp.experts.w2_weight_scale");
        var packed_weight = mw.lane orelse return error.MissingExpertLayout;
        packed_weight.codes = mw.ptr + local * (mw.bytes / mw.dims[0]);
        packed_weight.scales = ms.ptr + local * (ms.bytes / ms.dims[0]);
        return packed_weight;
    }
};
