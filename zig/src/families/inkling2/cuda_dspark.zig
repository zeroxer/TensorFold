//! Native DSpark block proposal: target-feature context, five dense layers, then the seven-step Markov walk.
const std = @import("std");
const cuda = @import("cuda");
const tw = @import("cuda_weights.zig");
const dw = @import("cuda_dspark_weights.zig");
const tk = @import("cuda_kernels.zig");
const lane = @import("cuda_lane.zig");
const images = @import("cuda_images.zig");
const H = 4608;
const Q = 1536;
const KV = 256;

pub const Cache = struct {
    storage: cuda.DeviceBuffer,
    capacity: u32,
    rows: u32 = 0,
    held: [7]u32 = undefined,
    held_n: u32 = 0,
    pub fn init(d: *const cuda.Driver, capacity: u32) !Cache {
        if (capacity < 8 or capacity > 1032) return error.BadDraftCapacity;
        var storage = try cuda.DeviceBuffer.alloc(d, 5 * 2 * capacity * KV * 4 + H * 4);
        errdefer storage.free();
        try storage.fill8(0, null);
        try d.synchronize();
        return .{ .storage = storage, .capacity = capacity };
    }
    pub fn deinit(c: *Cache) void {
        c.storage.free();
        c.* = undefined;
    }
    pub fn key(c: *const Cache, l: usize) u64 {
        return c.storage.ptr + l * 2 * c.capacity * KV * 4;
    }
    pub fn value(c: *const Cache, l: usize) u64 {
        return c.key(l) + c.capacity * KV * 4;
    }
    pub fn recurrent(c: *const Cache) u64 {
        return c.storage.ptr + 5 * 2 * c.capacity * KV * 4;
    }
};

const Op = enum { draft_mix, draft_rope, draft_silu, draft_add_norm, draft_attn, markov_gate, draft_softcap };
const Scratch = struct {
    storage: cuda.DeviceBuffer,
    anchor: u64,
    embed: u64,
    res0: u64,
    res1: u64,
    norm: u64,
    mix: u64,
    q: u64,
    k: u64,
    v: u64,
    attn: u64,
    gathered: u64,
    reduced: u64,
    gate: u64,
    up: u64,
    act: u64,
    base: u64,
    bias: u64,
    logits: u64,
    latent: u64,
    state0: u64,
    state1: u64,
    context: u64,
    raw: u64,
    context_input: u64,
    state_input: u64,
    picks: u64,
    walk: u64,
    zero: u64,
    fn init(d: *const cuda.Driver, n: usize) !Scratch {
        const fields = .{
            .{ "anchor", H },        .{ "embed", 8 * H }, .{ "res0", 8 * H }, .{ "res1", 8 * H },      .{ "norm", 8 * H },         .{ "mix", 8 * H },
            .{ "q", 8 * Q },         .{ "k", 8 * KV },    .{ "v", 8 * KV },   .{ "attn", 8 * Q },      .{ "gathered", 4 * 8 * H }, .{ "reduced", 8 * H },
            .{ "gate", 8 * H },      .{ "up", 8 * H },    .{ "act", 8 * H },  .{ "base", 7 * n },      .{ "bias", n },             .{ "logits", n },
            .{ "latent", 256 },      .{ "state0", 256 },  .{ "state1", 256 }, .{ "context", 7 * 768 }, .{ "raw", 768 },            .{ "context_input", 7 * 9216 },
            .{ "state_input", 512 }, .{ "picks", 8 },     .{ "walk", 8 },     .{ "zero", 8 * H },
        };
        var count: usize = 0;
        inline for (fields) |f| count += f[1];
        var b: Scratch = undefined;
        b.storage = try cuda.DeviceBuffer.alloc(d, count * 4);
        errdefer b.storage.free();
        try b.storage.fill8(0, null);
        try d.synchronize();
        var at = b.storage.ptr;
        inline for (fields) |f| {
            @field(b, f[0]) = at;
            at += f[1] * 4;
        }
        return b;
    }
};

pub const Model = struct {
    a: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    group: *const cuda.tp_group.Group,
    target: *const tw.Weights,
    target_k: *const tk.Kernels,
    math: *const lane.Math,
    w: dw.Weights,
    module: cuda.Module,
    functions: [7]cuda.Function,
    rope: cuda.DeviceBuffer,
    b: Scratch,
    blocks: u64 = 0,
    proposed: u64 = 0,

    pub fn init(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, group: *const cuda.tp_group.Group, target: *const tw.Weights, target_k: *const tk.Kernels, math: *const lane.Math, dir: []const u8, image_dir: []const u8, capacity: u32) !Model {
        const code = try images.read(a, io, image_dir, "inkling2_draft");
        defer a.free(code);
        var module = try cuda.Module.load(d, code);
        errdefer module.unload();
        var fns: [7]cuda.Function = undefined;
        inline for (@typeInfo(Op).@"enum".field_names, 0..) |name, i| fns[i] = try module.function("tf_ink2_" ++ name);
        var w = try dw.load(a, io, d, stream, math, dir, target);
        errdefer w.deinit();
        const table = try a.alloc(f32, capacity * 128);
        defer a.free(table);
        for (0..capacity) |p| for (0..64) |i| {
            const inv: f32 = 1 / std.math.pow(f32, w.c.rope_theta, @as(f32, @floatFromInt(2 * i)) / 128);
            const angle: @TypeOf(inv) = @as(f32, @floatFromInt(p)) * inv;
            table[p * 128 + i] = @cos(angle);
            table[p * 128 + 64 + i] = @sin(angle);
        };
        var rope = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(table));
        errdefer rope.free();
        const b = try Scratch.init(d, target.p.vocab.count);
        return .{ .a = a, .d = d, .stream = stream, .group = group, .target = target, .target_k = target_k, .math = math, .w = w, .module = module, .functions = fns, .rope = rope, .b = b };
    }
    pub fn deinit(m: *Model) void {
        m.stream.synchronize() catch {};
        m.b.storage.free();
        m.rope.free();
        m.w.deinit();
        m.module.unload();
        m.* = undefined;
    }
    fn go(m: *const Model, op: Op, grid: cuda.Dim3, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        try cuda.launch.launch(m.functions[@backingInt(op)], .{ .grid = grid, .block = .{ .x = 256 } }, m.stream, &a);
    }
    fn copy(m: *const Model, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, m.stream.handle), "DSpark copy");
    }
    fn mix(m: *const Model, features: u64, row: usize, n: usize, out: u64, norm: u64) !void {
        try m.go(.draft_mix, .{ .x = @intCast(n) }, .{ features, m.w.alpha + row * 8 * 4, norm, out, @as(c_int, H), @as(c_int, 8), lane.int(n), m.w.c.eps });
    }
    fn ropeRows(m: *const Model, x: u64, norm: u64, n: usize, heads: usize, pos: usize) !void {
        try m.go(.draft_rope, .{ .x = @intCast(heads), .y = @intCast(n) }, .{ x, norm, m.rope.ptr, lane.int(n), lane.int(heads), @as(c_int, 128), lane.int(pos), @as(c_int, 1), m.w.c.eps });
    }
    fn matrix(m: *const Model, l: usize, name: []const u8, x: u64, rows: usize, out: u64) !void {
        try m.math.mm(m.stream, try m.w.layerMatrix(l, name), x, rows, out, true);
    }
    fn slot(m: *const Model, rows: usize) u64 {
        return m.b.gathered + m.target.p.rank * rows * H * 4;
    }
    fn gather(m: *const Model, rows: usize) !void {
        try m.group.allGather(m.stream, m.slot(rows), m.b.gathered, rows * H * 4);
        // The existing merge primitive is elementwise; each row retains the same rank-order FP32 sum.
        var a: cuda.Args = .{};
        for ([_]u64{ m.b.gathered, m.b.reduced }) |v| a.add(v);
        a.add(lane.int(rows * H));
        a.add(@as(c_int, 4));
        a.add(@as(c_int, 1));
        try cuda.launch.launch(m.target_k.functions[@backingInt(tk.Op.merge)], .{ .grid = .{ .x = lane.blocks(rows * H) }, .block = .{ .x = 256 } }, m.stream, &a);
    }
    fn addNorm(m: *const Model, res: u64, next: u64, norm: u64) !void {
        try m.go(.draft_add_norm, .{ .x = 8 }, .{ res, m.b.reduced, norm, next, m.b.norm, @as(c_int, H), @as(c_int, 8), m.w.c.eps });
    }

    pub fn absorb(m: *Model, c: *Cache, features: u64, start: u32, rows: u32) !void {
        if (start != c.rows or start + rows > c.capacity or rows == 0) return error.BadDraftContext;
        var done: u32 = 0;
        while (done < rows) {
            const n: u32 = @min(rows - done, 8);
            const src = features + done * 8 * H * 4;
            for (0..5) |l| {
                try m.mix(src, l, n, m.b.mix, try m.w.layer(l, "dflare_norm.weight"));
                try m.matrix(l, "self_attn.k_proj_ctx.weight", m.b.mix, n, m.b.k);
                try m.matrix(l, "self_attn.v_proj_ctx.weight", m.b.mix, n, m.b.v);
                try m.ropeRows(m.b.k, try m.w.layer(l, "self_attn.k_norm_ctx.weight"), n, 2, start + done);
                try m.copy(c.key(l) + (start + done) * KV * 4, m.b.k, n * KV * 4);
                try m.copy(c.value(l) + (start + done) * KV * 4, m.b.v, n * KV * 4);
            }
            done += n;
        }
        try m.mix(features + (rows - 1) * 8 * H * 4, 5, 1, c.recurrent(), try m.w.get("markov_head.dflare_norm.weight"));
        c.rows = start + rows;
        c.held_n = 0;
    }

    pub fn hold(m: *Model, c: *Cache, anchor: u32, position: u32, depth: u32) !void {
        if (depth == 0 or depth > 7 or position != c.rows or position + 8 > c.capacity or anchor >= 200058) return error.BadDraftBlock;
        const b = &m.b;
        const p = m.target.p;
        try m.target_k.embed(m.stream, (try m.target.get("model.llm.embed.weight")).ptr, m.slot(1), anchor, p.vocab.start, p.valid_vocab.count, H);
        try m.gather(1);
        try m.target_k.rms(m.stream, b.reduced, (try m.target.get("model.llm.embed_norm.weight")).ptr, b.anchor, H, m.w.c.eps);
        try m.copy(b.res0, b.anchor, H * 4);
        for (1..8) |row| try m.math.convert(m.stream, try m.w.get("mask_embed"), b.res0 + row * H * 4, H, false);
        for (0..8) |row| try m.target_k.rms(m.stream, b.res0 + row * H * 4, try m.w.layer(0, "input_layernorm.weight"), b.norm + row * H * 4, H, m.w.c.eps);
        for (0..5) |l| {
            try m.matrix(l, "self_attn.q_proj.weight", b.norm, 8, b.q);
            try m.matrix(l, "self_attn.k_proj.weight", b.norm, 8, b.k);
            try m.matrix(l, "self_attn.v_proj.weight", b.norm, 8, b.v);
            try m.ropeRows(b.q, try m.w.layer(l, "self_attn.q_norm.weight"), 8, 12, position);
            try m.ropeRows(b.k, try m.w.layer(l, "self_attn.k_norm.weight"), 8, 2, position);
            try m.go(.draft_attn, .{ .x = 12, .y = 8 }, .{ b.q, c.key(l), c.value(l), b.k, b.v, b.attn, @as(c_int, 12), @as(c_int, 2), @as(c_int, 128), @as(c_int, 8), lane.int(position), @as(c_int, if (m.w.c.local(@intCast(l))) 1024 else 0) });
            try m.matrix(l, "self_attn.o_proj.weight", b.attn, 8, m.slot(8));
            try m.gather(8);
            try m.addNorm(b.res0, b.res1, try m.w.layer(l, "post_attention_layernorm.weight"));
            try m.matrix(l, "mlp.gate_proj.weight", b.norm, 8, b.gate);
            try m.matrix(l, "mlp.up_proj.weight", b.norm, 8, b.up);
            try m.go(.draft_silu, .{ .x = lane.blocks(H), .y = 8 }, .{ b.gate, b.up, b.act, @as(c_int, H), @as(c_int, 8) });
            try m.matrix(l, "mlp.down_proj.weight", b.act, 8, m.slot(8));
            try m.gather(8);
            try m.addNorm(b.res1, b.res0, if (l == 4) try m.w.get("norm.weight") else try m.w.layer(l + 1, "input_layernorm.weight"));
        }
        try m.math.mm(m.stream, m.w.head, b.norm + H * 4, 7, b.base, true);
        try m.math.go(m.stream, .rnn_context_input, .{ .x = lane.blocks(2 * H), .y = 7 }, .{ b.norm, c.recurrent(), b.context_input, @as(c_int, H), @as(c_int, 7) });
        try m.math.mmBf16(m.stream, m.w.context, b.context_input, 7, b.context, false);
        try m.math.go(m.stream, .add_bias, .{ .x = 3, .y = 7 }, .{ b.context, try m.w.get("markov_head.rnn_proj.bias"), b.context, @as(c_int, 768), @as(c_int, 7), @as(c_int, 0) });
        try m.d.check(m.d.api.cuMemsetD8Async(b.state0, 0, 256 * 4, m.stream.handle), "DSpark Markov reset");
        const view: cuda.DeviceBuffer = .{ .d = m.d, .ptr = b.walk, .len = 8 * 4, .borrowed = true };
        try view.upload(0, std.mem.asBytes(&anchor));
        var old = b.state0;
        var next = b.state1;
        for (0..depth) |j| {
            try m.math.go(m.stream, .rnn_state_input, .{ .x = 2 }, .{ old, try m.w.get("markov_head.markov_w1.weight"), b.walk + j * 4, b.state_input, @as(c_int, 256), @as(c_int, 201024) });
            try m.math.mmBf16(m.stream, m.w.previous, b.state_input, 1, b.raw, false);
            try m.math.go(m.stream, .add_float, .{ .x = 3 }, .{ b.raw, b.context + j * 768 * 4, b.raw, @as(c_int, 768) });
            try m.go(.markov_gate, .{ .x = 1 }, .{ b.raw, old, next, b.latent, @as(c_int, 256) });
            try m.math.mm(m.stream, m.w.fold, b.latent, 1, b.bias, false);
            try m.math.go(m.stream, .add_bias, .{ .x = lane.blocks(p.vocab.count) }, .{ b.bias, m.w.fold_bias, b.bias, lane.int(p.vocab.count), @as(c_int, 1), @as(c_int, 1) });
            try m.go(.draft_softcap, .{ .x = lane.blocks(p.vocab.count) }, .{ b.base + j * p.vocab.count * 4, b.bias, b.logits, lane.int(p.vocab.count), lane.int(p.valid_vocab.count), @as(f32, 30) });
            try m.math.go(m.stream, .pick, .{ .x = 1 }, .{ b.logits, b.picks + p.rank * 8, lane.int(p.valid_vocab.count), lane.int(p.vocab.start) });
            try m.group.allGather(m.stream, b.picks + p.rank * 8, b.picks, 8);
            try m.math.go(m.stream, .pick_merge, .{ .x = 1 }, .{ b.picks, b.walk, @as(c_int, 4), lane.int(j + 1) });
            const swap = old;
            old = next;
            next = swap;
        }
        try m.stream.synchronize();
        try view.download(4, std.mem.sliceAsBytes(c.held[0..depth]));
        for (c.held[0..depth]) |id| if (id >= 200058) return error.InvalidDraftToken;
        c.held_n = depth;
        m.blocks += 1;
        m.proposed += depth;
    }
};
