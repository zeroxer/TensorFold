//! One admitted stream's target and DSpark device state, driven by TensorFold's shared lane scheduler.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const forward = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const kernels = @import("cuda_kernels.zig");
const lane = @import("cuda_lane.zig");
const dspark = @import("cuda_dspark.zig");
const history = @import("cuda_history.zig");
const batch_mod = @import("cuda_batch.zig");
const link = @import("cuda_link.zig");

pub const Runtime = struct {
    a: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    group: *const cuda.tp_group.Group,
    w: *const weights.Weights,
    k: *const kernels.Kernels,
    math: *const lane.Math,
    draft: *dspark.Model,
    seq: state.State,
    scratch: state.Scratch,
    taps: cuda.DeviceBuffer,
    chosen: cuda.DeviceBuffer,
    batch: batch_mod.Batch,
    hist: history.History,
    draft_cache: ?dspark.Cache = null,
    link: link.Link,
    last: u32 = 0,
    verified: [8]u32 = undefined,
    target_rows: u64 = 0,
    verified_rows: u64 = 0,
    verify_windows: u64 = 0,
    accepted_rows: u64 = 0,
    rejected_rows: u64 = 0,

    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, g: *const cuda.tp_group.Group, w: *const weights.Weights, k: *const kernels.Kernels, m: *const lane.Math, draft: *dspark.Model, cap: u32) !Runtime {
        if (w.layers != 48 or cap > 1032) return error.BadRuntimeGeometry;
        var seq = try state.State.init(a, d, w.c, w.p, 48, cap);
        errdefer seq.deinit();
        var scratch = try state.Scratch.init(d, cap);
        errdefer scratch.deinit();
        var taps = try cuda.DeviceBuffer.alloc(d, cap * 8 * 4608 * 4);
        errdefer taps.free();
        var chosen = try cuda.DeviceBuffer.alloc(d, 8 * 4);
        errdefer chosen.free();
        var hist = try history.History.init(d, &seq);
        errdefer hist.deinit();
        var batches = try batch_mod.Batch.init(d, cap);
        errdefer batches.deinit();
        const wire = try link.Link.init(a, d, g, s);
        return .{ .a = a, .d = d, .stream = s, .group = g, .w = w, .k = k, .math = m, .draft = draft, .seq = seq, .scratch = scratch, .taps = taps, .chosen = chosen, .batch = batches, .hist = hist, .link = wire };
    }
    pub fn deinit(r: *Runtime) void {
        r.stream.synchronize() catch {};
        if (r.draft_cache) |*c| c.deinit();
        r.link.deinit();
        r.hist.deinit();
        r.chosen.free();
        r.batch.deinit();
        r.taps.free();
        r.scratch.deinit();
        r.seq.deinit();
        r.* = undefined;
    }
    fn fwd(r: *Runtime) forward.Forward {
        return .{ .d = r.d, .stream = r.stream, .k = r.k, .w = r.w, .b = &r.scratch, .group = r.group, .math = r.math, .features = r.taps.ptr };
    }
    /// Queue one greedy pick a row of the batched capped logits; one gather merges every row.
    fn queuePicks(r: *Runtime, first: usize, rows: usize) !void {
        const p = r.w.p;
        const span = rows * 8;
        for (0..rows) |i|
            try r.math.go(r.stream, .pick, .{ .x = 1 }, .{ r.batch.capped + (first + i) * p.vocab.count * 4, r.batch.picks + p.rank * span + i * 8, lane.int(p.valid_vocab.count), lane.int(p.vocab.start) });
        try r.group.allGather(r.stream, r.batch.picks + p.rank * span, r.batch.picks, span);
        try r.math.pickMergeRows(r.stream, r.batch.picks, r.chosen.ptr, 4, rows);
    }
    fn pick(r: *Runtime) !u32 {
        try r.queuePicks(0, 1);
        try r.stream.synchronize();
        var id: u32 = undefined;
        try r.chosen.download(0, std.mem.asBytes(&id));
        if (id >= 200058) return error.InvalidSample;
        return id;
    }
    pub fn call(r: *Runtime, kind: link.Kind, a: u32, b: u32, ids: []const u32) !void {
        if (r.group.rank != 0) return error.NotLeader;
        try r.link.send(kind, a, b, ids);
        try r.execute(kind, a, b, ids);
    }
    pub fn follow(r: *Runtime) !void {
        while (true) {
            const cmd = try r.link.receive();
            const kind = std.enums.fromInt(link.Kind, cmd.h.kind) orelse return error.BadCommand;
            try r.execute(kind, cmd.h.a, cmd.h.b, cmd.ids);
            if (kind == .stop) return;
        }
    }
    fn execute(r: *Runtime, kind: link.Kind, a: u32, b: u32, ids: []const u32) !void {
        switch (kind) {
            .reset => {
                try r.seq.reset();
                r.last = 0;
                r.hist.rows = 0;
            },
            .prefill => {
                if (ids.len == 0 or ids.len >= r.seq.capacity) return error.BadPrompt;
                var f = r.fwd();
                var at: usize = 0;
                while (at < ids.len) {
                    const n = @min(8, ids.len - at);
                    const tail: forward.Forward.Tail = if (at + n == ids.len) .last else .none;
                    try f.window(&r.seq, &r.batch, &r.hist, ids[at..][0..n], tail);
                    r.target_rows += n;
                    at += n;
                }
                r.last = try r.pick();
            },
            .verify => {
                if (ids.len == 0 or ids.len > 8 or r.seq.pos + ids.len > r.seq.capacity) return error.BadWindow;
                try r.hist.begin(&r.seq, r.stream);
                var f = r.fwd();
                try f.window(&r.seq, &r.batch, &r.hist, ids, .all);
                r.hist.rows = @intCast(ids.len);
                try r.queuePicks(0, ids.len);
                r.target_rows += ids.len;
                try r.stream.synchronize();
                try r.chosen.download(0, std.mem.sliceAsBytes(r.verified[0..ids.len]));
                for (r.verified[0..ids.len]) |id| if (id >= 200058) return error.InvalidSample;
                r.verified_rows += ids.len;
                r.verify_windows += 1;
                r.accepted_rows += ids.len;
                r.last = r.verified[ids.len - 1];
            },
            .keep => {
                const offered = r.hist.rows;
                try r.hist.keep(&r.seq, r.stream, a);
                r.accepted_rows -= offered - a;
                r.rejected_rows += offered - a;
            },
            .draft_open => {
                if (r.draft_cache != null) return error.DraftAlreadyOpen;
                r.draft_cache = try dspark.Cache.init(r.d, r.seq.capacity);
            },
            .absorb => {
                const c = if (r.draft_cache) |*c| c else return error.NoDraftCache;
                if (a + b > r.seq.pos) return error.UncommittedFeatures;
                try r.draft.absorb(c, r.taps.ptr + a * 8 * 4608 * 4, a, b);
            },
            .hold => {
                const c = if (r.draft_cache) |*c| c else return error.NoDraftCache;
                if (ids.len != 1) return error.BadDraftAnchor;
                try r.draft.hold(c, ids[0], a, b);
            },
            .draft_close => {
                if (r.draft_cache) |*c| c.deinit();
                r.draft_cache = null;
            },
            .stop => cuda.tp_watch.ended(),
        }
    }
};
