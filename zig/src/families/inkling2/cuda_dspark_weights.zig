//! DSpark's stored BF16 tensors and owned, losslessly packed projections; original files remain read-only.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("dspark_config.zig");
const schema = @import("dspark_schema.zig");
const target = @import("cuda_weights.zig");
const math = @import("cuda_lane.zig");

pub const Weights = struct {
    c: cfg.Config,
    owner: target.Weights,
    matrices: std.StringHashMapUnmanaged(cuda.qmmf.Weight) = .empty,
    alpha: u64 = 0,
    head: cuda.qmmf.Weight = undefined,
    fold: cuda.qmmf.Weight = undefined,
    fold_bias: u64 = 0,
    context: cuda.qmmf.Weight = undefined,
    previous: cuda.qmmf.Weight = undefined,

    pub fn deinit(w: *Weights) void {
        w.matrices.deinit(w.owner.a);
        w.owner.deinit();
        w.* = undefined;
    }
    pub fn get(w: *const Weights, name: []const u8) !u64 {
        return (try w.owner.get(name)).ptr;
    }
    pub fn matrix(w: *const Weights, name: []const u8) !cuda.qmmf.Weight {
        return w.matrices.get(name) orelse error.MissingDraftMatrix;
    }
    pub fn layer(w: *const Weights, l: usize, tail: []const u8) !u64 {
        var buf: [160]u8 = undefined;
        return w.get(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ l, tail }));
    }
    pub fn layerMatrix(w: *const Weights, l: usize, tail: []const u8) !cuda.qmmf.Weight {
        var buf: [160]u8 = undefined;
        return w.matrix(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ l, tail }));
    }
    fn own(w: *Weights, b: cuda.DeviceBuffer) !u64 {
        errdefer {
            var lost = b;
            lost.free();
        }
        try w.owner.buffers.append(w.owner.a, b);
        w.owner.bytes += b.len;
        return b.ptr;
    }
    fn alloc(w: *Weights, d: *const cuda.Driver, bytes: usize) !u64 {
        return w.own(try cuda.DeviceBuffer.alloc(d, bytes));
    }
};

pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, m: *const math.Math, dir: []const u8, base: *const target.Weights) !Weights {
    const c = try cfg.Config.read(a, io, dir);
    try c.admitReleased(121, base.p.world);
    var ck: core.Checkpoint = .{ .gpa = a, .io = io };
    defer ck.close();
    try ck.add(dir, "mtp.safetensors");
    var w: Weights = .{ .c = c, .owner = .{ .a = a, .c = base.c, .p = base.p, .layers = 0 } };
    errdefer w.deinit();
    const spans = try schema.rankSpans(c, base.p.rank, base.p.world);
    var count: usize = 0;
    for (ck.files.items) |*file| for (file.names.keys()) |name| {
        const t = try ck.get(name);
        const r = try schema.expected(c, base.c, name);
        if (t.dtype != .bf16 or t.rank != r.rank) return error.UnexpectedDraftTensor;
        for (t.shape[0..t.rank], r.dims()) |x, y| if (x != y) return error.UnexpectedDraftTensor;
        const sel: ?target.Selection = switch (r.place) {
            .replicated => null,
            .q_heads => .{ .axis = 0, .start = spans.q.start * c.head_dim, .count = spans.q.count * c.head_dim, .valid = spans.q.count * c.head_dim },
            .kv_heads => .{ .axis = 0, .start = spans.kv.start * c.head_dim, .count = spans.kv.count * c.head_dim, .valid = spans.kv.count * c.head_dim },
            .output_columns => .{ .axis = 1, .start = spans.q.start * c.head_dim, .count = spans.q.count * c.head_dim, .valid = spans.q.count * c.head_dim },
            .intermediate_rows => .{ .axis = 0, .start = spans.intermediate.start, .count = spans.intermediate.count, .valid = spans.intermediate.count },
            .intermediate_columns => .{ .axis = 1, .start = spans.intermediate.start, .count = spans.intermediate.count, .valid = spans.intermediate.count },
        };
        var dims: [3]usize = @splat(1);
        @memcpy(dims[0..t.rank], t.shape[0..t.rank]);
        if (sel) |s| dims[s.axis] = s.count;
        var raw = try target.upload(a, d, t, sel);
        errdefer raw.free();
        const matrix = std.mem.startsWith(u8, name, "layers.") and
            (std.mem.endsWith(u8, name, "_proj.weight") or std.mem.endsWith(u8, name, "_proj_ctx.weight"));
        var packed_weight: ?cuda.qmmf.Weight = null;
        if (matrix) {
            const made = try m.bf16Pack(d, stream, raw.ptr, dims[0], dims[1]);
            raw.free();
            raw = made.storage;
            packed_weight = made.weight;
        }
        try w.owner.entries.ensureUnusedCapacity(a, 1);
        try w.owner.buffers.ensureUnusedCapacity(a, 1);
        try w.matrices.ensureUnusedCapacity(a, 1);
        const key = try a.dupe(u8, name);
        w.owner.entries.putAssumeCapacityNoClobber(key, .{ .ptr = raw.ptr, .dims = dims, .rank = t.rank, .bytes = raw.len });
        w.owner.buffers.appendAssumeCapacity(raw);
        w.owner.bytes += raw.len;
        if (packed_weight) |p| w.matrices.putAssumeCapacityNoClobber(key, p);
        count += 1;
    };
    if (count != 84 or ck.unused() != 0) return error.UnexpectedDraftCensus;
    w.alpha = try w.alloc(d, 6 * 8 * 4);
    try m.go(stream, .fuse_alpha, .{ .x = 6 }, .{ try w.get("fuse_weight"), w.alpha, @as(c_int, 6), @as(c_int, 8) });
    try stream.synchronize();
    const head = try base.get("model.llm.unembed.weight");
    if (head.lane) |packed_head| w.head = packed_head else {
        const hp = try m.bf16Pack(d, stream, head.ptr, head.dims[0], head.dims[1]);
        _ = try w.own(hp.storage);
        w.head = hp.weight;
    }
    const rn = try ck.get("markov_head.rnn_proj.weight");
    inline for (.{ true, false }) |context| {
        const first: usize = if (context) 512 else 0;
        const width: usize = if (context) 9216 else 512;
        var raw = try target.upload(a, d, rn, .{ .axis = 1, .start = first, .count = width, .valid = width });
        defer raw.free();
        const pk = try m.bf16Pack(d, stream, raw.ptr, 768, width);
        _ = try w.own(pk.storage);
        if (context) w.context = pk.weight else w.previous = pk.weight;
    }
    var folded = try cuda.DeviceBuffer.alloc(d, base.p.vocab.count * 256 * 2);
    defer folded.free();
    w.fold_bias = try w.alloc(d, base.p.vocab.count * 2);
    try m.go(stream, .fold, .{ .x = base.p.vocab.count }, .{ head.ptr, try w.get("markov_head.hidden_out.weight"), try w.get("markov_head.hidden_out.bias"), folded.ptr, w.fold_bias, math.int(base.p.vocab.count), @as(c_int, 4608), @as(c_int, 256) });
    try stream.synchronize();
    const fp = try m.bf16Pack(d, stream, folded.ptr, base.p.vocab.count, 256);
    _ = try w.own(fp.storage);
    w.fold = fp.weight;
    std.log.info("inkling2 DSpark rank {d}: 84 stored tensors, {d} owned bytes, eight target taps", .{ base.p.rank, w.owner.bytes });
    return w;
}
