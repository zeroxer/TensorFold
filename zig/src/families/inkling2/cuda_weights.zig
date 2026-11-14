//! The released text weights, unchanged bytes, selected by whole heads and whole experts for one rank.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("shard.zig");
const schema = @import("weight_schema.zig");

pub const Weight = struct { ptr: u64, dims: [3]usize, rank: u8, bytes: usize, lane: ?cuda.qmmf.Weight = null };
pub const Weights = struct {
    a: std.mem.Allocator,
    c: cfg.Config,
    p: shard.Plan,
    layers: u32,
    entries: std.StringHashMapUnmanaged(Weight) = .empty,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.a);
        var it = w.entries.keyIterator();
        while (it.next()) |name| w.a.free(name.*);
        w.entries.deinit(w.a);
        w.* = undefined;
    }

    pub fn get(w: *const Weights, name: []const u8) !Weight {
        return w.entries.get(name) orelse {
            std.log.err("inkling2 missing loaded tensor {s}", .{name});
            return error.MissingLoadedTensor;
        };
    }
    pub fn layer(w: *const Weights, l: u32, tail: []const u8) !Weight {
        var buf: [200]u8 = undefined;
        return w.get(try std.fmt.bufPrint(&buf, "model.llm.layers.{d}.{s}", .{ l, tail }));
    }
    pub fn query(w: *const Weights, kind: []const u8, index: u32) !Weight {
        var buf: [200]u8 = undefined;
        return w.get(try std.fmt.bufPrint(&buf, "model.llm.attn_res.{s}_queries.{d}", .{ kind, index }));
    }
};

/// `layers < 48` is explicitly a partial-stack development load, never the registered serving model.
pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: cfg.Config, p: shard.Plan, layers: u32) !Weights {
    if (layers == 0) return error.BadLayers;
    return loadPart(a, io, d, dir, c, p, layers, null);
}

/// Shared embeddings and head for a draft-only device check; no target transformer layer is allocated.
pub fn loadBase(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: cfg.Config, p: shard.Plan) !Weights {
    return loadPart(a, io, d, dir, c, p, 0, null);
}

/// One real transformer layer for a bounded numerical probe beside the resident service.
pub fn loadLayer(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: cfg.Config, p: shard.Plan, layer_id: u32) !Weights {
    if (layer_id >= c.layers) return error.BadLayers;
    return loadPart(a, io, d, dir, c, p, 1, layer_id);
}

fn loadPart(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: cfg.Config, p: shard.Plan, layers: u32, only_layer: ?u32) !Weights {
    if (layers > c.layers) return error.BadLayers;
    try c.admitReleased(121, p.world);
    var ck = try core.Checkpoint.openModelPrefix(a, io, dir, "model.llm.");
    defer ck.close();
    var w: Weights = .{ .a = a, .c = c, .p = p, .layers = layers };
    errdefer w.deinit();
    var census: usize = 0;
    for (ck.files.items) |*file| for (file.names.keys()) |name| {
        const t = try ck.get(name);
        var shape: [3]u32 = undefined;
        if (t.rank > shape.len) return error.WrongShape;
        for (t.shape[0..t.rank], 0..) |dim, i| shape[i] = std.math.cast(u32, dim) orelse return error.WrongShape;
        const r = try schema.expected(c, name);
        const dt: core.checkpoint.DType = switch (r.dtype) {
            .bf16 => .bf16,
            .f32 => .f32,
            .u8 => .u8,
            .f8_e4m3 => .f8_e4m3,
        };
        if (dt != t.dtype or !std.mem.eql(u32, shape[0..t.rank], r.dims())) return error.UnexpectedInklingTensor;
        census += 1;
        if (!included(name, layers, only_layer)) continue;
        if (w.entries.contains(name)) return error.DuplicateTensor;
        try w.entries.ensureUnusedCapacity(a, 1);
        try w.buffers.ensureUnusedCapacity(a, 1);
        const key = try a.dupe(u8, name);
        errdefer a.free(key);
        const selection = select(c, p, r, name);
        const b = try upload(a, d, t, selection);
        var dims: [3]usize = @splat(1);
        @memcpy(dims[0..t.rank], t.shape[0..t.rank]);
        if (selection) |s| dims[s.axis] = s.count;
        w.entries.putAssumeCapacityNoClobber(key, .{ .ptr = b.ptr, .dims = dims, .rank = t.rank, .bytes = b.len });
        w.buffers.appendAssumeCapacity(b);
        w.bytes += b.len;
        if (b.len >= 64 << 20) ck.releaseExcept(&.{});
    };
    if (census != 1348 or ck.unused() != 0) return error.UnexpectedTensorCensus;
    std.log.info("inkling2 rank {d}: {d}/{d} layers, {d} tensor shards, {d} bytes (text only)", .{ p.rank, layers, c.layers, w.entries.count(), w.bytes });
    return w;
}

fn included(name: []const u8, layers: u32, only_layer: ?u32) bool {
    if (only_layer != null and (std.mem.eql(u8, name, "model.llm.embed.weight") or
        std.mem.eql(u8, name, "model.llm.unembed.weight"))) return false;
    if (layers > 0 and layers < 48 and std.mem.eql(u8, name, "model.llm.unembed.weight")) return false;
    const prefix = "model.llm.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return true;
    const rest = name[prefix.len..];
    const at = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
    const n = std.fmt.parseInt(u32, rest[0..at], 10) catch return false;
    return if (only_layer) |l| n == l else n < layers;
}

pub const Selection = struct { axis: usize, start: usize, count: usize, valid: usize };
fn select(c: cfg.Config, p: shard.Plan, r: schema.Requirement, name: []const u8) ?Selection {
    const span: shard.Span = switch (r.place) {
        .replicated => return null,
        .experts => p.experts,
        .vocabulary => return .{ .axis = 0, .start = p.vocab.start, .count = p.vocab.count, .valid = p.valid_vocab.count },
        .q_heads => .{ .start = p.q.start * (r.shape[0] / c.heads), .count = p.q.count * (r.shape[0] / c.heads) },
        .kv_heads => if (std.mem.endsWith(u8, name, ".static_w")) p.kvColumns(c) else .{ .start = p.kv.start * (r.shape[0] / c.kv_heads), .count = p.kv.count * (r.shape[0] / c.kv_heads) },
        .output_columns => p.qColumns(c),
    };
    const axis: usize = if (r.place == .output_columns or std.mem.endsWith(u8, name, ".static_w")) 1 else 0;
    return .{ .axis = axis, .start = span.start, .count = span.count, .valid = span.count };
}

pub fn upload(a: std.mem.Allocator, d: *const cuda.Driver, t: core.checkpoint.Tensor, s_: ?Selection) !cuda.DeviceBuffer {
    const s = s_ orelse return cuda.DeviceBuffer.fromHost(d, t.bytes);
    var outer: usize = 1;
    var inner: usize = t.dtype.size();
    for (t.shape[0..s.axis]) |dim| outer *= dim;
    for (t.shape[s.axis + 1 .. t.rank]) |dim| inner *= dim;
    if (s.start + s.valid > t.shape[s.axis] or s.valid > s.count) return error.BadShardSelection;
    const bytes = outer * s.count * inner;
    if (outer == 1 and s.count == s.valid)
        return cuda.DeviceBuffer.fromHost(d, t.bytes[s.start * inner ..][0..bytes]);
    const staging = try a.alloc(u8, bytes);
    defer a.free(staging);
    @memset(staging, 0);
    for (0..outer) |row| {
        const src = (row * t.shape[s.axis] + s.start) * inner;
        @memcpy(staging[row * s.count * inner ..][0 .. s.valid * inner], t.bytes[src..][0 .. s.valid * inner]);
    }
    return cuda.DeviceBuffer.fromHost(d, staging);
}
