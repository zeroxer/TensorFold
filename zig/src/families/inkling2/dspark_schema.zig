//! DSpark's 84 stored draft tensors and their TP4 placement; this admits metadata, not a drafter.
const std = @import("std");
const cfg = @import("config.zig");
const draft = @import("dspark_config.zig");
const shard = @import("shard.zig");

pub const DType = enum { bf16 };
pub const Place = enum { replicated, q_heads, kv_heads, output_columns, intermediate_rows, intermediate_columns };

pub const Requirement = struct {
    dtype: DType,
    shape: [3]u32,
    rank: u8,
    place: Place,
    pub fn dims(r: *const Requirement) []const u32 {
        return r.shape[0..r.rank];
    }
};

fn req(dtype: DType, shape: []const u32, place: Place) Requirement {
    var r: Requirement = .{ .dtype = dtype, .shape = @splat(1), .rank = @intCast(shape.len), .place = place };
    @memcpy(r.shape[0..shape.len], shape);
    return r;
}
fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn expected(m: draft.Config, c: cfg.Config, name: []const u8) !Requirement {
    if (m.host_layers != c.layers) return error.HostDraftMismatch;
    const H = m.hidden;
    const R = m.markov_rank;
    if (eq(name, "mask_embed") or eq(name, "norm.weight")) return req(.bf16, &.{H}, .replicated);
    // fuse_weight is the released 6x8 block; its composition belongs to the drafter's ABI.
    if (eq(name, "fuse_weight")) return req(.bf16, &.{ 6, 8 }, .replicated);
    if (eq(name, "markov_head.dflare_norm.weight")) return req(.bf16, &.{H}, .replicated);
    if (eq(name, "markov_head.hidden_out.weight")) return req(.bf16, &.{ H, R }, .replicated);
    if (eq(name, "markov_head.hidden_out.bias")) return req(.bf16, &.{H}, .replicated);
    // markov_w1 stays replicated: the previous token can come from any vocabulary rank's shard.
    if (eq(name, "markov_head.markov_w1.weight")) return req(.bf16, &.{ c.vocab, R }, .replicated);
    // The released rnn gate widths 768x9728 are checkpoint constants no parsed field derives.
    if (eq(name, "markov_head.rnn_proj.weight")) return req(.bf16, &.{ 768, 9728 }, .replicated);
    if (eq(name, "markov_head.rnn_proj.bias")) return req(.bf16, &.{768}, .replicated);
    if (!std.mem.startsWith(u8, name, "layers.")) return error.UnknownTensor;
    const rest = name["layers.".len..];
    const split = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnknownTensor;
    const layer = std.fmt.parseInt(u32, rest[0..split], 10) catch return error.UnknownTensor;
    if (layer >= m.layers) return error.UnknownTensor;
    const tail = rest[split + 1 ..];
    const Q = m.heads * m.head_dim;
    const KV = m.kv_heads * m.head_dim;
    const I = m.intermediate;
    for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight", "dflare_norm.weight" }) |k|
        if (eq(tail, k)) return req(.bf16, &.{H}, .replicated);
    for ([_][]const u8{ "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.k_norm_ctx.weight" }) |k|
        if (eq(tail, k)) return req(.bf16, &.{m.head_dim}, .replicated);
    if (eq(tail, "self_attn.q_proj.weight")) return req(.bf16, &.{ Q, H }, .q_heads);
    for ([_][]const u8{ "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.k_proj_ctx.weight", "self_attn.v_proj_ctx.weight" }) |k|
        if (eq(tail, k)) return req(.bf16, &.{ KV, H }, .kv_heads);
    if (eq(tail, "self_attn.o_proj.weight")) return req(.bf16, &.{ H, Q }, .output_columns);
    if (eq(tail, "mlp.gate_proj.weight") or eq(tail, "mlp.up_proj.weight"))
        return req(.bf16, &.{ I, H }, .intermediate_rows);
    if (eq(tail, "mlp.down_proj.weight")) return req(.bf16, &.{ H, I }, .intermediate_columns);
    return error.UnknownTensor;
}

// Every stored DSpark tensor is BF16; any other dtype spelling is a different checkpoint.
pub fn validate(m: draft.Config, c: cfg.Config, name: []const u8, dtype: []const u8, shape: []const u32) !Requirement {
    const r = try expected(m, c, name);
    if (!eq(dtype, "BF16")) return error.WrongDType;
    if (!std.mem.eql(u32, r.dims(), shape)) return error.WrongShape;
    return r;
}

pub const RankSpans = struct {
    q: shard.Span,
    kv: shard.Span,
    intermediate: shard.Span,
};

// Spans count whole heads for q/kv and whole intermediate units; projections multiply by head_dim.
pub fn rankSpans(m: draft.Config, rank: u32, world: u32) !RankSpans {
    if (world != 4) return error.Inkling2RequiresFourRanks;
    if (rank >= world) return error.BadRank;
    if (m.heads % world != 0 or m.kv_heads % world != 0 or m.intermediate % world != 0)
        return error.UnevenShard;
    return .{
        .q = .{ .start = rank * (m.heads / world), .count = m.heads / world },
        .kv = .{ .start = rank * (m.kv_heads / world), .count = m.kv_heads / world },
        .intermediate = .{ .start = rank * (m.intermediate / world), .count = m.intermediate / world },
    };
}

test "all 84 released DSpark tensor headers match the native schema in BF16" {
    const a = std.testing.allocator;
    const c = try cfg.Config.parse(a, @embedFile("testdata/config.json"));
    const m = try draft.Config.parse(a, @embedFile("testdata/config.json"));
    var p = try std.json.parseFromSlice(std.json.Value, a, @embedFile("testdata/mtp-headers.json"), .{});
    defer p.deinit();
    var places: [6]u32 = @splat(0);
    var count: usize = 0;
    var it = p.value.object.iterator();
    while (it.next()) |e| {
        var shape: [3]u32 = undefined;
        const dims = e.value_ptr.object.get("shape").?.array.items;
        try std.testing.expect(dims.len <= shape.len);
        for (dims, 0..) |v, i| shape[i] = @intCast(v.integer);
        const r = validate(m, c, e.key_ptr.*, e.value_ptr.object.get("dtype").?.string, shape[0..dims.len]) catch |err| {
            std.debug.print("dspark schema {s}: {s}\n", .{ e.key_ptr.*, @errorName(err) });
            return err;
        };
        places[@backingInt(r.place)] += 1;
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 84), count);
    try std.testing.expectEqual([6]u32{ 39, 5, 20, 5, 10, 5 }, places);
}

test "dspark schema rejects unknown, fused and out-of-range tensor names" {
    const c = try cfg.Config.parse(std.testing.allocator, @embedFile("testdata/config.json"));
    const m = try draft.Config.parse(std.testing.allocator, @embedFile("testdata/config.json"));
    const q = try expected(m, c, "layers.0.self_attn.q_proj.weight");
    try std.testing.expectEqual(Place.q_heads, q.place);
    try std.testing.expectEqualSlices(u32, &.{ 6144, 4608 }, q.dims());
    try std.testing.expectError(error.UnknownTensor, expected(m, c, "layers.5.input_layernorm.weight"));
    try std.testing.expectError(error.UnknownTensor, expected(m, c, "layers.0.self_attn.qkv_proj.weight"));
    try std.testing.expectError(error.UnknownTensor, expected(m, c, "layers.0.self_attn.q_proj.bias"));
    try std.testing.expectError(error.UnknownTensor, expected(m, c, "markov_head.markov_w2.weight"));
    try std.testing.expectError(error.UnknownTensor, expected(m, c, "model.llm.embed.weight"));
    var d = m;
    d.host_layers = 47;
    try std.testing.expectError(error.HostDraftMismatch, expected(d, c, "norm.weight"));
}

test "dspark schema rejects wrong dtype and wrong shapes on real names" {
    const c = try cfg.Config.parse(std.testing.allocator, @embedFile("testdata/config.json"));
    const m = try draft.Config.parse(std.testing.allocator, @embedFile("testdata/config.json"));
    try std.testing.expectError(error.WrongDType, validate(m, c, "norm.weight", "F32", &.{4608}));
    try std.testing.expectError(error.WrongDType, validate(m, c, "fuse_weight", "F8_E4M3", &.{ 6, 8 }));
    try std.testing.expectError(error.WrongShape, validate(m, c, "layers.0.self_attn.q_proj.weight", "BF16", &.{ 6144, 6144 }));
    try std.testing.expectError(error.WrongShape, validate(m, c, "markov_head.markov_w1.weight", "BF16", &.{ 200058, 256 }));
    _ = try validate(m, c, "markov_head.rnn_proj.weight", "BF16", &.{ 768, 9728 });
    _ = try validate(m, c, "layers.4.mlp.down_proj.weight", "BF16", &.{ 4608, 18432 });
}

test "dspark placement splits whole heads and intermediate units across four ranks" {
    const m = try draft.Config.parse(std.testing.allocator, @embedFile("testdata/config.json"));
    var q_end: u32 = 0;
    var i_end: u32 = 0;
    for (0..4) |rank| {
        const s = try rankSpans(m, @intCast(rank), 4);
        try std.testing.expectEqual(@as(u32, 12), s.q.count);
        try std.testing.expectEqual(@as(u32, 2), s.kv.count);
        try std.testing.expectEqual(@as(u32, 4608), s.intermediate.count);
        try std.testing.expectEqual(q_end, s.q.start);
        try std.testing.expectEqual(i_end, s.intermediate.start);
        try std.testing.expectEqual(@as(u32, 12 * 128), s.q.count * m.head_dim);
        try std.testing.expectEqual(@as(u32, 2 * 128), s.kv.count * m.head_dim);
        q_end = s.q.end();
        i_end = s.intermediate.end();
    }
    try std.testing.expectEqual(@as(u32, 48), q_end);
    try std.testing.expectEqual(@as(u32, 18432), i_end);
    try std.testing.expectError(error.Inkling2RequiresFourRanks, rankSpans(m, 0, 2));
    try std.testing.expectError(error.BadRank, rankSpans(m, 4, 4));
    var uneven = m;
    uneven.heads = 50;
    try std.testing.expectError(error.UnevenShard, rankSpans(uneven, 0, 4));
    uneven = m;
    uneven.intermediate = 18430;
    try std.testing.expectError(error.UnevenShard, rankSpans(uneven, 0, 4));
}
