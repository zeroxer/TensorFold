//! The release's stored text tensors. This schema admits metadata; it does not claim a GPU loader.
const std = @import("std");
const cfg = @import("config.zig");
pub const DType = enum { bf16, f32, u8, f8_e4m3 };
pub const Place = enum { replicated, q_heads, kv_heads, output_columns, experts, vocabulary };
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

/// Vision and MTP are explicitly out of this experimental text target's namespace.
pub fn outsideText(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "model.visual.") or std.mem.startsWith(u8, name, "layers.") or
        std.mem.startsWith(u8, name, "markov_head.") or eq(name, "fuse_weight") or
        eq(name, "mask_embed") or eq(name, "norm.weight");
}

pub fn expected(c: cfg.Config, name: []const u8) !Requirement {
    if (outsideText(name)) return error.OutsideTextTarget;
    const prefix = "model.llm.";
    if (!std.mem.startsWith(u8, name, prefix)) return error.UnknownTensor;
    const key = name[prefix.len..];
    const H = c.hidden;
    if (eq(key, "embed.weight") or eq(key, "unembed.weight"))
        return req(.bf16, &.{ c.vocab, H }, .vocabulary);
    if (eq(key, "embed_norm.weight") or eq(key, "norm.weight")) return req(.bf16, &.{H}, .replicated);
    if (eq(key, "attn_res.output_query")) return req(.f32, &.{H}, .replicated);
    for ([_][]const u8{ "attn_res.attn_queries.", "attn_res.mlp_queries." }, 0..) |p, q| {
        if (!std.mem.startsWith(u8, key, p)) continue;
        const n = std.fmt.parseInt(u32, key[p.len..], 10) catch return error.UnknownTensor;
        if (n >= c.layers - @as(u32, @intFromBool(q == 0))) return error.UnknownTensor;
        if (q == 0 and !c.local(n + 1)) return req(.f32, &.{ 3, H }, .replicated);
        return req(.f32, &.{H}, .replicated);
    }
    if (!std.mem.startsWith(u8, key, "layers.")) return error.UnknownTensor;
    const rest = key[7..];
    const split = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnknownTensor;
    const layer = std.fmt.parseInt(u32, rest[0..split], 10) catch return error.UnknownTensor;
    if (layer >= c.layers) return error.UnknownTensor;
    const tail = rest[split + 1 ..];
    const Q = c.heads * c.head_dim;
    const KV = c.kv_heads * c.head_dim;
    const I = c.intermediate;
    const E = c.experts;
    if (eq(tail, "mlp.gate.weight")) return req(.bf16, &.{ E, H }, .replicated);
    if (eq(tail, "mlp.gate.bias")) return req(.f32, &.{E}, .replicated);
    if (eq(tail, "attn_sconv.weight") or eq(tail, "mlp_sconv.weight"))
        return req(.bf16, &.{ H, 1, c.conv_kernel }, .replicated);
    if (eq(tail, "mlp_norm.weight")) return req(.bf16, &.{H}, .replicated);
    if (c.local(layer)) {
        if (eq(tail, "attn_norm.weight")) return req(.bf16, &.{H}, .replicated);
        if (eq(tail, "attn.k_sconv.weight") or eq(tail, "attn.v_sconv.weight"))
            return req(.bf16, &.{ KV, 1, c.conv_kernel }, .kv_heads);
    } else {
        for ([_][]const u8{ "q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight" }) |k|
            if (eq(tail, k)) return req(.bf16, &.{H}, .replicated);
        for ([_][]const u8{ "attn.k_sconv.", "attn.v_sconv." }) |p| {
            if (!std.mem.startsWith(u8, tail, p)) continue;
            const k = tail[p.len..];
            if (eq(k, "static_w")) return req(.bf16, &.{ c.conv_kernel, KV }, .kv_heads);
            if (eq(k, "weight_proj")) return req(.bf16, &.{ c.kv_heads * c.conv_kernel, H }, .kv_heads);
        }
    }
    if (eq(tail, "attn.wq_du.weight") or eq(tail, "attn.gate_attn.weight"))
        return req(.bf16, &.{ Q, H }, .q_heads);
    if (eq(tail, "attn.wk_dv.weight") or eq(tail, "attn.wv_dv.weight"))
        return req(.bf16, &.{ KV, H }, .kv_heads);
    if (eq(tail, "attn.wo_ud.weight")) return req(.bf16, &.{ H, Q }, .output_columns);
    if (eq(tail, "attn.wr_du.weight")) return req(.bf16, &.{ c.heads * c.rel_dim, H }, .q_heads);
    if (eq(tail, "attn.attn_sink_proj.weight")) return req(.bf16, &.{ c.heads, H }, .q_heads);
    if (eq(tail, "attn.q_l2_clip.scale")) return req(.bf16, &.{ c.heads, c.head_dim }, .q_heads);
    if (eq(tail, "attn.rel_l2_clip.scale")) return req(.bf16, &.{ c.heads, c.rel_dim }, .q_heads);
    if (eq(tail, "attn.rel_logits_proj.proj")) return req(.bf16, &.{ c.rel_dim, c.rel_extent }, .replicated);
    const ep = "mlp.experts.";
    if (!std.mem.startsWith(u8, tail, ep)) return error.UnknownTensor;
    const part = tail[ep.len..];
    const up = std.mem.startsWith(u8, part, "w13_");
    if (!up and !std.mem.startsWith(u8, part, "w2_")) return error.UnknownTensor;
    const k = if (up) part[4..] else part[3..];
    const N = if (up) 2 * I else H;
    const K = if (up) H else I;
    const nv = c.expertFormat(layer) == .nvfp4;
    if (eq(k, "weight")) return req(if (nv) .u8 else .f8_e4m3, &.{ E, N, K / @as(u32, if (nv) 2 else 1) }, .experts);
    if (eq(k, "weight_scale")) return req(if (nv) .f8_e4m3 else .u8, &.{ E, N, K / @as(u32, if (nv) 16 else 32) }, .experts);
    if (nv and eq(k, "weight_amax")) return req(.f32, &.{E}, .experts);
    if (nv and eq(k, "weight_outer_scale_denominator")) return req(.f32, &.{1}, .replicated);
    return error.UnknownTensor;
}

pub fn validate(c: cfg.Config, name: []const u8, dtype: []const u8, shape: []const u32) !Requirement {
    const r = try expected(c, name);
    const spelling = switch (r.dtype) {
        .bf16 => "BF16",
        .f32 => "F32",
        .u8 => "U8",
        .f8_e4m3 => "F8_E4M3",
    };
    if (!eq(dtype, spelling)) return error.WrongDType;
    if (!std.mem.eql(u32, r.dims(), shape)) return error.WrongShape;
    return r;
}

test "mixed expert formats preserve stored scales and whole expert placement" {
    const c = try cfg.Config.parse(std.testing.allocator, @embedFile("testdata/config.json"));
    const nv = try validate(c, "model.llm.layers.39.mlp.experts.w13_weight", "U8", &.{ 384, 4608, 2304 });
    try std.testing.expectEqual(Place.experts, nv.place);
    _ = try validate(c, "model.llm.layers.40.mlp.experts.w13_weight", "F8_E4M3", &.{ 384, 4608, 4608 });
    try std.testing.expectError(error.WrongDType, validate(c, "model.llm.layers.40.mlp.experts.w13_weight_scale", "F8_E4M3", &.{ 384, 4608, 144 }));
    try std.testing.expectError(error.WrongShape, validate(c, "model.llm.layers.39.mlp.experts.w2_weight", "U8", &.{ 384, 4608, 576 }));
    try std.testing.expectError(error.UnknownTensor, expected(c, "model.llm.layers.48.mlp_norm.weight"));
    try std.testing.expectError(error.UnknownTensor, expected(c, "model.llm.layers.0.q_input_norm.weight"));
}

test "all 1348 released text tensor headers match the native schema" {
    const a = std.testing.allocator;
    const c = try cfg.Config.parse(a, @embedFile("testdata/config.json"));
    var p = try std.json.parseFromSlice(std.json.Value, a, @embedFile("testdata/headers.json"), .{});
    defer p.deinit();
    var text: usize = 0;
    var outside: usize = 0;
    var it = p.value.object.iterator();
    while (it.next()) |e| {
        if (outsideText(e.key_ptr.*)) {
            outside += 1;
            continue;
        }
        var shape: [3]u32 = undefined;
        const dims = e.value_ptr.object.get("shape").?.array.items;
        try std.testing.expect(dims.len <= shape.len);
        for (dims, 0..) |v, i| shape[i] = @intCast(v.integer);
        _ = validate(c, e.key_ptr.*, e.value_ptr.object.get("dtype").?.string, shape[0..dims.len]) catch |err| {
            std.debug.print("inkling2 schema {s}: {s}\n", .{ e.key_ptr.*, @errorName(err) });
            return err;
        };
        text += 1;
    }
    try std.testing.expectEqual(@as(usize, 1348), text);
    try std.testing.expectEqual(@as(usize, 506), outside);
}
