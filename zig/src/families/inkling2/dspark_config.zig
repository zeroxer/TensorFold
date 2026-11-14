//! DSpark draft-head checkpoint geometry; parsing admits the released MTP metadata, not a drafter.
const std = @import("std");
const cfg = @import("config.zig");

pub const max_taps = 8;
pub const HeadType = enum { rnn_hidden };

const released_local_mask: u64 = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4);
const released_taps = [max_taps]u32{ 5, 11, 17, 23, 29, 35, 41, 48 };
const released_eps: f32 = 1e-6;
const released_theta: f32 = 1000000.0;

pub const Config = struct {
    hidden: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    max_position: u32,
    window: u32,
    block_size: u32,
    markov_rank: u32,
    head_type: HeadType,
    local_mask: u64,
    taps: [max_taps]u32,
    tap_count: u8,
    eps: f32,
    rope_theta: f32,
    host_layers: u32,

    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
        defer a.free(text);
        return parse(a, text);
    }

    pub fn parse(a: std.mem.Allocator, text: []const u8) !Config {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer parsed.deinit();
        return fromValue(parsed.value);
    }

    pub fn fromValue(root: std.json.Value) !Config {
        const outer = try object(root);
        const kind = outer.get("model_type") orelse return error.InvalidConfiguration;
        if (kind != .string or !std.mem.eql(u8, kind.string, cfg.model_type)) return error.WrongModelType;
        const host = try object(outer.get("text_config") orelse return error.InvalidConfiguration);
        const host_layers = try uint(host, "num_hidden_layers");
        const host_hidden = try uint(host, "hidden_size");
        const m = try object(outer.get("mtp_config") orelse return error.MissingDraftConfig);
        const count = try uint(m, "num_hidden_layers");
        if (count == 0 or count > 64) return error.UnsupportedGeometry;
        const locals = m.get("local_layer_ids") orelse return error.InvalidConfiguration;
        if (locals != .array) return error.InvalidConfiguration;
        var mask: u64 = 0;
        for (locals.array.items) |id| {
            if (id != .integer or id.integer < 0 or id.integer >= count) return error.InvalidLocalLayer;
            const bit = @as(u64, 1) << @as(u6, @intCast(id.integer));
            if (mask & bit != 0) return error.DuplicateLocalLayer;
            mask |= bit;
        }
        if (try boolean(m, "q_bias", false) or try boolean(m, "o_bias", false)) return error.UnsupportedBias;
        const type_value = m.get("markov_head_type") orelse return error.InvalidConfiguration;
        if (type_value != .string) return error.InvalidConfiguration;
        const head_type: HeadType = if (std.mem.eql(u8, type_value.string, "rnn_hidden"))
            .rnn_hidden
        else
            return error.UnsupportedMarkovHead;
        const tap_values = m.get("target_layer_ids") orelse return error.InvalidConfiguration;
        if (tap_values != .array or tap_values.array.items.len == 0) return error.InvalidConfiguration;
        if (tap_values.array.items.len > max_taps) return error.UnsupportedGeometry;
        var taps: [max_taps]u32 = undefined;
        for (tap_values.array.items, 0..) |v, i| {
            if (v != .integer or v.integer < 1 or v.integer > host_layers) return error.InvalidTargetLayer;
            taps[i] = @intCast(v.integer);
        }
        const tap_count: u8 = @intCast(tap_values.array.items.len);
        var i: usize = 0;
        while (i < tap_count) : (i += 1) {
            var j: usize = i + 1;
            while (j < tap_count) : (j += 1) {
                if (taps[i] == taps[j]) return error.DuplicateTargetLayer;
            }
        }
        const result: Config = .{
            .hidden = try uint(m, "hidden_size"),
            .layers = count,
            .heads = try uint(m, "num_attention_heads"),
            .kv_heads = try uint(m, "num_key_value_heads"),
            .head_dim = try uint(m, "head_dim"),
            .intermediate = try uint(m, "intermediate_size"),
            .max_position = try uint(m, "max_position_embeddings"),
            .window = try uint(m, "sliding_window_size"),
            .block_size = try uint(m, "block_size"),
            .markov_rank = try uint(m, "markov_rank"),
            .head_type = head_type,
            .local_mask = mask,
            .taps = taps,
            .tap_count = tap_count,
            .eps = try positive(m, "rms_norm_eps"),
            .rope_theta = try positive(m, "rope_theta"),
            .host_layers = host_layers,
        };
        if (result.hidden != host_hidden) return error.UnsupportedGeometry;
        try result.validate();
        return result;
    }

    pub fn validate(c: Config) !void {
        if (c.hidden == 0 or c.hidden % 32 != 0) return error.UnsupportedGeometry;
        if (c.heads == 0 or c.kv_heads == 0 or c.head_dim == 0 or c.heads % c.kv_heads != 0)
            return error.UnsupportedGeometry;
        if (c.intermediate == 0 or c.intermediate % 64 != 0) return error.UnsupportedGeometry;
        // Draft blocks must tile the sliding window exactly; a ragged tail has no stored ABI.
        if (c.window == 0 or c.block_size == 0 or c.window % c.block_size != 0) return error.UnsupportedGeometry;
        if (c.markov_rank == 0 or c.max_position == 0) return error.UnsupportedGeometry;
        if (c.host_layers == 0 or c.host_layers > 64) return error.UnsupportedGeometry;
    }

    pub fn local(c: Config, layer: u32) bool {
        std.debug.assert(layer < c.layers);
        return c.local_mask & (@as(u64, 1) << @as(u6, @intCast(layer))) != 0;
    }

    /// A metadata gate for the one released DSpark head; it is not a drafter qualification.
    pub fn admitReleased(c: Config, sm: u32, world: u32) !void {
        if (sm != 121) return error.Inkling2RequiresSm121;
        if (world != 4) return error.Inkling2RequiresFourRanks;
        if (c.hidden != 4608 or c.layers != 5 or c.heads != 48 or c.kv_heads != 8 or c.head_dim != 128 or
            c.intermediate != 18432 or c.window != 1024 or c.block_size != 8 or c.markov_rank != 256)
            return error.UnsupportedReleasedGeometry;
        if (c.local_mask != released_local_mask or c.tap_count != max_taps or
            !std.mem.eql(u32, &c.taps, &released_taps)) return error.UnsupportedReleasedGeometry;
        if (c.head_type != .rnn_hidden or c.eps != released_eps or c.rope_theta != released_theta)
            return error.UnsupportedReleasedArithmetic;
    }
};

fn object(value: std.json.Value) !std.json.ObjectMap {
    return if (value == .object) value.object else error.InvalidConfiguration;
}

fn uint(o: std.json.ObjectMap, name: []const u8) !u32 {
    const value = o.get(name) orelse return error.InvalidConfiguration;
    if (value != .integer or value.integer < 0 or value.integer > std.math.maxInt(u32))
        return error.InvalidConfiguration;
    return @intCast(value.integer);
}

fn positive(o: std.json.ObjectMap, name: []const u8) !f32 {
    const value = o.get(name) orelse return error.InvalidConfiguration;
    const number: f64 = switch (value) {
        .float => value.float,
        .integer => @floatFromInt(value.integer),
        else => return error.InvalidConfiguration,
    };
    const out: f32 = @floatCast(number);
    if (!std.math.isFinite(out) or out <= 0) return error.InvalidConfiguration;
    return out;
}

fn boolean(o: std.json.ObjectMap, name: []const u8, fallback: bool) !bool {
    const value = o.get(name) orelse return fallback;
    return if (value == .bool) value.bool else error.InvalidConfiguration;
}

const fixture = @embedFile("testdata/config.json");

test "dspark config reads the released draft geometry, local schedule and target taps" {
    const c = try Config.parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(@as(u32, 4608), c.hidden);
    try std.testing.expectEqual(@as(u32, 5), c.layers);
    try std.testing.expectEqual(@as(u32, 48), c.heads);
    try std.testing.expectEqual(@as(u32, 8), c.kv_heads);
    try std.testing.expectEqual(@as(u32, 128), c.head_dim);
    try std.testing.expectEqual(@as(u32, 18432), c.intermediate);
    try std.testing.expectEqual(@as(u32, 1048576), c.max_position);
    try std.testing.expectEqual(@as(u32, 1024), c.window);
    try std.testing.expectEqual(@as(u32, 8), c.block_size);
    try std.testing.expectEqual(@as(u32, 256), c.markov_rank);
    try std.testing.expectEqual(HeadType.rnn_hidden, c.head_type);
    try std.testing.expectEqual(released_eps, c.eps);
    try std.testing.expectEqual(released_theta, c.rope_theta);
    try std.testing.expect(c.local(0) and c.local(1) and !c.local(2) and c.local(3) and c.local(4));
    try std.testing.expectEqual(@as(u8, max_taps), c.tap_count);
    try std.testing.expectEqualSlices(u32, &released_taps, &c.taps);
    try std.testing.expectEqual(@as(u32, 48), c.host_layers);
    try c.admitReleased(121, 4);
}

test "dspark config refuses duplicate and out-of-range target taps" {
    var p = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, fixture, .{});
    defer p.deinit();
    const mtp = p.value.object.getPtr("mtp_config").?;
    const ids = mtp.object.getPtr("target_layer_ids").?;
    ids.array.items[2] = ids.array.items[1];
    try std.testing.expectError(error.DuplicateTargetLayer, Config.fromValue(p.value));
    ids.array.items[2] = .{ .integer = 49 };
    try std.testing.expectError(error.InvalidTargetLayer, Config.fromValue(p.value));
    ids.array.items[2] = .{ .integer = 0 };
    try std.testing.expectError(error.InvalidTargetLayer, Config.fromValue(p.value));
}

test "dspark config refuses duplicate and out-of-range local layers" {
    var p = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, fixture, .{});
    defer p.deinit();
    const mtp = p.value.object.getPtr("mtp_config").?;
    const ids = mtp.object.getPtr("local_layer_ids").?;
    ids.array.items[1] = ids.array.items[0];
    try std.testing.expectError(error.DuplicateLocalLayer, Config.fromValue(p.value));
    ids.array.items[1] = .{ .integer = 5 };
    try std.testing.expectError(error.InvalidLocalLayer, Config.fromValue(p.value));
}

test "dspark config refuses attention bias and non-rnn markov head types" {
    var p = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, fixture, .{});
    defer p.deinit();
    const mtp = p.value.object.getPtr("mtp_config").?;
    mtp.object.getPtr("q_bias").?.* = .{ .bool = true };
    try std.testing.expectError(error.UnsupportedBias, Config.fromValue(p.value));
    mtp.object.getPtr("q_bias").?.* = .{ .bool = false };
    mtp.object.getPtr("o_bias").?.* = .{ .bool = true };
    try std.testing.expectError(error.UnsupportedBias, Config.fromValue(p.value));
    mtp.object.getPtr("o_bias").?.* = .{ .bool = false };
    mtp.object.getPtr("markov_head_type").?.* = .{ .string = "mlp_hidden" };
    try std.testing.expectError(error.UnsupportedMarkovHead, Config.fromValue(p.value));
}

test "dspark config refuses draft geometry the released tensors cannot express" {
    var p = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, fixture, .{});
    defer p.deinit();
    const mtp = p.value.object.getPtr("mtp_config").?;
    mtp.object.getPtr("num_attention_heads").?.* = .{ .integer = 47 };
    try std.testing.expectError(error.UnsupportedGeometry, Config.fromValue(p.value));
    mtp.object.getPtr("num_attention_heads").?.* = .{ .integer = 48 };
    mtp.object.getPtr("num_key_value_heads").?.* = .{ .integer = 7 };
    try std.testing.expectError(error.UnsupportedGeometry, Config.fromValue(p.value));
    mtp.object.getPtr("num_key_value_heads").?.* = .{ .integer = 8 };
    mtp.object.getPtr("block_size").?.* = .{ .integer = 7 };
    try std.testing.expectError(error.UnsupportedGeometry, Config.fromValue(p.value));
    mtp.object.getPtr("block_size").?.* = .{ .integer = 8 };
    mtp.object.getPtr("hidden_size").?.* = .{ .integer = 4096 };
    try std.testing.expectError(error.UnsupportedGeometry, Config.fromValue(p.value));
}

test "released dspark admission pins the exact schedule and four-rank world" {
    const c = try Config.parse(std.testing.allocator, fixture);
    try c.admitReleased(121, 4);
    try std.testing.expectError(error.Inkling2RequiresSm121, c.admitReleased(120, 4));
    try std.testing.expectError(error.Inkling2RequiresFourRanks, c.admitReleased(121, 2));
    var d = c;
    d.taps[0] = 11;
    try std.testing.expectError(error.UnsupportedReleasedGeometry, d.admitReleased(121, 4));
    d = c;
    d.local_mask = 0b100111;
    try std.testing.expectError(error.UnsupportedReleasedGeometry, d.admitReleased(121, 4));
    d = c;
    d.block_size = 4;
    try std.testing.expectError(error.UnsupportedReleasedGeometry, d.admitReleased(121, 4));
    d = c;
    d.eps = 2e-6;
    try std.testing.expectError(error.UnsupportedReleasedArithmetic, d.admitReleased(121, 4));
}

test "dspark config requires the mtp_config section inside an inkling2 root" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.MissingDraftConfig, Config.parse(a, "{\"model_type\":\"inkling2\",\"text_config\":{\"num_hidden_layers\":48,\"hidden_size\":4608}}"));
    try std.testing.expectError(error.WrongModelType, Config.parse(a, "{\"model_type\":\"llama\"}"));
}
