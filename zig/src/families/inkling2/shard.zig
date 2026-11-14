//! Whole-head attention TP4 and whole-expert EP4. NVFP4 activations keep their full-row outer scale.
const std = @import("std");
const Config = @import("config.zig").Config;

pub const Span = struct {
    start: u32,
    count: u32,
    pub fn end(s: Span) u32 {
        return s.start + s.count;
    }
    pub fn contains(s: Span, id: u32) bool {
        return id >= s.start and id < s.end();
    }
};

pub const Plan = struct {
    rank: u32,
    world: u32,
    q: Span,
    kv: Span,
    experts: Span,
    vocab: Span,
    valid_vocab: Span,
    expert_intermediate: u32,

    pub fn init(c: Config, rank: u32, world: u32) !Plan {
        if (world != 4) return error.Inkling2RequiresFourRanks;
        if (rank >= world) return error.BadRank;
        if (c.heads % world != 0 or c.kv_heads % world != 0 or c.experts % world != 0)
            return error.UnevenShard;
        const rows = std.mem.alignForward(u32, std.math.divCeil(u32, c.unpadded_vocab, world) catch
            return error.BadVocabulary, 128);
        const start = rank * rows;
        return .{
            .rank = rank,
            .world = world,
            .q = .{ .start = rank * (c.heads / world), .count = c.heads / world },
            .kv = .{ .start = rank * (c.kv_heads / world), .count = c.kv_heads / world },
            .experts = .{ .start = rank * (c.experts / world), .count = c.experts / world },
            .vocab = .{ .start = start, .count = rows },
            .valid_vocab = .{ .start = start, .count = @min(rows, c.unpadded_vocab -| start) },
            .expert_intermediate = c.intermediate,
        };
    }

    pub fn qColumns(p: Plan, c: Config) Span {
        return .{ .start = p.q.start * c.head_dim, .count = p.q.count * c.head_dim };
    }

    pub fn kvColumns(p: Plan, c: Config) Span {
        return .{ .start = p.kv.start * c.head_dim, .count = p.kv.count * c.head_dim };
    }
};

test "EP4 never splits the expert's activation-quantization row" {
    const c = try Config.parse(std.testing.allocator, @embedFile("testdata/config.json"));
    for (0..4) |rank| {
        const p = try Plan.init(c, @intCast(rank), 4);
        try std.testing.expectEqual(@as(u32, 2304), p.expert_intermediate);
        try std.testing.expectEqual(@as(u32, 96), p.experts.count);
        try std.testing.expectEqual(@as(u32, 12), p.q.count);
        try std.testing.expectEqual(@as(u32, 2), p.kv.count);
        try std.testing.expectEqual(@as(u32, @intCast(rank * 96)), p.experts.start);
    }
}

test "vocabulary shards cover only valid ids and pad an equal launch width" {
    const c = try Config.parse(std.testing.allocator, @embedFile("testdata/config.json"));
    var end: u32 = 0;
    for (0..4) |rank| {
        const p = try Plan.init(c, @intCast(rank), 4);
        try std.testing.expectEqual(end, p.valid_vocab.start);
        try std.testing.expectEqual(@as(u32, 50048), p.vocab.count);
        end = p.valid_vocab.end();
        if (rank == 3) try std.testing.expectEqual(@as(u32, 49914), p.valid_vocab.count);
    }
    try std.testing.expectEqual(c.unpadded_vocab, end);
}

test "shards reject invalid placement before touching tensors" {
    var c = try Config.parse(std.testing.allocator, @embedFile("testdata/config.json"));
    try std.testing.expectError(error.Inkling2RequiresFourRanks, Plan.init(c, 0, 2));
    try std.testing.expectError(error.BadRank, Plan.init(c, 4, 4));
    c.kv_heads = 7;
    try std.testing.expectError(error.UnevenShard, Plan.init(c, 0, 4));
}
