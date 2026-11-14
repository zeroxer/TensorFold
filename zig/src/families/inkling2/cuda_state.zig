//! One experimental text sequence's KV and raw convolution histories; scratch is separate from persistent state.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const Plan = @import("shard.zig").Plan;

pub const Layer = struct { k: u64, v: u64, hist_k: u64, hist_v: u64, hist_a: u64, hist_m: u64 };
pub const State = struct {
    a: std.mem.Allocator,
    storage: cuda.DeviceBuffer,
    layers: []Layer,
    capacity: u32,
    history_bytes: usize,
    pos: u32 = 0,

    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, c: Config, p: Plan, layers: u32, capacity: u32) !State {
        if (capacity == 0 or capacity > 32768 or layers == 0 or layers > c.layers) return error.BadStateGeometry;
        const kv: usize = p.kv.count * c.head_dim;
        const cache = capacity * kv * 4;
        const kh = 3 * kv * 4;
        const hh = 3 * c.hidden * 4;
        const stride = 2 * cache + 2 * kh + 2 * hh;
        var storage = try cuda.DeviceBuffer.alloc(d, stride * layers);
        errdefer storage.free();
        try storage.fill8(0, null);
        try d.synchronize();
        const ls = try a.alloc(Layer, layers);
        for (ls, 0..) |*l, i| {
            const b = storage.ptr + stride * i;
            l.* = .{ .k = b, .v = b + cache, .hist_k = b + 2 * cache, .hist_v = b + 2 * cache + kh, .hist_a = b + 2 * cache + 2 * kh, .hist_m = b + 2 * cache + 2 * kh + hh };
        }
        return .{ .a = a, .storage = storage, .layers = ls, .capacity = capacity, .history_bytes = 2 * kh + 2 * hh };
    }
    pub fn reset(s: *State) !void {
        try s.storage.fill8(0, null);
        try s.storage.d.synchronize();
        s.pos = 0;
    }
    pub fn deinit(s: *State) void {
        s.storage.free();
        s.a.free(s.layers);
        s.* = undefined;
    }
};

pub const Scratch = struct {
    storage: cuda.DeviceBuffer,
    h: u64,
    mid: u64,
    x: u64,
    mix: u64,
    norm: u64,
    stack: u64,
    q: u64,
    r: u64,
    qo: u64,
    ro: u64,
    k: u64,
    v: u64,
    kc: u64,
    vc: u64,
    dyn_k: u64,
    dyn_v: u64,
    gate: u64,
    sink: u64,
    profile: u64,
    scores: u64,
    attn: u64,
    gathered: u64,
    out: u64,
    conv: u64,
    router: u64,
    ids: u64,
    gammas: u64,
    xq: u64,
    gu: u64,
    act: u64,
    aq: u64,
    expert_parts: u64,
    logits: u64,
    capped: u64,
    picks: u64,
    x_norm: u64,
    a_norm: u64,
    x_amax: u64,
    a_amax: u64,

    pub fn init(d: *const cuda.Driver, capacity: u32) !Scratch {
        if (capacity == 0 or capacity > 32768) return error.BadStateGeometry;
        const H = 4608;
        const specs = .{
            .{ "h", H },               .{ "mid", H },                            .{ "x", H },          .{ "mix", 3 * H },      .{ "norm", 3 * H }, .{ "stack", 8 * H },
            .{ "q", 1536 },            .{ "r", 192 },                            .{ "qo", 1536 },      .{ "ro", 192 },         .{ "k", 256 },      .{ "v", 256 },
            .{ "kc", 256 },            .{ "vc", 256 },                           .{ "dyn_k", 8 },      .{ "dyn_v", 8 },        .{ "gate", 1536 },  .{ "sink", 12 },
            .{ "profile", 12 * 1024 }, .{ "scores", @as(usize, 12) * capacity }, .{ "attn", 1536 },    .{ "gathered", 4 * H }, .{ "out", H },      .{ "conv", H },
            .{ "router", 384 },        .{ "ids", 8 },                            .{ "gammas", 8 },     .{ "xq", H },           .{ "gu", 4608 },    .{ "act", 2304 },
            .{ "aq", 2304 },           .{ "expert_parts", 8 * H },               .{ "logits", 50048 }, .{ "capped", 50048 },   .{ "picks", 8 },    .{ "x_norm", 4608 },
            .{ "a_norm", 2304 },       .{ "x_amax", 16 },                        .{ "a_amax", 16 },
        };
        var count: usize = 0;
        inline for (specs) |entry| count += entry[1];
        var s: Scratch = undefined;
        s.storage = try cuda.DeviceBuffer.alloc(d, count * 4);
        var at = s.storage.ptr;
        inline for (specs) |entry| {
            @field(s, entry[0]) = at;
            at += entry[1] * 4;
        }
        return s;
    }
    pub fn deinit(s: *Scratch) void {
        s.storage.free();
        s.* = undefined;
    }
};
