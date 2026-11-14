//! Row-major scratch and host expert grouping for <=8 target rows; independent of the scalar oracle scratch.
const std = @import("std");
const cuda = @import("cuda");

/// One (row, pick) pair an owned expert must compute for the window.
pub const Member = struct { row: u8, pick: u8 };

pub const Batch = struct {
    dev: cuda.DeviceBuffer,
    h: u64,
    x: u64,
    mid: u64,
    mix1: u64,
    mix3: u64,
    norm1: u64,
    norm3: u64,
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
    g_out: u64,
    conv: u64,
    router: u64,
    ids: u64,
    gammas: u64,
    x_norm: u64,
    gu: u64,
    act: u64,
    act_g: u64,
    a_norm: u64,
    expert_parts: u64,
    logits: u64,
    capped: u64,
    picks: u64,
    taus: u64,
    taus_one: u64,
    x_amax: u64,
    g_amax: u64,
    g_a_amax: u64,
    g_in16: u64,
    g_meta: u64,
    xrow: u64,
    host_ids: [8][8]i32 = undefined,
    host_gammas: [8][8]f32 = undefined,
    groups: [96][8]Member = undefined,
    counts: [96]u8 = @splat(0),
    meta: [96][24]u32 = @splat(@splat(0)),

    pub fn init(d: *const cuda.Driver, capacity: u32) !Batch {
        if (capacity == 0 or capacity > 32768) return error.BadStateGeometry;
        const H = 4608;
        const R = 8;
        const V: usize = 50048;
        // f32 units; the two u16 activation views are allocated at double width rather than mixing units.
        const specs = .{
            .{ "h", R * H },               .{ "x", R * H },                              .{ "mid", R * H },       .{ "mix1", R * H },
            .{ "mix3", R * 3 * H },        .{ "norm1", R * H },                          .{ "norm3", R * 3 * H }, .{ "stack", R * 8 * H },
            .{ "q", R * 1536 },            .{ "r", R * 192 },                            .{ "qo", R * 1536 },     .{ "ro", R * 192 },
            .{ "k", R * 256 },             .{ "v", R * 256 },                            .{ "kc", R * 256 },      .{ "vc", R * 256 },
            .{ "dyn_k", R * 8 },           .{ "dyn_v", R * 8 },                          .{ "gate", R * 1536 },   .{ "sink", R * 12 },
            .{ "profile", R * 12 * 1024 }, .{ "scores", R * 12 * @as(usize, capacity) }, .{ "attn", R * 1536 },   .{ "gathered", 4 * R * H },
            .{ "out", R * H },             .{ "g_out", R * H },                          .{ "conv", R * H },      .{ "router", R * 384 },
            .{ "ids", 64 },                .{ "gammas", 64 },                            .{ "x_norm", R * H },    .{ "gu", R * H },
            .{ "act", R * 2304 },          .{ "act_g", R * 2304 },                       .{ "a_norm", R * 2304 }, .{ "expert_parts", R * 8 * H },
            .{ "logits", R * V },          .{ "capped", R * V },                         .{ "picks", 64 },        .{ "taus", 8 },
            .{ "taus_one", 8 },            .{ "x_amax", 8 },                             .{ "g_amax", 8 },        .{ "g_a_amax", 8 },
            .{ "g_in16", R * H },          .{ "g_meta", 96 * 24 },                       .{ "xrow", H },
        };
        var count: usize = 0;
        inline for (specs) |entry| count += entry[1];
        var b: Batch = undefined;
        b.dev = try cuda.DeviceBuffer.alloc(d, count * 4);
        errdefer b.dev.free();
        b.counts = @splat(0);
        b.meta = @splat(@splat(0));
        var at = b.dev.ptr;
        inline for (specs) |entry| {
            @field(b, entry[0]) = at;
            at += entry[1] * 4;
        }
        const ones: [8]f32 = @splat(1);
        const taus_one: cuda.DeviceBuffer = .{ .d = d, .ptr = b.taus_one, .len = 8 * 4, .borrowed = true };
        try taus_one.upload(0, std.mem.asBytes(&ones));
        return b;
    }
    pub fn deinit(b: *Batch) void {
        b.dev.free();
        b.* = undefined;
    }
};
