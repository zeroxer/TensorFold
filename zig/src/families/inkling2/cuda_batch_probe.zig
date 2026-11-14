//! Whole-graph row parity: scalar oracle versus mixed-width windows, including every tapped feature and cache byte.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const forward = @import("cuda_forward.zig");

fn equal(r: anytype, tag: []const u8, ptr: u64, want: []const u8) !void {
    const got = try r.a.alloc(u8, want.len);
    defer r.a.free(got);
    try cuda.memory.download(r.d, ptr, got);
    if (!std.mem.eql(u8, got, want)) {
        var at: usize = 0;
        while (at < got.len and got[at] == want[at]) : (at += 1) {}
        std.debug.print("INKLING2_BATCH_PARITY_FAIL rank={d} tag={s} byte={d} got={d} want={d}\n", .{ r.group.rank, tag, at, got[at], want[at] });
        return error.WindowScalarByteMismatch;
    }
}

pub fn run(r: anytype) !void {
    const ids = [_]u32{ 42, 1000, 1795, 4474, 82, 13, 68014, 1 };
    const full = r.w.layers == r.w.c.layers;
    const row_bytes = (if (full) r.w.p.vocab.count else r.w.c.hidden) * 4;
    const features_len = ids.len * 8 * 4608 * 4;
    var oracle = try state.State.init(r.a, r.d, r.w.c, r.w.p, r.w.layers, r.seq.capacity);
    defer oracle.deinit();
    var feature = try cuda.DeviceBuffer.alloc(r.d, features_len);
    defer feature.free();
    const logits = try r.a.alloc(u8, ids.len * row_bytes);
    defer r.a.free(logits);
    const features = try r.a.alloc(u8, features_len);
    defer r.a.free(features);
    const cache = try r.a.alloc(u8, oracle.storage.len);
    defer r.a.free(cache);
    var f: forward.Forward = .{ .d = r.d, .stream = r.stream, .k = r.k, .w = r.w, .b = &r.scratch, .group = r.group, .math = r.math, .features = feature.ptr };
    for (ids, 0..) |id, row| {
        try f.token(&oracle, id);
        try r.stream.synchronize();
        try cuda.memory.download(r.d, if (full) r.scratch.capped else r.scratch.h, logits[row * row_bytes ..][0..row_bytes]);
    }
    if (full) try feature.download(0, features);
    try oracle.storage.download(0, cache);
    f.features = r.taps.ptr;
    const plans = [_][]const usize{ &.{8}, &.{ 3, 5 }, &.{ 1, 7 }, &.{8} };
    for (plans, 0..) |chunks, case| {
        try r.seq.reset();
        var at: usize = 0;
        for (chunks) |n| {
            try f.window(&r.seq, &r.batch, &r.hist, ids[at..][0..n], if (case == 3) .none else .all);
            try r.stream.synchronize();
            if (case != 3) try equal(r, if (full) "logits" else "partial-hidden", if (full) r.batch.capped else r.batch.h, logits[at * row_bytes ..][0 .. n * row_bytes]);
            at += n;
        }
        if (full) try equal(r, "features", r.taps.ptr, features);
        try equal(r, "cache-and-history", r.seq.storage.ptr, cache);
    }
    try r.seq.reset();
    r.hist.rows = 0;
    std.debug.print("INKLING2_BATCH_PARITY_PASS rank={d} layers={d} cases=4 outputs_state_exact=true features_checked={}\n", .{ r.group.rank, r.w.layers, full });
}
