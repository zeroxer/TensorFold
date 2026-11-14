//! Fixed eight-row controls exercise each intended commit, independent of the scheduler's adaptive depth.
const std = @import("std");
const lanes = @import("lanes");
const Adapter = @import("cuda_lanes.zig").Adapter;
const be = lanes.backend;

pub fn run(a: std.mem.Allocator, io: std.Io, x: *Adapter, prompt: []const u32, golden: []const u32, out_dir: []const u8) !void {
    if (golden.len < 10) return error.ControlNeedsContinuation;
    var records: [3]struct { name: []const u8, accepted_rows: usize, next_token: u32 } = undefined;
    for ([_]?usize{ 0, 3, null }, [_][]const u8{ "reject-first", "accept-prefix-3", "accept-all" }, 0..) |reject, name, ci| {
        var wrapped = try lanes.drafted.Drafted.init(a, x.target(), x.drafter());
        defer wrapped.deinit();
        const backend = wrapped.backend();
        var s = try lanes.Stream.init(a, .{ .id = name, .prompt = prompt, .max_new = 32, .drafts = true });
        defer s.deinit(a);
        defer backend.release(&s);
        try backend.prefill(&s);
        if (try backend.read(try backend.first(&s, prompt.len)) != golden[0]) return error.ControlFirstMismatch;
        x.control_tokens = golden;
        x.control_reject = reject;
        defer {
            x.control_tokens = null;
            x.control_reject = null;
        }
        try backend.draft(&.{.{ .stream = &s, .follow = &.{}, .first = .{ .value = golden[0] }, .rows = null, .start = 0, .position = prompt.len + 1, .depth = 7 }});
        var positions: [8]u64 = undefined;
        for (&positions, 0..) |*p, i| p.* = prompt.len + 1 + i;
        const windows = [_]be.Window{.{ .stream = &s, .pending = golden[0], .held = 7, .tokens = &.{}, .parents = null, .positions = &positions }};
        var sampled: [8]u32 = undefined;
        var drafts: [7]u32 = undefined;
        var out = [_]be.Verified{.{ .sampled = &sampled, .drafts = &drafts }};
        try backend.verify(&windows, &out);
        var inputs: [8]u32 = undefined;
        inputs[0] = golden[0];
        @memcpy(inputs[1..], &drafts);
        const parents = try lanes.accept.rowParents(a, 8, null);
        defer a.free(parents);
        const path = try lanes.accept.acceptPath(a, &inputs, parents, &sampled);
        defer a.free(path);
        const expected: usize = if (reject) |n| n + 1 else 8;
        if (path.len != expected) return error.ControlPathNotExercised;
        try backend.keep(&windows, &.{path});
        if (x.r.seq.pos != prompt.len + expected) return error.ControlPositionMismatch;
        x.control_tokens = null;
        x.control_reject = null;
        try backend.draft(&.{.{ .stream = &s, .follow = golden[1 .. expected + 1], .rows = path, .start = prompt.len, .position = prompt.len + expected + 1, .depth = 7 }});
        if (x.r.draft_cache.?.rows != x.r.seq.pos) return error.ControlDraftCommitMismatch;
        var next: [1]u32 = undefined;
        var empty: [0]u32 = .{};
        var check = [_]be.Verified{.{ .sampled = &next, .drafts = &empty }};
        const pos = [_]u64{x.r.seq.pos + 1};
        try backend.verify(&.{.{ .stream = &s, .pending = golden[expected], .held = 0, .tokens = &.{}, .parents = null, .positions = &pos }}, &check);
        if (next[0] != golden[expected + 1]) return error.ControlRollbackMismatch;
        records[ci] = .{ .name = name, .accepted_rows = path.len, .next_token = next[0] };
        std.debug.print("INKLING2_FIXED_CONTROL name={s} accepted_rows={d} next_equal=true\n", .{ name, path.len });
    }
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try std.json.Stringify.value(.{ .width = 8, .all_passed = true, .records = records }, .{}, &text.writer);
    const path = try std.fmt.allocPrint(a, "{s}/fixed-controls.json", .{out_dir});
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text.written() });
}
