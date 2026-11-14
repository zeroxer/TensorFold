//! Finite real-model checks through the shared scheduler, including planted rejection and accepted-prefix controls.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const weights = @import("cuda_weights.zig");
const kernels = @import("cuda_kernels.zig");
const math = @import("cuda_lane.zig");
const draft = @import("cuda_dspark.zig");
const Runtime = @import("cuda_runtime.zig").Runtime;
const Adapter = @import("cuda_lanes.zig").Adapter;

const Record = struct {
    drafts: bool,
    repeat: usize,
    warmup: bool,
    control: []const u8,
    ids: []u32,
    total_seconds: f64,
    first_seconds: f64,
    decode_tokens_per_second: f64,
    target_rows: u64,
    verified_rows: u64,
    accepted_rows: u64,
    rejected_rows: u64,
    proposals: u64,
    draft_blocks: u64,
};

fn one(a: std.mem.Allocator, io: std.Io, x: *Adapter, prompt: []const u32, limit: u32, on: bool, repeat: usize, control: []const u8) !Record {
    var wrapped = try lanes.drafted.Drafted.init(a, x.target(), x.drafter());
    defer wrapped.deinit();
    var cfg = try lanes.Config.init(a, if (on) wrapped.facts(Adapter.facts()) else Adapter.facts(), 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.backend.WallClock = .{ .io = io };
    const backend = if (on) wrapped.backend() else x.target();
    var engine = lanes.Engine.init(a, &cfg, backend, clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "inkling2-native-check", .prompt = prompt, .max_new = limit, .eos = &.{200006}, .drafts = on });
    defer s.deinit(a);
    defer backend.release(&s);
    const old = .{ x.r.target_rows, x.r.verified_rows, x.r.accepted_rows, x.r.rejected_rows, x.r.draft.proposed, x.r.draft.blocks };
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var first_ns: i96 = 0;
    try engine.addStream(&s);
    while (!s.finished) {
        try engine.step();
        if (first_ns == 0 and s.emitted().len > 0) first_ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    }
    if (s.reason == .@"error" or s.reason == .cancelled) return error.NativeGenerationFailed;
    const seconds = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) * 1e-9;
    const out = try a.dupe(u32, s.emitted());
    const first_seconds = @as(f64, @floatFromInt(first_ns)) * 1e-9;
    return .{ .drafts = on, .repeat = repeat, .warmup = repeat == 0, .control = control, .ids = out, .total_seconds = seconds, .first_seconds = first_seconds, .decode_tokens_per_second = @as(f64, @floatFromInt(out.len -| 1)) / @max(seconds - first_seconds, 1e-9), .target_rows = x.r.target_rows - old[0], .verified_rows = x.r.verified_rows - old[1], .accepted_rows = x.r.accepted_rows - old[2], .rejected_rows = x.r.rejected_rows - old[3], .proposals = x.r.draft.proposed - old[4], .draft_blocks = x.r.draft.blocks - old[5] };
}

fn save(a: std.mem.Allocator, io: std.Io, out_dir: []const u8, records: []const Record, done: bool) !void {
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try std.json.Stringify.value(.{ .shared_scheduler = true, .completed = done, .max_draft_depth = 7, .records = records }, .{}, &text.writer);
    const path = try std.fmt.allocPrint(a, "{s}/shared-dspark-benchmark.json", .{out_dir});
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text.written() });
}

fn cancel(a: std.mem.Allocator, io: std.Io, x: *Adapter, prompt: []const u32, limit: u32) !void {
    var wrapped = try lanes.drafted.Drafted.init(a, x.target(), x.drafter());
    defer wrapped.deinit();
    var cfg = try lanes.Config.init(a, wrapped.facts(Adapter.facts()), 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.backend.WallClock = .{ .io = io };
    var engine = lanes.Engine.init(a, &cfg, wrapped.backend(), clock.clock());
    defer engine.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "cancel-check", .prompt = prompt, .max_new = limit, .drafts = true });
    defer s.deinit(a);
    defer wrapped.backend().release(&s);
    try engine.addStream(&s);
    while (s.emitted().len < 4 and !s.finished) try engine.step();
    if (s.finished) return error.CancellationNotExercised;
    const n = s.emitted().len;
    engine.discard(&s);
    if (!s.finished or s.reason != .cancelled or s.emitted().len != n or x.active != null or x.r.draft_cache != null or x.r.seq.pos != 0)
        return error.CancellationStateLeaked;
}

pub fn run(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, group: *const cuda.tp_group.Group, w: *const weights.Weights, k: *const kernels.Kernels, m: *const math.Math, model_dir: []const u8, image_dir: []const u8, prompt: []const u32, limit: u32, cap: u32, out_dir: []const u8) !void {
    var head = try draft.Model.init(a, io, d, stream, group, w, k, m, model_dir, image_dir, cap + 8);
    defer head.deinit();
    var runtime = try Runtime.init(a, d, stream, group, w, k, m, &head, cap + 8);
    defer runtime.deinit();
    try @import("cuda_batch_probe.zig").run(&runtime);
    if (group.rank != 0) {
        try runtime.follow();
        return;
    }
    var stopped = false;
    defer if (!stopped) runtime.call(.stop, 0, 0, &.{}) catch {};
    var adapter: Adapter = .{ .r = &runtime };
    var records: std.ArrayList(Record) = .empty;
    defer {
        for (records.items) |r| a.free(r.ids);
        records.deinit(a);
    }
    var golden: []const u32 = &.{};
    for (0..4) |rep| {
        for ([_]bool{ false, true }) |on| {
            const row = try one(a, io, &adapter, prompt, limit, on, rep, "");
            try records.append(a, row);
            try save(a, io, out_dir, records.items, false);
            if (rep == 0 and !on) golden = row.ids else if (!std.mem.eql(u32, golden, row.ids)) return error.DraftedPlainMismatch;
            if (on and (row.proposals == 0 or row.draft_blocks == 0)) return error.SilentDraftFallback;
            std.debug.print("INKLING2_SHARED_RUN repeat={d} drafts={} tokens={d} seconds={d:.6} proposals={d} rejected_rows={d}\n", .{ rep, on, row.ids.len, row.total_seconds, row.proposals, row.rejected_rows });
        }
    }
    try @import("cuda_controls.zig").run(a, io, &adapter, prompt, golden, out_dir);
    try cancel(a, io, &adapter, prompt, limit);
    const after = try one(a, io, &adapter, prompt, limit, true, 5, "after-cancel");
    try records.append(a, after);
    if (!std.mem.eql(u32, golden, after.ids)) return error.AfterCancellationMismatch;
    try save(a, io, out_dir, records.items, true);
    try runtime.call(.stop, 0, 0, &.{});
    stopped = true;
    std.debug.print("INKLING2_SHARED_DSPARK_PASS plain_drafted_equal=true controls=3 cancel_reset=true\n", .{});
}
