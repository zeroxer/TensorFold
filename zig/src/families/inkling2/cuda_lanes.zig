//! Inkling2's target and DSpark adapters to the existing lane backend and external-drafter wrapper.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Runtime = @import("cuda_runtime.zig").Runtime;
const be = lanes.backend;
const dr = lanes.drafter;
pub const taps = [_]u32{ 5, 11, 17, 23, 29, 35, 41, 48 };

pub const Adapter = struct {
    r: *Runtime,
    active: ?*lanes.Stream = null,
    control_tokens: ?[]const u32 = null,
    control_reject: ?usize = null,
    prompt_len: usize = 0,
    pub fn target(x: *Adapter) be.Backend {
        return .{ .ptr = x, .vtable = &target_vt };
    }
    pub fn drafter(x: *Adapter) dr.Drafter {
        return .{ .ptr = x, .vtable = &draft_vt, .kind = "Inkling2 DSpark" };
    }
    pub fn facts() lanes.Model {
        return .{ .exact_width = 8, .gpu_tokens = false, .copy_drafts = false, .root_copies = false, .context_copies = false, .lone_copies = false, .batch_rows = 8, .max_streams = 1 };
    }
    fn of(p: *anyopaque) *Adapter {
        return @ptrCast(@alignCast(p));
    }
    fn require(x: *Adapter, s: *lanes.Stream) !void {
        if (x.active != s) return error.UnknownStream;
    }
    const target_vt: be.Backend.VTable = .{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = noDraft, .release = release, .prepare_features = prepareFeatures, .features = features };
    const draft_vt: dr.Drafter.VTable = .{ .taps = draftTaps, .facts = draftFacts, .open = draftOpen, .absorb = absorb, .hold = hold, .held = held, .release = draftRelease };

    fn prefill(p: *anyopaque, s: *lanes.Stream) !void {
        const x = of(p);
        if (x.active != null) return error.OnlyOneStream;
        if (s.isCancelled()) return error.Cancelled;
        if (s.images.len != 0) return error.Inkling2VisionNotImplemented;
        if (s.sampling != null) return error.Inkling2GreedyOnly;
        if (s.logprobs != null or s.prompt_logprobs != null) return error.Inkling2LogprobsNotImplemented;
        if (s.reuse.saved != null) return error.Inkling2ResumeNotImplemented;
        if (s.prompt().len + s.max_new + 8 > x.r.seq.capacity) return error.ContextFull;
        try x.r.call(.reset, 0, 0, &.{});
        try x.r.call(.prefill, 0, 0, s.prompt());
        s.cached = 0;
        x.active = s;
        x.prompt_len = s.prompt().len;
    }
    fn first(p: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
        const x = of(p);
        try x.require(s);
        if (position != s.prompt_len) return error.PositionMismatch;
        return x.r.last;
    }
    fn queue(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) !u64 {
        return error.NotPipelined;
    }
    fn read(_: *anyopaque, h: u64) !u32 {
        if (h >= 200058) return error.InvalidTokenHandle;
        return @intCast(h);
    }
    fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const x = of(p);
        if (windows.len != 1 or out.len != 1) return error.OnlyOneStream;
        const w = windows[0];
        try x.require(w.stream);
        if (w.parents != null) return error.TreesNotBuilt;
        if (w.masks.len != 0) return error.Inkling2GrammarNotImplemented;
        if (w.held != 0) return error.ExternalDraftsNotResolved;
        const n = w.rows();
        if (n > 8 or out[0].sampled.len < n or out[0].drafts.len < w.tokens.len) return error.BadWindow;
        for (w.positions, 0..) |pos, i| if (pos != x.r.seq.pos + 1 + i) return error.PositionMismatch;
        var ids: [8]u32 = undefined;
        ids[0] = w.pending;
        @memcpy(ids[1..][0..w.tokens.len], w.tokens);
        try x.r.call(.verify, 0, 0, ids[0..n]);
        @memcpy(out[0].sampled[0..n], x.r.verified[0..n]);
        @memcpy(out[0].drafts[0..w.tokens.len], w.tokens);
    }
    fn keep(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const x = of(p);
        if (windows.len != 1 or paths.len != 1) return error.OnlyOneStream;
        try x.require(windows[0].stream);
        const rows = paths[0];
        for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
        try x.r.call(.keep, @intCast(rows.len), 0, &.{});
    }
    fn noDraft(_: *anyopaque, _: []const be.DraftRequest) !void {
        return error.ExternalDrafterRequired;
    }
    fn release(p: *anyopaque, s: *lanes.Stream) void {
        const x = of(p);
        if (x.active != s) return;
        x.r.call(.reset, 0, 0, &.{}) catch |err| cuda.tp_watch.fail(@errorName(err));
        x.active = null;
    }
    fn prepareFeatures(_: *anyopaque, asked: []const u32) !void {
        if (!std.mem.eql(u32, &taps, asked)) return error.UnsupportedTargetTaps;
    }
    fn features(p: *anyopaque, s: *lanes.Stream, asked: []const u32, start: u64, rows: u32) !be.Features {
        const x = of(p);
        try x.require(s);
        try prepareFeatures(p, asked);
        if (rows == 0 or start + rows > x.r.seq.pos) return error.UncommittedFeatures;
        return .{ .buffer = x.r.taps.ptr, .offset = start * 8 * 4608 * 4, .rows = rows, .row_bytes = 8 * 4608 * 4, .space = .device, .device = 0, .dtype = .f32 };
    }
    fn draftTaps(_: *anyopaque) []const u32 {
        return &taps;
    }
    fn draftFacts(_: *anyopaque) dr.Facts {
        return .{ .depth = 7, .step_ms = 0, .plain_guard = false, .batched = false };
    }
    fn draftOpen(p: *anyopaque, s: *lanes.Stream) !void {
        const x = of(p);
        try x.require(s);
        try x.r.call(.draft_open, 0, 0, &.{});
    }
    fn absorb(p: *anyopaque, items: []const dr.Absorb) !void {
        const x = of(p);
        for (items) |item| {
            try x.require(item.stream);
            if (item.rows != null) return error.TreesNotBuilt;
            if (item.features.space != .device or item.features.dtype != .f32 or item.features.row_bytes != 8 * 4608 * 4 or
                item.features.buffer != x.r.taps.ptr or item.features.offset != item.start * 8 * 4608 * 4) return error.BadTargetFeatures;
            if (item.start > 1032) return error.ContextFull;
            try x.r.call(.absorb, @intCast(item.start), item.features.rows, &.{});
        }
    }
    fn hold(p: *anyopaque, items: []const dr.Hold) !void {
        const x = of(p);
        for (items) |item| {
            try x.require(item.stream);
            if (item.lanes != null) return error.TreesNotBuilt;
            if (item.sampling != null) return error.Inkling2GreedyOnly;
            if (item.position == 0 or item.position > 1032) return error.BadDraftPosition;
            try x.r.call(.hold, @intCast(item.position - 1), item.depth, &.{item.pending});
        }
    }
    fn held(p: *anyopaque, streams: []const *lanes.Stream, outputs: []const []u32) !void {
        const x = of(p);
        if (streams.len != 1 or outputs.len != 1) return error.OnlyOneStream;
        try x.require(streams[0]);
        const c = x.r.draft_cache orelse return error.NoDraftCache;
        if (outputs[0].len > c.held_n) return error.NoHeldDrafts;
        @memcpy(outputs[0], c.held[0..outputs[0].len]);
        if (x.control_tokens) |known| for (outputs[0], 0..) |*id, i| {
            const at = x.r.seq.pos + 1 + i - x.prompt_len;
            if (at < known.len) id.* = known[at];
            if (x.control_reject != null and x.control_reject.? == i) id.* = (id.* + 1) % 200058;
        };
    }
    fn draftRelease(p: *anyopaque, _: *lanes.Stream) void {
        const x = of(p);
        x.r.call(.draft_close, 0, 0, &.{}) catch |err| cuda.tp_watch.fail(@errorName(err));
    }
};
