//! An explicit development tool, not the server: check the CUDA ABI, optionally load this rank's text shards.
const std = @import("std");
const cuda = @import("cuda");
const kernels = @import("cuda_kernels.zig");
const config = @import("config.zig");
const shard = @import("shard.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const forward = @import("cuda_forward.zig");
const core = @import("core");
const trace = @import("cuda_trace.zig");
const draft = @import("cuda_dspark.zig");
const lane_math = @import("cuda_lane.zig");
const prepare = @import("cuda_prepare.zig");
const bench = @import("cuda_bench.zig");

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const lane_probe = args.len == 3 and std.mem.eql(u8, args[2], "--lane-probe");
    if (!lane_probe and args.len != 2 and args.len != 5 and args.len != 7 and args.len != 8) {
        std.debug.print("usage: tf-inkling2-cuda <fatbin-dir> [<model-dir> <rank:0..3> <layers:1..48> [<ids.json> <out-dir> [<greedy-new-tokens:0..128>]]]\n", .{});
        return 2;
    }
    var d = try cuda.Driver.open();
    defer d.close();
    var ctx = try cuda.Context.init(&d, 0);
    defer ctx.deinit();
    if (try d.capability(0) != 121) return error.Inkling2RequiresSm121;
    var stream = try cuda.Stream.init(&d, false);
    defer stream.deinit();
    var k = try kernels.Kernels.load(a, io, &d, args[1]);
    defer k.deinit();
    if (lane_probe) {
        try @import("cuda_lane_probe.zig").run(a, io, &d, stream, &k, args[1]);
        return 0;
    }
    const H = 4608;
    const x = try a.alloc(f32, H);
    defer a.free(x);
    const w = try a.alloc(u16, H);
    defer a.free(w);
    @memset(x, 2);
    @memset(w, 0x3f80);
    var dx = try cuda.DeviceBuffer.fromHost(&d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dw = try cuda.DeviceBuffer.fromHost(&d, std.mem.sliceAsBytes(w));
    defer dw.free();
    var dy = try cuda.DeviceBuffer.alloc(&d, H * 4);
    defer dy.free();
    try k.rms(stream, dx.ptr, dw.ptr, dy.ptr, H, 1e-6);
    try stream.synchronize();
    try dy.download(0, std.mem.sliceAsBytes(x));
    for (x) |v| if (v != 1) return error.CudaAbiMismatch;
    std.debug.print("INKLING2_NATIVE_ABI_PASS hidden={d} native_driver=true registered_server=false\n", .{H});
    if (args.len >= 5) {
        const c = try config.Config.read(a, io, args[2]);
        const rank = try std.fmt.parseInt(u32, args[3], 10);
        const layers = try std.fmt.parseInt(u32, args[4], 10);
        const p = try shard.Plan.init(c, rank, 4);
        var isolated: ?u32 = null;
        var draft_probe = false;
        if (args.len >= 7) {
            const input = try std.Io.Dir.cwd().readFileAlloc(io, args[5], a, .limited(1 << 20));
            defer a.free(input);
            var parsed = try std.json.parseFromSlice(std.json.Value, a, input, .{});
            defer parsed.deinit();
            if (parsed.value == .object) if (parsed.value.object.get("isolated_layer")) |v| {
                if (v != .integer or v.integer < 0 or v.integer >= c.layers or layers != 1) return error.BadLayerProbe;
                isolated = @intCast(v.integer);
            };
            if (parsed.value == .object) if (parsed.value.object.get("draft_probe")) |v| {
                draft_probe = v == .bool and v.bool;
            };
        }
        var loaded = if (draft_probe) try weights.loadBase(a, io, &d, args[2], c, p) else if (isolated) |l| try weights.loadLayer(a, io, &d, args[2], c, p, l) else try weights.load(a, io, &d, args[2], c, p, layers);
        defer loaded.deinit();
        std.debug.print("INKLING2_NATIVE_LOAD_PASS rank={d} layers={d}/48 tensors={d} bytes={d} inference=false\n", .{ rank, layers, loaded.entries.count(), loaded.bytes });
        if (args.len >= 7) try run(a, io, &d, stream, &k, &loaded, args[2], args[1], args[5], args[6], if (args.len == 8) try std.fmt.parseInt(u32, args[7], 10) else 0);
    }
    return 0;
}

fn run(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, k: *const kernels.Kernels, w: *weights.Weights, model_dir: []const u8, image_dir: []const u8, ids_file: []const u8, out_dir: []const u8, new_tokens: u32) !void {
    const input = try std.Io.Dir.cwd().readFileAlloc(io, ids_file, a, .limited(1 << 20));
    defer a.free(input);
    var p = try std.json.parseFromSlice(std.json.Value, a, input, .{});
    defer p.deinit();
    const ids = if (p.value == .array) p.value else if (p.value == .object) p.value.object.get("ids") orelse return error.BadIds else return error.BadIds;
    const prompt = if (p.value == .object) p.value.object.get("prompt") else null;
    const core_mode = p.value == .object and if (p.value.object.get("shared_dspark")) |v| v == .bool and v.bool else false;
    const isolated = if (p.value == .object) p.value.object.get("isolated_layer") else null;
    try validIds(ids, w.c.unpadded_vocab);
    if (new_tokens > 128 or (new_tokens > 0 and w.layers != 48)) return error.ExperimentalFullGreedyOnly;
    if (prompt) |list| try validIds(list, w.c.unpadded_vocab);
    const identity = try core.checkpoint_identity.identity(a, io, model_dir);
    try cuda.tp_watch.start(io, w.p.rank, 4);
    errdefer cuda.tp_watch.fail("Inkling2DevelopmentForwardFailed");
    var lib = try cuda.nccl.Library.open();
    var moved = false;
    errdefer if (!moved) lib.close();
    try lib.requireSupportedVersion();
    var id: cuda.nccl.UniqueId = undefined;
    if (w.p.rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        cuda.tp_watch.sendId(&id.internal);
    } else id.internal = try cuda.tp_watch.takeId(io);
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, 4, id, @intCast(w.p.rank)), "ncclCommInitRank");
    var group: cuda.tp_group.Group = .{ .lib = lib, .comm = comm, .rank = w.p.rank, .world = 4 };
    moved = true;
    defer group.close();
    try cuda.tp_watch.ready(io, identity);
    if (w.layers == 0) {
        try draftProbe(a, io, d, stream, k, w, &group, model_dir, image_dir, out_dir, p.value);
        cuda.tp_watch.ended();
        var barrier = try cuda.DeviceBuffer.alloc(d, 16);
        defer barrier.free();
        try barrier.fill8(0, null);
        try group.allGather(stream, barrier.ptr + w.p.rank * 4, barrier.ptr, 4);
        try stream.synchronize();
        return;
    }
    const prompt_rows = if (prompt) |list| list.array.items.len else ids.array.items.len;
    var math: ?lane_math.Math = null;
    defer if (math) |*m| m.deinit();
    const fast = core_mode or (p.value == .object and if (p.value.object.get("fast")) |v| v == .bool and v.bool else false);
    if (fast) {
        math = try lane_math.Math.load(a, io, d, image_dir);
        try prepare.target(w, &math.?, d, stream);
    }
    const cap: u32 = @intCast(@max(32, @max(ids.array.items.len, prompt_rows + new_tokens)));
    if (cap > 1024) return error.ExperimentalContextLimit;
    var seq = try state.State.init(a, d, w.c, w.p, w.layers, cap);
    defer seq.deinit();
    var scratch = try state.Scratch.init(d, cap);
    defer scratch.deinit();
    var f: forward.Forward = .{ .d = d, .stream = stream, .k = k, .w = w, .b = &scratch, .group = &group };
    if (math) |*m| f.math = m;
    try std.Io.Dir.cwd().createDirPath(io, out_dir);
    const trace_dir = try std.fmt.allocPrint(a, "{s}/trace", .{out_dir});
    defer a.free(trace_dir);
    var dump: trace.Dump = .{ .a = a, .io = io, .d = d, .stream = stream, .dir = trace_dir };
    const traced = p.value == .object and if (p.value.object.get("trace")) |v| v == .bool and v.bool else false;
    if (traced and w.p.rank == 0) {
        try std.Io.Dir.cwd().createDirPath(io, trace_dir);
        f.watch = dump.watch();
    }
    const count: usize = if (w.layers == w.c.layers) w.p.vocab.count else w.c.hidden;
    const host = try a.alloc(f32, count);
    defer a.free(host);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    for (ids.array.items, 0..) |v, row| {
        if (isolated) |value| {
            const layer_id: u32 = @intCast(value.integer);
            if (new_tokens != 0 or w.layers != 1) return error.BadLayerProbe;
            const folder = p.value.object.get("fixture_dir") orelse return error.MissingLayerFixtures;
            if (folder != .string) return error.MissingLayerFixtures;
            var depth: usize = (layer_id + 5) / 6;
            try putFixture(a, io, d, folder.string, @intCast(row), layer_id, "entry", scratch.h, 4608);
            try putFixture(a, io, d, folder.string, @intCast(row), layer_id, "stack", scratch.stack, 4608 * depth);
            seq.pos = @intCast(row);
            try f.layer(&seq, seq.layers[0], layer_id, &depth);
        } else try f.token(&seq, @intCast(v.integer));
        try stream.synchronize();
        try group.check();
        try cuda.memory.download(d, if (w.layers == w.c.layers) scratch.capped else scratch.h, std.mem.sliceAsBytes(host));
        const path = try std.fmt.allocPrint(a, "{s}/rank{d}-row{d}-{s}.f32", .{ out_dir, w.p.rank, row, if (w.layers == w.c.layers) "logits" else "partial-hidden" });
        defer a.free(path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(host) });
    }
    f.watch = null;
    if (p.value == .object and if (p.value.object.get("batch_parity")) |v| v == .bool and v.bool else false) {
        if (math == null) return error.BatchNeedsPackedWeights;
        var batch = try @import("cuda_batch.zig").Batch.init(d, cap);
        defer batch.deinit();
        var histories = try @import("cuda_history.zig").History.init(d, &seq);
        defer histories.deinit();
        var taps = try cuda.DeviceBuffer.alloc(d, cap * 8 * 4608 * 4);
        defer taps.free();
        var rig = .{ .a = a, .d = d, .stream = stream, .group = &group, .w = w, .k = k, .math = &math.?, .seq = seq, .scratch = scratch, .taps = taps, .batch = batch, .hist = histories };
        try @import("cuda_batch_probe.zig").run(&rig);
    }
    const elapsed = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) * 1e-9;
    std.debug.print("INKLING2_NATIVE_FORWARD_PASS rank={d} layers={d}/48 rows={d} seconds={d:.6} full_logits={}\n", .{ w.p.rank, w.layers, ids.array.items.len, elapsed, w.layers == w.c.layers });
    if (core_mode) {
        const list = prompt orelse return error.MissingDraftPrompt;
        const prompt_ids = try a.alloc(u32, list.array.items.len);
        defer a.free(prompt_ids);
        for (list.array.items, prompt_ids) |v, *token_id| token_id.* = @intCast(v.integer);
        try bench.run(a, io, d, stream, &group, w, k, &math.?, model_dir, image_dir, prompt_ids, new_tokens, cap, out_dir);
    } else if (new_tokens > 0) {
        if (prompt) |list| {
            try seq.reset();
            const start = std.Io.Clock.awake.now(io).toNanoseconds();
            for (list.array.items) |v| try f.token(&seq, @intCast(v.integer));
            try stream.synchronize();
            try cuda.memory.download(d, scratch.capped, std.mem.sliceAsBytes(host));
            const seconds = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - start)) * 1e-9;
            std.debug.print("INKLING2_NATIVE_PROMPT rank={d} tokens={d} seconds={d:.6}\n", .{ w.p.rank, list.array.items.len, seconds });
        }
        try generate(a, io, &f, &seq, host, new_tokens, out_dir);
    }
    // Mark every rank as intentionally ending before the final collective permits any process to exit.
    cuda.tp_watch.ended();
    try group.allGather(stream, scratch.gathered + w.p.rank * 4, scratch.gathered, 4);
    try stream.synchronize();
}

fn draftProbe(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, k: *const kernels.Kernels, w: *const weights.Weights, group: *const cuda.tp_group.Group, model_dir: []const u8, image_dir: []const u8, out_dir: []const u8, request: std.json.Value) !void {
    const feature_path = request.object.get("features") orelse return error.MissingDraftFeatures;
    const rows_value = request.object.get("context_rows") orelse return error.MissingDraftFeatures;
    const anchor_value = request.object.get("anchor") orelse return error.MissingDraftFeatures;
    if (feature_path != .string or rows_value != .integer or anchor_value != .integer or
        rows_value.integer < 1 or rows_value.integer > 1024 or anchor_value.integer < 0 or anchor_value.integer >= 200058)
        return error.BadDraftFixture;
    const rows: u32 = @intCast(rows_value.integer);
    const anchor: u32 = @intCast(anchor_value.integer);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, feature_path.string, a, .limited(rows * 8 * 4608 * 4 + 1));
    defer a.free(bytes);
    if (bytes.len != rows * 8 * 4608 * 4) return error.BadDraftFixture;
    var feature = try cuda.DeviceBuffer.fromHost(d, bytes);
    defer feature.free();
    var math = try lane_math.Math.load(a, io, d, image_dir);
    defer math.deinit();
    var model = try draft.Model.init(a, io, d, stream, group, w, k, &math, model_dir, image_dir, rows + 8);
    defer model.deinit();
    var cache = try draft.Cache.init(d, rows + 8);
    defer cache.deinit();
    try model.absorb(&cache, feature.ptr, 0, rows);
    try model.hold(&cache, anchor, rows, 7);
    const expected = cache.held;
    for (1..8) |depth| {
        try model.hold(&cache, anchor, rows, @intCast(depth));
        if (!std.mem.eql(u32, expected[0..depth], cache.held[0..depth])) return error.DraftDepthChangedPrefix;
    }
    for (0..4) |_| {
        try model.hold(&cache, anchor, rows, 7);
        if (!std.mem.eql(u32, &expected, &cache.held)) return error.DraftNotRepeatable;
    }
    try std.Io.Dir.cwd().createDirPath(io, out_dir);
    const path = try std.fmt.allocPrint(a, "{s}/rank{d}-dspark.json", .{ out_dir, w.p.rank });
    defer a.free(path);
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try std.json.Stringify.value(.{ .draft_ids = expected, .context_rows = rows, .anchor = anchor, .depths_checked = 7, .repeats_checked = 5, .stored_tensors = 84, .native = true }, .{}, &text.writer);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text.written() });
    std.debug.print("INKLING2_DSPARK_BLOCK_PASS rank={d} ids={any}\n", .{ w.p.rank, expected });
}

fn putFixture(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, row: u32, l: u32, tag: []const u8, dst: u64, count: usize) !void {
    const path = try std.fmt.allocPrint(a, "{s}/r{d}-l{d}-{s}.f32", .{ dir, row, l, tag });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(count * 4 + 1));
    defer a.free(bytes);
    if (bytes.len != count * 4) return error.BadFixtureSize;
    const view: cuda.DeviceBuffer = .{ .d = d, .ptr = dst, .len = count * 4, .borrowed = true };
    try view.upload(0, bytes);
}

fn validIds(v: std.json.Value, vocab: u32) !void {
    if (v != .array or v.array.items.len == 0 or v.array.items.len > 1024) return error.BadIds;
    for (v.array.items) |id| if (id != .integer or id.integer < 0 or id.integer >= vocab) return error.BadToken;
}

const Pick = extern struct { score: f32, id: u32 };
fn generate(a: std.mem.Allocator, io: std.Io, f: *forward.Forward, s: *state.State, host: []f32, limit: u32, out_dir: []const u8) !void {
    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(a);
    const view: cuda.DeviceBuffer = .{ .d = f.d, .ptr = f.b.picks, .len = 32, .borrowed = true };
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    var forwards: u32 = 0;
    for (0..limit) |step| {
        var mine: Pick = .{ .score = -std.math.inf(f32), .id = std.math.maxInt(u32) };
        for (host[0..f.w.p.valid_vocab.count], 0..) |v, i| {
            if (!std.math.isFinite(v)) return error.NonfiniteLogits;
            if (v > mine.score) mine = .{ .score = v, .id = @intCast(i + f.w.p.vocab.start) };
        }
        try view.upload(f.w.p.rank * 8, std.mem.asBytes(&mine));
        try f.group.allGather(f.stream, view.ptr + f.w.p.rank * 8, view.ptr, 8);
        try f.stream.synchronize();
        var all: [4]Pick = undefined;
        try view.download(0, std.mem.asBytes(&all));
        var best = mine;
        for (all) |p| if (p.score > best.score or (p.score == best.score and p.id < best.id)) {
            best = p;
        };
        if (best.id >= f.w.c.unpadded_vocab) return error.InvalidSample;
        try generated.append(a, best.id);
        if (best.id == 200006 or step + 1 == limit) break;
        try f.token(s, best.id);
        forwards += 1;
        try f.stream.synchronize();
        try cuda.memory.download(f.d, f.b.capped, std.mem.sliceAsBytes(host));
    }
    const seconds = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) * 1e-9;
    var text: std.Io.Writer.Allocating = .init(a);
    defer text.deinit();
    try std.json.Stringify.value(.{ .ids = generated.items, .decode_forwards = forwards, .decode_seconds = seconds, .decode_tokens_per_second = @as(f64, @floatFromInt(forwards)) / seconds, .greedy = true, .drafts = false, .context_limit = 1024, .native_server = false }, .{}, &text.writer);
    const path = try std.fmt.allocPrint(a, "{s}/rank{d}-generation.json", .{ out_dir, f.w.p.rank });
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text.written() });
    std.debug.print("INKLING2_NATIVE_GENERATE_PASS rank={d} generated={d} decode_forwards={d} seconds={d:.6}\n", .{ f.w.p.rank, generated.items.len, forwards, seconds });
}
