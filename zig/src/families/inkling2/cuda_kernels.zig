//! Inkling2's experimental CUDA ABI over TensorFold's shared driver, modules and launch path.
const std = @import("std");
const cuda = @import("cuda");

pub const Op = enum { rms, attnres, sconv, add, swiglu, route, dense, quant, qmv, qr, kv, relative, scores, attend, embed, merge, softcap, quant_norm, rms_rows, attnres_rows, sconv_window, add_rows, situ_rows, route_rows, merge_rows, softcap_rows, qr_rows, kv_rows, relative_rows, scores_rows, attend_rows };
/// The fatbin each op lives in: rows, quant or attention, in Op declaration order.
const op_module = [_]u8{ 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2 };
fn int(n: usize) c_int {
    return @intCast(n);
}
fn blocks(n: usize) u32 {
    return @intCast((n + 255) / 256);
}

pub const Kernels = struct {
    mods: [3]cuda.Module,
    functions: [op_module.len]cuda.Function,

    pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Kernels {
        var k: Kernels = undefined;
        var made: usize = 0;
        errdefer for (k.mods[0..made]) |*m| m.unload();
        for ([_][]const u8{ "inkling2_rows", "inkling2_quant", "inkling2_attention" }, 0..) |name, i| {
            const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ dir, name });
            defer a.free(path);
            const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 26));
            defer a.free(raw);
            const image = try a.alignedAlloc(u8, .@"8", raw.len);
            defer a.free(image);
            @memcpy(image, raw);
            k.mods[i] = try cuda.Module.load(d, image);
            made += 1;
        }
        inline for (@typeInfo(Op).@"enum".field_names, 0..) |name, i| {
            k.functions[i] = try k.mods[op_module[i]].function("tf_ink2_" ++ name);
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (&k.mods) |*m| m.unload();
        k.* = undefined;
    }

    fn launch(k: *const Kernels, s: cuda.Stream, op: Op, grid: cuda.Dim3, params: anytype) !void {
        var args: cuda.Args = .{};
        inline for (params) |p| args.add(p);
        try cuda.launch.launch(k.functions[@backingInt(op)], .{ .grid = grid, .block = .{ .x = 256 } }, s, &args);
    }

    pub fn rms(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, y: u64, H: usize, eps: f32) !void {
        if (H == 0 or H > 4608) return error.BadHidden;
        try k.launch(s, .rms, .{ .x = 1 }, .{ x, w, y, int(H), eps });
    }
    pub fn residual(k: *const Kernels, s: cuda.Stream, states: u64, partial: u64, query: u64, y: u64, H: usize, depth: usize, queries: usize, eps: f32) !void {
        if (H == 0 or H > 4608 or depth > 8 or (queries != 1 and queries != 3)) return error.BadResidualShape;
        try k.launch(s, .attnres, .{ .x = @intCast(queries) }, .{ states, partial, query, y, int(H), int(depth), int(queries), eps });
    }
    pub fn conv(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, hist: u64, dyn: u64, y: u64, H: usize, D: usize, dynamic: bool, rounded: bool) !void {
        if (H == 0 or H > 4608 or (dynamic and (D == 0 or H % D != 0 or dyn == 0))) return error.BadConvShape;
        const scale: f32 = if (dynamic) 1.0 / 3.0 else 1.0 / @sqrt(@as(f32, 5));
        try k.launch(s, .sconv, .{ .x = 1 }, .{ x, w, hist, dyn, y, int(H), int(D), @as(c_int, @intFromBool(dynamic)), scale, @as(c_int, @intFromBool(rounded)) });
    }
    pub fn add(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, z: u64, H: usize, replace: bool, rounded: bool) !void {
        if (H == 0 or H > 4608) return error.BadHidden;
        try k.launch(s, .add, .{ .x = blocks(H) }, .{ x, y, z, int(H), @as(c_int, @intFromBool(replace)), @as(c_int, @intFromBool(rounded)) });
    }
    pub fn situ(k: *const Kernels, s: cuda.Stream, gu: u64, y: u64, I: usize, gamma: f32) !void {
        if (I == 0 or I > 2304 or !std.math.isFinite(gamma)) return error.BadExpertShape;
        try k.launch(s, .swiglu, .{ .x = blocks(I) }, .{ gu, y, int(I), gamma });
    }
    pub fn route(k: *const Kernels, s: cuda.Stream, logits: u64, bias: u64, ids: u64, weights: u64, E: usize, topk: usize, scale: f32) !void {
        if (E != 384 or topk != 8) return error.BadRouterShape;
        try k.launch(s, .route, .{ .x = 1 }, .{ logits, bias, ids, weights, int(E), int(topk), scale });
    }
    pub fn dense(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, y: u64, N: usize, K: usize, bf16: bool) !void {
        if (N == 0 or N > 50048 or K == 0 or K > 6144) return error.BadMatrixShape;
        try k.launch(s, .dense, .{ .x = @intCast(N) }, .{ x, w, y, int(N), int(K), @as(c_int, @intFromBool(bf16)) });
    }
    pub fn quant(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, K: usize, mx: bool) !void {
        if (K == 0 or K > 4608 or K % 32 != 0) return error.BadQuantShape;
        try k.launch(s, .quant, .{ .x = 1 }, .{ x, y, int(K), @as(c_int, @intFromBool(mx)) });
    }
    pub fn quantNorm(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, normalized: u64, amax: u64, K: usize, mx: bool, rows: usize) !void {
        if (K == 0 or K > 4608 or K % 32 != 0 or rows == 0 or rows > 8) return error.BadQuantShape;
        try k.launch(s, .quant_norm, .{ .x = @intCast(rows) }, .{ x, y, normalized, amax, int(K), @as(c_int, @intFromBool(mx)), int(rows) });
    }
    pub fn qmv(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, scales: u64, amax: u64, denom: u64, y: u64, N: usize, K: usize, mx: bool, bf16: bool) !void {
        if (N != 4608 or (K != 2304 and K != 4608)) return error.BadExpertShape;
        if (!mx and (amax == 0 or denom == 0)) return error.MissingOuterScale;
        try k.launch(s, .qmv, .{ .x = @intCast(N) }, .{ x, w, scales, amax, denom, y, int(N), int(K), @as(c_int, @intFromBool(mx)), @as(c_int, @intFromBool(bf16)) });
    }
    pub fn qr(k: *const Kernels, s: cuda.Stream, q: u64, r: u64, qs: u64, rs: u64, oq: u64, orr: u64, heads: usize, D: usize, R: usize, qclip: f32, rclip: f32, tau: f32, eps: f32) !void {
        if (heads != 12 or D != 128 or R != 16) return error.BadAttentionShape;
        try k.launch(s, .qr, .{ .x = @intCast(heads) }, .{ q, r, qs, rs, oq, orr, int(D), int(R), qclip, rclip, tau, eps });
    }
    pub fn kv(k: *const Kernels, s: cuda.Stream, kin: u64, vin: u64, kc: u64, vc: u64, D: usize, heads: usize, pos: usize, capacity: usize, clip: f32, eps: f32) !void {
        if (D != 128 or heads != 2 or pos >= capacity) return error.BadAttentionShape;
        try k.launch(s, .kv, .{ .x = @intCast(heads) }, .{ kin, vin, kc, vc, int(D), int(heads), int(pos), clip, eps });
    }
    pub fn relative(k: *const Kernels, s: cuda.Stream, r: u64, w: u64, out: u64) !void {
        try k.launch(s, .relative, .{ .x = 4, .y = 12 }, .{ r, w, out, @as(c_int, 16), @as(c_int, 1024) });
    }
    pub fn scores(k: *const Kernels, s: cuda.Stream, q: u64, kc: u64, rel: u64, out: u64, capacity: usize, pos: usize, local: bool) !void {
        if (pos >= capacity or capacity > 32768) return error.ContextFull;
        try k.launch(s, .scores, .{ .x = blocks(pos + 1), .y = 12 }, .{ q, kc, rel, out, @as(c_int, 12), @as(c_int, 2), @as(c_int, 128), @as(c_int, 1024), int(capacity), int(pos), @as(c_int, if (local) 1024 else 0) });
    }
    pub fn attend(k: *const Kernels, s: cuda.Stream, scores_: u64, sink: u64, vc: u64, gate: u64, out: u64, capacity: usize, pos: usize, local: bool) !void {
        if (pos >= capacity or capacity > 32768) return error.ContextFull;
        try k.launch(s, .attend, .{ .x = 12 }, .{ scores_, sink, vc, gate, out, @as(c_int, 12), @as(c_int, 2), @as(c_int, 128), int(capacity), int(pos), @as(c_int, if (local) 1024 else 0) });
    }
    pub fn embed(k: *const Kernels, s: cuda.Stream, table: u64, out: u64, id: u32, first: u32, valid: u32, H: usize) !void {
        if (id >= 200058 or H != 4608) return error.BadToken;
        try k.launch(s, .embed, .{ .x = blocks(H) }, .{ table, out, @as(c_int, @intCast(id)), @as(c_int, @intCast(first)), @as(c_int, @intCast(valid)), int(H) });
    }
    pub fn merge(k: *const Kernels, s: cuda.Stream, parts: u64, out: u64, H: usize, count: usize, bf16: bool) !void {
        if (H == 0 or H > 4608 or count == 0 or count > 8) return error.BadMergeShape;
        try k.launch(s, .merge, .{ .x = blocks(H) }, .{ parts, out, int(H), int(count), @as(c_int, @intFromBool(bf16)) });
    }
    pub fn mergeRanksRows(k: *const Kernels, s: cuda.Stream, parts: u64, out: u64, H: usize, rows: usize, rounded: bool) !void {
        if (H != 4608 or rows == 0 or rows > 8) return error.BadMergeShape;
        const n = rows * H;
        try k.launch(s, .merge, .{ .x = blocks(n) }, .{ parts, out, int(n), @as(c_int, 4), @as(c_int, @intFromBool(rounded)) });
    }
    pub fn softcap(k: *const Kernels, s: cuda.Stream, logits: u64, out: u64, count: usize, valid: usize, cap: f32) !void {
        if (valid > count or count > 50048 or cap <= 0) return error.BadVocabulary;
        try k.launch(s, .softcap, .{ .x = blocks(count) }, .{ logits, out, int(count), int(valid), cap });
    }

    /// Windowed <=8-row kernels: per-row arithmetic identical to the one-token ops above.
    pub fn rmsRows(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, w: u64, y: u64, y_stride: usize, H: usize, rows: usize, eps: f32) !void {
        if (H == 0 or H > 4608 or rows == 0 or rows > 8) return error.BadHidden;
        try k.launch(s, .rms_rows, .{ .x = @intCast(rows) }, .{ x, @as(i64, @intCast(x_stride)), w, y, @as(i64, @intCast(y_stride)), int(H), int(rows), eps });
    }
    pub fn residualRows(k: *const Kernels, s: cuda.Stream, states: u64, partial: u64, query: u64, y: u64, H: usize, depth: usize, queries: usize, eps: f32, rows: usize) !void {
        if (H == 0 or H > 4608 or depth > 8 or (queries != 1 and queries != 3) or rows == 0 or rows > 8) return error.BadResidualShape;
        try k.launch(s, .attnres_rows, .{ .x = @intCast(queries * rows) }, .{ states, partial, query, y, int(H), int(depth), int(queries), @as(c_int, 8), eps, int(rows) });
    }
    pub fn convWindow(k: *const Kernels, s: cuda.Stream, x: u64, w: u64, hist: u64, dyn: u64, y: u64, H: usize, D: usize, dynamic: bool, rounded: bool, rows: usize, snap: u64, snap_stride: usize) !void {
        if (H == 0 or H > 4608 or rows == 0 or rows > 8 or (dynamic and (D == 0 or H % D != 0 or dyn == 0))) return error.BadConvShape;
        const scale: f32 = if (dynamic) 1.0 / 3.0 else 1.0 / @sqrt(@as(f32, 5));
        try k.launch(s, .sconv_window, .{ .x = 1 }, .{ x, w, hist, dyn, y, snap, int(H), int(D), @as(c_int, @intFromBool(dynamic)), scale, @as(c_int, @intFromBool(rounded)), int(rows), @as(i64, @intCast(snap_stride)) });
    }
    pub fn addRows(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, z: u64, H: usize, rows: usize, replace: bool, rounded: bool) !void {
        if (H == 0 or H > 4608 or rows == 0 or rows > 8) return error.BadHidden;
        try k.launch(s, .add_rows, .{ .x = blocks(rows * H) }, .{ x, y, z, int(H), int(rows), @as(c_int, @intFromBool(replace)), @as(c_int, @intFromBool(rounded)) });
    }
    pub fn situRows(k: *const Kernels, s: cuda.Stream, gu: u64, y: u64, I: usize, gammas: u64, row_map: u64, rows: usize) !void {
        if (I == 0 or I > 2304 or rows == 0 or rows > 8) return error.BadExpertShape;
        try k.launch(s, .situ_rows, .{ .x = blocks(rows * I) }, .{ gu, y, int(I), gammas, row_map, int(rows) });
    }
    pub fn routeRows(k: *const Kernels, s: cuda.Stream, logits: u64, bias: u64, ids: u64, weights: u64, E: usize, topk: usize, scale: f32, rows: usize) !void {
        if (E != 384 or topk != 8 or rows == 0 or rows > 8) return error.BadRouterShape;
        try k.launch(s, .route_rows, .{ .x = @intCast(rows) }, .{ logits, bias, ids, weights, int(E), int(topk), scale, int(rows) });
    }
    pub fn mergeRows(k: *const Kernels, s: cuda.Stream, parts: u64, out: u64, H: usize, count: usize, rows: usize, bf16: bool) !void {
        if (H == 0 or H > 4608 or count == 0 or count > 8 or rows == 0 or rows > 8) return error.BadMergeShape;
        try k.launch(s, .merge_rows, .{ .x = blocks(rows * H) }, .{ parts, out, int(H), int(count), int(rows), @as(c_int, @intFromBool(bf16)) });
    }
    pub fn softcapRows(k: *const Kernels, s: cuda.Stream, logits: u64, out: u64, count: usize, valid: usize, cap: f32, rows: usize) !void {
        if (valid > count or count > 50048 or cap <= 0 or rows == 0 or rows > 8) return error.BadVocabulary;
        try k.launch(s, .softcap_rows, .{ .x = blocks(rows * count) }, .{ logits, out, int(count), int(valid), cap, int(rows) });
    }
    pub fn qrRows(k: *const Kernels, s: cuda.Stream, q: u64, r: u64, qs: u64, rs: u64, oq: u64, orr: u64, heads: usize, D: usize, R: usize, taus: u64, qclip: f32, rclip: f32, eps: f32, rows: usize) !void {
        if (heads != 12 or D != 128 or R != 16 or rows == 0 or rows > 8) return error.BadAttentionShape;
        const stride = heads * D;
        const r_stride = heads * R;
        try k.launch(s, .qr_rows, .{ .x = @intCast(heads), .y = @intCast(rows) }, .{ q, r, qs, rs, oq, orr, int(D), int(R), int(stride), int(r_stride), qclip, rclip, taus, eps });
    }
    pub fn kvRows(k: *const Kernels, s: cuda.Stream, kin: u64, vin: u64, kc: u64, vc: u64, D: usize, heads: usize, base: usize, capacity: usize, clip: f32, eps: f32, rows: usize) !void {
        if (D != 128 or heads != 2 or rows == 0 or rows > 8 or base + rows > capacity) return error.BadAttentionShape;
        try k.launch(s, .kv_rows, .{ .x = @intCast(heads), .y = @intCast(rows) }, .{ kin, vin, kc, vc, int(D), int(heads), int(base), int(rows), clip, eps });
    }
    pub fn relativeRows(k: *const Kernels, s: cuda.Stream, r: u64, w: u64, out: u64, heads: usize, R: usize, extent: usize, rows: usize) !void {
        if (heads != 12 or R != 16 or extent != 1024 or rows == 0 or rows > 8) return error.BadAttentionShape;
        try k.launch(s, .relative_rows, .{ .x = 4, .y = @intCast(heads), .z = @intCast(rows) }, .{ r, w, out, @as(c_int, 16), @as(c_int, 1024), int(heads) });
    }
    pub fn scoresRows(k: *const Kernels, s: cuda.Stream, q: u64, kc: u64, rel: u64, out: u64, capacity: usize, base: usize, rows: usize, local: bool) !void {
        if (base + rows > capacity or capacity > 32768 or rows == 0 or rows > 8) return error.ContextFull;
        try k.launch(s, .scores_rows, .{ .x = blocks(base + rows), .y = 12, .z = @intCast(rows) }, .{ q, kc, rel, out, @as(c_int, 12), @as(c_int, 2), @as(c_int, 128), @as(c_int, 1024), int(capacity), int(base), int(rows), @as(c_int, if (local) 1024 else 0) });
    }
    pub fn attendRows(k: *const Kernels, s: cuda.Stream, scores_: u64, sink: u64, vc: u64, gate: u64, out: u64, capacity: usize, base: usize, rows: usize, local: bool) !void {
        if (base + rows > capacity or capacity > 32768 or rows == 0 or rows > 8) return error.ContextFull;
        try k.launch(s, .attend_rows, .{ .x = 12, .y = @intCast(rows) }, .{ scores_, sink, vc, gate, out, @as(c_int, 12), @as(c_int, 2), @as(c_int, 128), int(capacity), int(base), int(rows), @as(c_int, if (local) 1024 else 0) });
    }
};
