//! Shared TensorFold tensor-core lanes and lossless packers for Inkling2 projections.
const std = @import("std");
const cuda = @import("cuda");
const images = @import("cuda_images.zig");

pub const Op = enum { to_bf16, from_bf16, round_bf16, fold, fuse_alpha, rnn_context_input, rnn_state_input, add_bias, add_float, pick, pick_merge, pack_mx_scales, quant_output, to_bf16_strided, gather_u16, gather_f32, scatter_f32, pick_merge_rows };

pub const Math = struct {
    lane: cuda.qmmf.Lane,
    packer: cuda.modelopt.Packer,
    bridge: cuda.Module,
    functions: [@typeInfo(Op).@"enum".field_names.len]cuda.Function,
    inputs: cuda.DeviceBuffer,
    parts: cuda.DeviceBuffer,

    pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Math {
        const q = try images.read(a, io, dir, "qmmf");
        defer a.free(q);
        var lane = try cuda.qmmf.Lane.loadImage(d, 12, q);
        errdefer lane.unload();
        const p = try images.read(a, io, dir, "modelopt_pack");
        defer a.free(p);
        var packer = try cuda.modelopt.Packer.loadImage(d, p);
        errdefer packer.unload();
        const code = try images.read(a, io, dir, "inkling2_bridge");
        defer a.free(code);
        var bridge = try cuda.Module.load(d, code);
        errdefer bridge.unload();
        var functions: [@typeInfo(Op).@"enum".field_names.len]cuda.Function = undefined;
        inline for (@typeInfo(Op).@"enum".field_names, 0..) |name, i|
            functions[i] = try bridge.function("tf_ink2_" ++ name);
        var input = try cuda.DeviceBuffer.alloc(d, 8 * 18432 * 2);
        errdefer input.free();
        const parts = try cuda.DeviceBuffer.alloc(d, 8 * 8 * 50048 * 4);
        return .{ .lane = lane, .packer = packer, .bridge = bridge, .functions = functions, .inputs = input, .parts = parts };
    }

    pub fn deinit(m: *Math) void {
        m.parts.free();
        m.inputs.free();
        m.bridge.unload();
        m.packer.unload();
        m.lane.unload();
        m.* = undefined;
    }

    pub fn go(m: *const Math, s: cuda.Stream, op: Op, grid: cuda.Dim3, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |v| a.add(v);
        try cuda.launch.launch(m.functions[@backingInt(op)], .{ .grid = grid, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn convert(m: *const Math, s: cuda.Stream, x: u64, y: u64, n: usize, to_bf16: bool) !void {
        try m.go(s, if (to_bf16) .to_bf16 else .from_bf16, .{ .x = blocks(n) }, .{ x, y, @as(i64, @intCast(n)) });
    }

    pub fn round(m: *const Math, s: cuda.Stream, x: u64, n: usize) !void {
        try m.go(s, .round_bf16, .{ .x = blocks(n) }, .{ x, @as(i64, @intCast(n)) });
    }

    pub fn mm(m: *const Math, s: cuda.Stream, w: cuda.qmmf.Weight, x: u64, rows: usize, out: u64, rounded: bool) !void {
        if (rows == 0 or rows > 8 or w.k > 18432 or w.n > 50048) return error.UnsupportedLaneShape;
        try m.convert(s, x, m.inputs.ptr, rows * w.k, true);
        try m.mmBf16(s, w, m.inputs.ptr, rows, out, rounded);
    }

    pub fn mmBf16(m: *const Math, s: cuda.Stream, w: cuda.qmmf.Weight, x: u64, rows: usize, out: u64, rounded: bool) !void {
        if (rows == 0 or rows > 8 or m.lane.partBytes(rows, w) > m.parts.len) return error.UnsupportedLaneShape;
        try m.lane.matmulOut(s, x, w.k, rows, w, out, .f32, m.parts.ptr);
        if (rounded) try m.round(s, out, rows * w.n);
    }

    /// Like mm, with input rows x_stride floats apart; packing preserves the flat input's BF16 bytes.
    pub fn mmStrided(m: *const Math, s: cuda.Stream, w: cuda.qmmf.Weight, x: u64, x_stride: usize, rows: usize, out: u64, rounded: bool) !void {
        if (rows == 0 or rows > 8 or w.k > 18432 or w.n > 50048 or x_stride < w.k) return error.UnsupportedLaneShape;
        try m.go(s, .to_bf16_strided, .{ .x = blocks(rows * w.k) }, .{ x, m.inputs.ptr, @as(i64, @intCast(x_stride)), int(w.k), @as(i64, @intCast(rows)) });
        try m.mmBf16(s, w, m.inputs.ptr, rows, out, rounded);
    }

    /// dst[g*k ..] = src[row_map[g]*k ..] for the grouped expert GEMMs (16-bit activations).
    pub fn gatherU16(m: *const Math, s: cuda.Stream, src: u64, dst: u64, map: u64, k: usize, rows: usize) !void {
        if (rows == 0 or rows > 8 or k == 0) return error.UnsupportedLaneShape;
        try m.go(s, .gather_u16, .{ .x = blocks(rows * k) }, .{ src, dst, map, int(k), @as(i64, @intCast(rows)) });
    }

    /// The same gather over 32-bit rows (amax scalars with k = 1, SiTU activations with k = I).
    pub fn gatherF32(m: *const Math, s: cuda.Stream, src: u64, dst: u64, map: u64, k: usize, rows: usize) !void {
        if (rows == 0 or rows > 8 or k == 0) return error.UnsupportedLaneShape;
        try m.go(s, .gather_f32, .{ .x = blocks(rows * k) }, .{ src, dst, map, int(k), @as(i64, @intCast(rows)) });
    }

    /// dst[(row_map[g]*picks + pick_map[g])*k ..] = src[g*k ..] back into the per-pick expert parts.
    pub fn scatterF32(m: *const Math, s: cuda.Stream, src: u64, dst: u64, row_map: u64, pick_map: u64, picks: usize, k: usize, rows: usize) !void {
        if (rows == 0 or rows > 8 or k == 0 or picks == 0 or picks > 8) return error.UnsupportedLaneShape;
        try m.go(s, .scatter_f32, .{ .x = blocks(rows * k) }, .{ src, dst, row_map, pick_map, int(picks), int(k), @as(i64, @intCast(rows)) });
    }

    /// One pick merge for every row of a window from the gathered [rank][row][2] pairs.
    pub fn pickMergeRows(m: *const Math, s: cuda.Stream, picks: u64, out: u64, world: usize, rows: usize) !void {
        if (rows == 0 or rows > 8 or world == 0 or world > 4) return error.UnsupportedLaneShape;
        try m.go(s, .pick_merge_rows, .{ .x = 1 }, .{ picks, out, int(world), int(rows) });
    }
    pub fn bf16Pack(m: *const Math, d: *const cuda.Driver, s: cuda.Stream, raw: u64, n: usize, k: usize) !Packed {
        if (k % 64 != 0) return error.BadPackedWidth;
        var storage = try cuda.DeviceBuffer.alloc(d, cuda.modelopt.padded(n) * k * 2);
        errdefer storage.free();
        try m.packer.bf16Lane(s, raw, storage.ptr, n, k);
        try s.synchronize();
        return .{ .storage = storage, .weight = cuda.modelopt.bf16Weight(storage.ptr, n, k) };
    }
};

pub const Packed = struct { storage: cuda.DeviceBuffer, weight: cuda.qmmf.Weight };
pub fn blocks(n: usize) u32 {
    return @intCast((n + 255) / 256);
}
pub fn int(n: usize) c_int {
    return @intCast(n);
}
