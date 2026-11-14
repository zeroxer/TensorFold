//! Independent checks for normalized activation representation, stored FP4 layout and row-width exactness.
const std = @import("std");
const cuda = @import("cuda");
const Math = @import("cuda_lane.zig").Math;
const Kernels = @import("cuda_kernels.zig").Kernels;
fn bf(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% 0x7fff +% ((b >> 16) & 1)) >> 16);
}
fn f(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}
fn e4(x: u8) f64 {
    const e = (x & 127) >> 3;
    const m = x & 7;
    const v = if (e == 0) @as(f64, @floatFromInt(m)) / 512 else (1 + @as(f64, @floatFromInt(m)) / 8) * std.math.pow(f64, 2, @as(f64, @floatFromInt(@as(i32, e) - 7)));
    return if (x & 128 != 0) -v else v;
}
pub fn run(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, k: *const Kernels, dir: []const u8) !void {
    const K = 4608;
    const N = 128;
    const R = 8;
    var math = try Math.load(a, io, d, dir);
    defer math.deinit();
    const x = try a.alloc(f32, R * K);
    defer a.free(x);
    for (x, 0..) |*v, i| v.* = f(bf(@as(f32, @floatFromInt(@as(i32, @intCast(i % 73)) - 36)) / 7));
    var dx = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var raw = try cuda.DeviceBuffer.alloc(d, R * K * 4);
    defer raw.free();
    var normalized = try cuda.DeviceBuffer.alloc(d, R * K * 2);
    defer normalized.free();
    var amax = try cuda.DeviceBuffer.alloc(d, R * 4);
    defer amax.free();
    try k.quantNorm(s, dx.ptr, raw.ptr, normalized.ptr, amax.ptr, K, false, R);
    try s.synchronize();
    const q = try a.alloc(u16, R * K);
    defer a.free(q);
    try normalized.download(0, std.mem.sliceAsBytes(q));
    const y = try a.alloc(f32, R * K);
    defer a.free(y);
    try raw.download(0, std.mem.sliceAsBytes(y));
    var outer: [R]f32 = undefined;
    try amax.download(0, std.mem.asBytes(&outer));
    for (y, q, 0..) |value, code, i| {
        const reconstructed = f(code) * (outer[i / K] * (@as(f32, 1) / 1536));
        if (@as(u32, @bitCast(value)) != @as(u32, @bitCast(reconstructed))) return error.NormalizedQuantChangedValue;
    }
    const codes = try a.alloc(u8, N * K / 2);
    defer a.free(codes);
    const scales = try a.alloc(u8, N * K / 16);
    defer a.free(scales);
    for (codes, 0..) |*v, i| v.* = @truncate(i * 37 + i / (K / 2));
    for (scales, 0..) |*v, i| v.* = @intCast(0x30 + i % 24);
    var dc = try cuda.DeviceBuffer.fromHost(d, codes);
    defer dc.free();
    var ds = try cuda.DeviceBuffer.fromHost(d, scales);
    defer ds.free();
    var pc = try cuda.DeviceBuffer.alloc(d, codes.len);
    defer pc.free();
    var ps = try cuda.DeviceBuffer.alloc(d, scales.len);
    defer ps.free();
    try math.packer.fp4Lane(s, dc.ptr, ds.ptr, pc.ptr, ps.ptr, N, K);
    const w = cuda.nvfp4.weight(pc.ptr, ps.ptr, 1, N, K);
    var result = try cuda.DeviceBuffer.alloc(d, R * N * 4);
    defer result.free();
    var one = try cuda.DeviceBuffer.alloc(d, N * 4);
    defer one.free();
    try math.mmBf16(s, w, normalized.ptr, R, result.ptr, false);
    try s.synchronize();
    var got: [R * N]f32 = undefined;
    try result.download(0, std.mem.asBytes(&got));
    const values = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    var max_error: f64 = 0;
    for (0..R) |row| {
        try math.mmBf16(s, w, normalized.ptr + row * K * 2, 1, one.ptr, false);
        try s.synchronize();
        var solo: [N]f32 = undefined;
        try one.download(0, std.mem.asBytes(&solo));
        if (!std.mem.eql(u8, std.mem.asBytes(&solo), std.mem.sliceAsBytes(got[row * N ..][0..N]))) return error.LaneWidthChangedBits;
        for (0..N) |n| {
            var want: f64 = 0;
            var magnitude: f64 = 0;
            for (0..K) |col| {
                const byte = codes[n * (K / 2) + col / 2];
                const code = if (col % 2 == 0) byte & 15 else byte >> 4;
                const weight = values[code & 7] * (if (code & 8 != 0) @as(f64, -1) else 1) * e4(scales[n * (K / 16) + col / 16]);
                const term = @as(f64, f(q[row * K + col])) * weight;
                want += term;
                magnitude += @abs(term);
            }
            const error_abs = @abs(@as(f64, got[row * N + n]) - want);
            max_error = @max(max_error, error_abs);
            const epsilon = std.math.floatEps(f32) / 2;
            const gamma = (K + 32) * epsilon / (1 - (K + 32) * epsilon);
            if (!std.math.isFinite(got[row * N + n]) or error_abs > gamma * magnitude) return error.LaneFloat32Bound;
        }
    }
    std.debug.print("INKLING2_NORMALIZED_LANE_PASS rows=8 values={d} width_bits_equal=true fp64_max_abs={d}\n", .{ R * K, max_error });
}
