//! Lossless runtime layouts for the maintained tensor-core lane; quantized code and scale values are unchanged.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("cuda_weights.zig");
const math = @import("cuda_lane.zig");

fn index(w: *weights.Weights, ptr: u64) !usize {
    for (w.buffers.items, 0..) |b, i| if (b.ptr == ptr) return i;
    return error.UnownedWeight;
}

pub fn target(w: *weights.Weights, m: *const math.Math, d: *const cuda.Driver, s: cuda.Stream) !void {
    var it = w.entries.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const old = e.value_ptr.*;
        if (old.lane != null or old.rank != 2 or old.dims[1] % 64 != 0) continue;
        const projection = std.mem.eql(u8, name, "model.llm.unembed.weight") or
            std.mem.endsWith(u8, name, "mlp.gate.weight") or
            std.mem.endsWith(u8, name, ".weight_proj") or
            (std.mem.indexOf(u8, name, ".attn.") != null and std.mem.endsWith(u8, name, ".weight") and
                std.mem.indexOf(u8, name, "sconv") == null);
        if (!projection) continue;
        const owner = try index(w, old.ptr);
        const keep_raw = std.mem.eql(u8, name, "model.llm.unembed.weight");
        if (keep_raw) try w.buffers.ensureUnusedCapacity(w.a, 1);
        const packed_weight = try m.bf16Pack(d, s, old.ptr, old.dims[0], old.dims[1]);
        if (keep_raw) {
            w.buffers.appendAssumeCapacity(packed_weight.storage);
            w.bytes += packed_weight.storage.len;
        } else {
            w.buffers.items[owner].free();
            w.buffers.items[owner] = packed_weight.storage;
            w.bytes = w.bytes - old.bytes + packed_weight.storage.len;
            e.value_ptr.ptr = packed_weight.storage.ptr;
            e.value_ptr.bytes = packed_weight.storage.len;
        }
        e.value_ptr.lane = packed_weight.weight;
    }
    for (0..w.c.layers) |l| for ([_]bool{ true, false }) |up| {
        var wn: [160]u8 = undefined;
        var sn: [160]u8 = undefined;
        const wname = try std.fmt.bufPrint(&wn, "model.llm.layers.{d}.mlp.experts.{s}_weight", .{ l, if (up) "w13" else "w2" });
        const sname = try std.fmt.bufPrint(&sn, "model.llm.layers.{d}.mlp.experts.{s}_weight_scale", .{ l, if (up) "w13" else "w2" });
        const code = w.entries.getPtr(wname) orelse continue;
        const scales = w.entries.getPtr(sname) orelse return error.MissingExpertScale;
        if (code.lane != null) continue;
        const ci = try index(w, code.ptr);
        const si = try index(w, scales.ptr);
        const n = code.dims[0] * code.dims[1];
        const k: usize = if (up) 4608 else 2304;
        const mx = w.c.expertFormat(@intCast(l)) == .mxfp8;
        var pc = try cuda.DeviceBuffer.alloc(d, code.bytes);
        errdefer pc.free();
        var ps = try cuda.DeviceBuffer.alloc(d, scales.bytes);
        errdefer ps.free();
        if (mx) {
            try m.packer.fp8Lane(s, code.ptr, pc.ptr, n, k);
            try m.go(s, .pack_mx_scales, .{ .x = math.blocks(ps.len) }, .{ scales.ptr, ps.ptr, math.int(n), math.int(k), @as(i64, @intCast(ps.len)) });
        } else try m.packer.fp4Lane(s, code.ptr, scales.ptr, pc.ptr, ps.ptr, n, k);
        try s.synchronize();
        w.buffers.items[ci].free();
        w.buffers.items[si].free();
        w.buffers.items[ci] = pc;
        w.buffers.items[si] = ps;
        code.ptr = pc.ptr;
        scales.ptr = ps.ptr;
        code.lane = if (mx) cuda.mxfp8.weight(pc.ptr, ps.ptr, 4608, k) else cuda.nvfp4.weight(pc.ptr, ps.ptr, 1, 4608, k);
    };
}
