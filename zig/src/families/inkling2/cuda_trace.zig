//! Explicit diagnostic seams; disabled during timed inference.
const std = @import("std");
const cuda = @import("cuda");

pub const Tag = enum {
    embedding,
    entry,
    stack,
    norm_q,
    norm_k,
    norm_v,
    attn_out,
    attn_conv,
    mid,
    mlp_in,
    route_logits,
    route_ids,
    gamma,
    x_quant,
    moe_out,
    mlp_conv,
    exit,
    final_mix,
    final_norm,
    logits,
};

pub const Watch = struct {
    ctx: *anyopaque,
    call: *const fn (*anyopaque, u32, u32, Tag, u64, usize) anyerror!void,
};

pub const Dump = struct {
    a: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    dir: []const u8,

    pub fn watch(x: *Dump) Watch {
        return .{ .ctx = x, .call = save };
    }

    fn save(ctx: *anyopaque, row: u32, layer: u32, tag: Tag, ptr: u64, count: usize) !void {
        const x: *Dump = @ptrCast(@alignCast(ctx));
        try x.stream.synchronize();
        const bytes = try x.a.alloc(u8, count * 4);
        defer x.a.free(bytes);
        try cuda.memory.download(x.d, ptr, bytes);
        const path = try std.fmt.allocPrint(x.a, "{s}/r{d}-l{d}-{s}.{s}", .{ x.dir, row, layer, @tagName(tag), if (tag == .route_ids) "i32" else "f32" });
        defer x.a.free(path);
        try std.Io.Dir.cwd().writeFile(x.io, .{ .sub_path = path, .data = bytes });
    }
};
