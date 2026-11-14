//! Rank-zero commands for the shared lane backend; followers execute the same native operations.
const std = @import("std");
const cuda = @import("cuda");
pub const Kind = enum(u32) { reset = 1, prefill, verify, keep, draft_open, absorb, hold, draft_close, stop };
pub const Header = extern struct { kind: u32, n: u32 = 0, a: u32 = 0, b: u32 = 0 };
pub const Link = struct {
    a: std.mem.Allocator,
    group: *const cuda.tp_group.Group,
    stream: cuda.Stream,
    storage: cuda.DeviceBuffer,
    ids: []u32,
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, g: *const cuda.tp_group.Group, s: cuda.Stream) !Link {
        var storage = try cuda.DeviceBuffer.alloc(d, 16 + 1032 * 4);
        errdefer storage.free();
        return .{ .a = a, .group = g, .stream = s, .storage = storage, .ids = try a.alloc(u32, 1032) };
    }
    pub fn deinit(l: *Link) void {
        l.storage.free();
        l.a.free(l.ids);
        l.* = undefined;
    }
    fn post(l: *Link, n: usize) !void {
        try l.group.lib.check(l.group.lib.api.ncclBroadcast(l.storage.ptr, l.storage.ptr, n, .u8, 0, l.group.comm, l.stream.handle), "Inkling2 command");
    }
    pub fn send(l: *Link, kind: Kind, a: u32, b: u32, ids: []const u32) !void {
        if (l.group.rank != 0 or ids.len > l.ids.len) return error.BadCommand;
        const h: Header = .{ .kind = @backingInt(kind), .n = @intCast(ids.len), .a = a, .b = b };
        try l.storage.uploadAsync(0, std.mem.asBytes(&h), l.stream.handle);
        if (ids.len > 0) try l.storage.uploadAsync(16, std.mem.sliceAsBytes(ids), l.stream.handle);
        try l.post(16);
        if (ids.len > 0) try l.post(16 + ids.len * 4);
    }
    pub fn receive(l: *Link) !struct { h: Header, ids: []const u32 } {
        try l.post(16);
        try l.stream.synchronize();
        var h: Header = undefined;
        try l.storage.download(0, std.mem.asBytes(&h));
        if (h.n > l.ids.len) return error.BadCommand;
        if (h.n > 0) {
            try l.post(16 + h.n * 4);
            try l.stream.synchronize();
            try l.storage.download(16, std.mem.sliceAsBytes(l.ids[0..h.n]));
        }
        return .{ .h = h, .ids = l.ids[0..h.n] };
    }
};
