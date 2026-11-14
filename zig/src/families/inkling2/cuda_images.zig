//! Aligned image reads for the family's explicit development artifact directory.
const std = @import("std");
pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]align(8) u8 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ dir, name });
    defer a.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    defer a.free(raw);
    const image = try a.alignedAlloc(u8, .@"8", raw.len);
    @memcpy(image, raw);
    return image;
}
