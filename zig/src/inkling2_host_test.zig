//! Experimental Inkling2 host checks, rooted above the family and the shared template implementation.
const std = @import("std");
const template = @import("core/template/template.zig");
test {
    _ = @import("families/inkling2/dspark_config.zig");
    _ = @import("families/inkling2/dspark_schema.zig");
    _ = @import("families/inkling2/config.zig");
    _ = @import("families/inkling2/shard.zig");
    _ = @import("families/inkling2/weight_schema.zig");
}

test "unchanged Inkling2 chat template matches independent Jinja2 fixtures" {
    const a = std.testing.allocator;
    var p = try std.json.parseFromSlice(std.json.Value, a, @embedFile("families/inkling2/testdata/template-parity.json"), .{});
    defer p.deinit();
    for (p.value.array.items) |case| {
        const c = case.object;
        var diag: template.Diag = .{};
        defer if (diag.msg.len > 0) a.free(diag.msg);
        const got = template.renderChat(a, @embedFile("families/inkling2/testdata/chat_template.jinja"), c.get("messages").?, c.get("tools").?, c.get("context").?, true, &diag);
        if (c.get("expected")) |want| {
            const out = got catch |err| {
                std.debug.print("{s}: {s}\n", .{ c.get("name").?.string, diag.msg });
                return err;
            };
            defer a.free(out);
            try std.testing.expectEqualStrings(want.string, out);
        } else {
            if (got) |out| {
                a.free(out);
                return error.ExpectedRefusal;
            } else |err| {
                try std.testing.expectEqual(error.TemplateFailed, err);
                try std.testing.expectEqualStrings(c.get("error").?.string, diag.msg);
            }
        }
    }
}
