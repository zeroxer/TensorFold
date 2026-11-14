//! Float-filter parity cases derived from Python Jinja2, including the Inkling effort macro.
const std = @import("std");
const template = @import("template.zig");

fn expectRender(source: []const u8, vars: []const u8, want: []const u8) !void {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, vars, .{});
    defer parsed.deinit();
    var diag: template.Diag = .{};
    defer if (diag.msg.len > 0) a.free(diag.msg);
    const out = template.renderChat(a, source, .null, .null, parsed.value, true, &diag) catch |err| {
        std.debug.print("float render diagnostic: {s}\n", .{diag.msg});
        return err;
    };
    defer a.free(out);
    try std.testing.expectEqualStrings(want, out);
}

fn expectError(source: []const u8) !void {
    const a = std.testing.allocator;
    var diag: template.Diag = .{};
    defer if (diag.msg.len > 0) a.free(diag.msg);
    if (template.renderChat(a, source, .null, .null, .null, true, &diag)) |out| {
        a.free(out);
        return error.TestExpectedError;
    } else |err| try std.testing.expectEqual(error.TemplateFailed, err);
}

test "float filter converts numeric inputs and preserves negative zero" {
    try expectRender("{{ true|float }}|{{ 7|float }}|{{ ' -0.0 '|float }}|{{ '+1.25e2'|float }}", "{}", "1.0|7.0|-0.0|125.0");
}

test "float filter uses the requested typed default only on conversion errors" {
    try expectRender("{{ 'bad'|float }}|{{ none|float(default=9) }}|{{ []|float('fallback') }}", "{}", "0.0|9|fallback");
    try expectError("{{ missing|float }}");
    try expectError("{{ '1'|float(2,3) }}");
    try expectError("{{ '1'|float(2,default=3) }}");
    try expectError("{{ '1'|float(unknown=3) }}");
}

test "float filter accepts Python decimal spelling but not malformed underscores or hex" {
    try expectRender("{{ '1_000.5'|float }}|{{ '.5'|float }}|{{ '1.e2'|float }}|{{ '1__0'|float(3) }}|{{ '0x1'|float(4) }}", "{}", "1000.5|0.5|100.0|3|4");
}

test "float filter handles Unicode whitespace and decimal digits as Python does" {
    try expectRender("{{ '\u{a0}0.9\u{2003}'|float }}|{{ '\u{660}.\u{669}'|float }}|{{ '\u{ff11}\u{ff12}'|float }}", "{}", "0.9|0.9|12.0");
}

test "float filter renders the checkpoint effort macro without changing its number" {
    const source = "{% set eff = reasoning_effort|float %}{% if eff < 0.0 or eff > 0.99 %}" ++ "{{ raise_exception('range') }}{% endif %}Thinking effort level: {% if eff == 0.0 %}0{% else %}{{ eff }}{% endif %}";
    try expectRender(source, "{\"reasoning_effort\":0.9}", "Thinking effort level: 0.9");
    try expectRender(source, "{\"reasoning_effort\":0}", "Thinking effort level: 0");
}
