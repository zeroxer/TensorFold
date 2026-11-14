//! Jinja float conversion: defaults on type/value errors; undefined and integer overflow still raise.
const std = @import("std");
const v = @import("value.zig");
const uni = @import("unicode.zig");
const methods = @import("methods.zig");
const eval = @import("eval.zig");
const Allocator = std.mem.Allocator;
const Value = v.Value;
const Error = v.Error;
const Kw = methods.Kw;

/// do_float(value, default=0.0): one optional positional argument or `default` keyword.
pub fn filter(c: *eval.Ctx, value: Value, pos: []const Value, kw: []const Kw) Error!Value {
    const bound = try methods.bind(&c.rt, "float", &.{"default"}, 0, pos, kw);
    if (v.isUndef(value)) return c.rt.fail("{s}", .{v.undefMsg(value)});
    const converted: ?f64 = switch (value) {
        .boolean => |b| @floatFromInt(@intFromBool(b)),
        .int => |i| @floatFromInt(i),
        .float => |f| f,
        // Python's float() of an int out of double range raises OverflowError, which the filter lets through.
        .big => |spelling| blk: {
            const f = std.fmt.parseFloat(f64, spelling) catch break :blk null;
            if (std.math.isInf(f)) return c.rt.fail("int too large to convert to float", .{});
            break :blk f;
        },
        .str => |s| try fromString(c.rt.a, s.s),
        // Every other type raises TypeError, and TypeError is what the filter replaces with its default.
        else => null,
    };
    return if (converted) |f| Value{ .float = f } else bound[0] orelse Value{ .float = 0.0 };
}

/// Python's float() on a str: null where Python raises ValueError.
fn fromString(a: Allocator, text: []const u8) Error!?f64 {
    const raw = trim(text);
    var folded: std.ArrayList(u8) = .empty;
    defer folded.deinit(a);
    var p: usize = 0;
    while (p < raw.len) {
        const start = p;
        const cp = uni.next(raw, &p);
        if (decimal(cp)) |d| try folded.append(a, '0' + d) else try folded.appendSlice(a, raw[start..p]);
    }
    const s = folded.items;
    if (s.len == 0) return null;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    if (s[0] == '-') try out.append(a, '-');
    const rest = if (s[0] == '-' or s[0] == '+') s[1..] else s;
    if (eqLower(rest, "inf") or eqLower(rest, "infinity"))
        return if (s[0] == '-') -std.math.inf(f64) else std.math.inf(f64);
    if (eqLower(rest, "nan")) return std.math.nan(f64);
    // number ::= [digits] ["." digits] [exponent], with at least one digit before the exponent.
    var at: usize = 0;
    var nonzero = false;
    const int = try digits(a, &out, rest, at, &nonzero);
    if (int) |n| at = n;
    if (at < rest.len and rest[at] == '.') {
        at += 1;
        const dot = out.items.len;
        try out.append(a, '.');
        const frac = try digits(a, &out, rest, at, &nonzero);
        if (frac) |n| at = n;
        if (int == null and frac == null) return null; // "." carries no digit at all
        if (int == null) try out.insert(a, dot, '0'); // ".5" is spelled "0.5"
        if (frac == null) try out.append(a, '0'); // "5." and "1.e3" have the value of "5.0"
    } else if (int == null) return null; // no digit and no fraction part: "+", "e5", ""
    var exp_neg = false;
    if (at < rest.len and (rest[at] == 'e' or rest[at] == 'E')) {
        at += 1;
        try out.append(a, 'e');
        if (at < rest.len and (rest[at] == '+' or rest[at] == '-')) {
            exp_neg = rest[at] == '-';
            if (exp_neg) try out.append(a, '-');
            at += 1;
        }
        const exp = try digits(a, &out, rest, at, null);
        if (exp == null) return null; // "1e", "1e_" and "1e+" have no exponent digits
        at = exp.?;
    }
    if (at != rest.len) return null; // trailing junk: "1 000", "1..2", "0x1F", "infinity8"
    return std.fmt.parseFloat(f64, out.items) catch null;
}

/// A decimal run with single underscores; returns its ending byte offset, or null when empty.
fn digits(a: Allocator, out: *std.ArrayList(u8), s: []const u8, from: usize, nonzero: ?*bool) Error!?usize {
    var i = from;
    while (i < s.len) {
        if (std.ascii.isDigit(s[i])) {
            if (nonzero) |n| n.* = n.* or s[i] != '0';
            try out.append(a, s[i]);
            i += 1;
        } else if (s[i] == '_' and i > from and i + 1 < s.len and std.ascii.isDigit(s[i + 1])) {
            i += 1;
        } else break;
    }
    return if (i > from) i else null;
}

/// Python's float() ignores leading and trailing whitespace, Unicode included.
fn trim(s: []const u8) []const u8 {
    var start: usize = 0;
    var end: usize = s.len;
    while (start < end) {
        var j = start;
        if (!uni.isSpace(uni.next(s, &j))) break;
        start = j;
    }
    while (end > start) {
        var b = end - 1;
        while (b > start and s[b] & 0xC0 == 0x80) b -= 1;
        var j = b;
        if (!uni.isSpace(uni.next(s, &j))) break;
        end = b;
    }
    return s[start..end];
}

fn eqLower(s: []const u8, comptime lower: []const u8) bool {
    if (s.len != lower.len) return false;
    for (s, lower) |x, y| if (std.ascii.toLower(x) != y) return false;
    return true;
}

// Unicode 16 decimal alphabet starts; source and license: unicode_decimal.txt.
fn decimal(cp: u21) ?u8 {
    const zero = [_]u21{
        0x30,    0x660,   0x6f0,   0x7c0,   0x966,   0x9e6,   0xa66,   0xae6,   0xb66,   0xbe6,
        0xc66,   0xce6,   0xd66,   0xde6,   0xe50,   0xed0,   0xf20,   0x1040,  0x1090,  0x17e0,
        0x1810,  0x1946,  0x19d0,  0x1a80,  0x1a90,  0x1b50,  0x1bb0,  0x1c40,  0x1c50,  0xa620,
        0xa8d0,  0xa900,  0xa9d0,  0xa9f0,  0xaa50,  0xabf0,  0xff10,  0x104a0, 0x10d30, 0x10d40,
        0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x116d0,
        0x116da, 0x11730, 0x118e0, 0x11950, 0x11bf0, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16130,
        0x16a60, 0x16ac0, 0x16b50, 0x16d70, 0x1ccf0, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6,
        0x1e140, 0x1e2f0, 0x1e4f0, 0x1e5f1, 0x1e950, 0x1fbf0,
    };
    for (zero) |z| {
        if (cp < z) return null;
        if (cp - z < 10) return @intCast(cp - z);
    }
    return null;
}
