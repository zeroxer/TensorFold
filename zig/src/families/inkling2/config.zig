//! Inkling 2's checkpoint geometry; parsing does not advertise a working backend.
const std = @import("std");

pub const model_type = "inkling2";
pub const architecture = "Inkling2ForConditionalGeneration";
pub const ExpertFormat = enum { nvfp4, mxfp8 };

pub const Config = struct {
    hidden: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    rel_dim: u32,
    rel_extent: u32,
    log_floor: u32,
    log_alpha: f32,
    vocab: u32,
    unpadded_vocab: u32,
    experts: u32,
    shared_experts: u32,
    gate_bias: bool,
    topk: u32,
    intermediate: u32,
    window: u32,
    residual_block: u32,
    conv_kernel: u32,
    trained_context: u32,
    local_mask: u64,
    rms_eps: f32,
    qk_clip: f32,
    rel_clip: f32,
    route_scale: f32,
    gate_cap: f32,
    up_cap: f32,
    logits_cap: f32,
    value_norm: bool,
    normalize_residual_values: bool,
    pre_norm_pre_affine: bool,
    final_norm_pre_affine: bool,
    has_vision: bool,
    has_draft: bool,

    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        defer a.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
        defer a.free(text);
        return parse(a, text);
    }

    pub fn parse(a: std.mem.Allocator, text: []const u8) !Config {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer parsed.deinit();
        return fromValue(parsed.value);
    }

    pub fn fromValue(root: std.json.Value) !Config {
        const outer = try object(root);
        const kind = outer.get("model_type") orelse return error.InvalidConfiguration;
        if (kind != .string or !std.mem.eql(u8, kind.string, model_type)) return error.WrongModelType;
        const arch = outer.get("architectures") orelse return error.InvalidConfiguration;
        if (arch != .array or arch.array.items.len != 1) return error.UnsupportedArchitecture;
        const name = arch.array.items[0];
        if (name != .string or !std.mem.eql(u8, name.string, architecture)) return error.UnsupportedArchitecture;
        const c = try object(outer.get("text_config") orelse return error.InvalidConfiguration);
        const count = try uint(c, "num_hidden_layers");
        if (count == 0 or count > 64) return error.UnsupportedGeometry;
        const ids = c.get("local_layer_ids") orelse return error.InvalidConfiguration;
        if (ids != .array) return error.InvalidConfiguration;
        var mask: u64 = 0;
        for (ids.array.items) |id| {
            if (id != .integer or id.integer < 0 or id.integer >= count) return error.InvalidLocalLayer;
            const bit = @as(u64, 1) << @as(u6, @intCast(id.integer));
            if (mask & bit != 0) return error.DuplicateLocalLayer;
            mask |= bit;
        }
        if (try boolean(c, "q_bias", false) or try boolean(c, "o_bias", false)) return error.UnsupportedBias;
        const result: Config = .{
            .hidden = try uint(c, "hidden_size"),
            .layers = count,
            .heads = try uint(c, "num_attention_heads"),
            .kv_heads = try uint(c, "num_key_value_heads"),
            .head_dim = try uint(c, "head_dim"),
            .rel_dim = try uint(c, "d_rel"),
            .rel_extent = try uint(c, "rel_extent"),
            .log_floor = try uint(c, "log_scaling_n_floor"),
            .log_alpha = try positive(c, "log_scaling_alpha"),
            .vocab = try uint(c, "vocab_size"),
            .unpadded_vocab = try uint(c, "unpadded_vocab_size"),
            .experts = try uint(c, "n_routed_experts"),
            .shared_experts = try uint(c, "n_shared_experts"),
            .gate_bias = try boolean(c, "use_gate_bias", false),
            .topk = try uint(c, "num_experts_per_tok"),
            .intermediate = try uint(c, "intermediate_size"),
            .window = try uint(c, "sliding_window_size"),
            .residual_block = try uint(c, "n_layers_per_attn_res_block"),
            .conv_kernel = try uint(c, "sconv_kernel_size"),
            .trained_context = try uint(c, "model_max_length"),
            .local_mask = mask,
            .rms_eps = try positive(c, "rms_norm_eps"),
            .qk_clip = try positive(c, "qk_l2_clip_threshold"),
            .rel_clip = try positive(c, "rel_l2_clip_threshold"),
            .route_scale = try positive(c, "route_scale"),
            .gate_cap = try positive(c, "mlp_gate_softcap"),
            .up_cap = try positive(c, "mlp_up_softcap"),
            .logits_cap = try positive(c, "final_logit_softcapping"),
            .value_norm = try boolean(c, "use_value_norm_no_gamma", true),
            .normalize_residual_values = try boolean(c, "attn_res_normalize_values", true),
            .pre_norm_pre_affine = try boolean(c, "pre_norm_pre_affine", false),
            .final_norm_pre_affine = try boolean(c, "final_norm_pre_affine", false),
            .has_vision = outer.get("vision_config") != null,
            .has_draft = outer.get("mtp_config") != null,
        };
        try result.validate();
        if (try uint(c, "swa_head_dim") != result.head_dim or
            try uint(c, "swa_num_attention_heads") != result.heads or
            try uint(c, "swa_num_key_value_heads") != result.kv_heads) return error.UnsupportedGeometry;
        return result;
    }

    pub fn validate(c: Config) !void {
        if (c.hidden == 0 or c.heads == 0 or c.kv_heads == 0 or c.head_dim == 0) return error.UnsupportedGeometry;
        if (c.heads % c.kv_heads != 0 or c.hidden % 32 != 0) return error.UnsupportedGeometry;
        if (c.experts == 0 or c.topk == 0 or c.topk > c.experts) return error.UnsupportedGeometry;
        if (c.intermediate == 0 or c.intermediate % 64 != 0) return error.UnsupportedGeometry;
        if (c.vocab == 0 or c.unpadded_vocab == 0 or c.unpadded_vocab > c.vocab) return error.InvalidVocabulary;
        if (c.window == 0 or c.residual_block == 0 or c.layers % c.residual_block != 0) return error.UnsupportedGeometry;
        if (c.conv_kernel != 4 or c.trained_context == 0) return error.UnsupportedGeometry;
        if (c.rel_dim == 0 or c.rel_extent == 0 or c.log_floor == 0) return error.UnsupportedGeometry;
    }

    pub fn local(c: Config, layer: u32) bool {
        std.debug.assert(layer < c.layers);
        return c.local_mask & (@as(u64, 1) << @as(u6, @intCast(layer))) != 0;
    }

    pub fn expertFormat(c: Config, layer: u32) ExpertFormat {
        std.debug.assert(layer < c.layers);
        return if (layer < 40) .nvfp4 else .mxfp8;
    }

    pub fn globalLayers(c: Config) u32 {
        return c.layers - @as(u32, @intCast(@popCount(c.local_mask)));
    }

    /// A geometry gate, not a claim that loading or inference has been qualified.
    pub fn admitReleased(c: Config, sm: u32, world: u32) !void {
        if (sm != 121) return error.Inkling2RequiresSm121;
        if (world != 4) return error.Inkling2RequiresFourRanks;
        if (c.hidden != 4608 or c.layers != 48 or c.heads != 48 or c.kv_heads != 8 or c.head_dim != 128 or
            c.rel_dim != 16 or c.rel_extent != 1024 or c.vocab != 201024 or c.unpadded_vocab != 200058 or
            c.intermediate != 2304 or c.experts != 384 or c.topk != 8 or c.window != 1024 or
            c.residual_block != 6 or c.globalLayers() != 8) return error.UnsupportedReleasedGeometry;
        for (0..c.layers) |i| {
            if (c.local(@intCast(i)) != (i % 6 != 5)) return error.UnsupportedReleasedGeometry;
        }
        if (!c.normalize_residual_values or !c.value_norm or c.pre_norm_pre_affine or c.final_norm_pre_affine or
            c.shared_experts != 0 or !c.gate_bias or c.gate_cap != 4 or c.up_cap != 25)
            return error.UnsupportedReleasedArithmetic;
    }
};

fn object(value: std.json.Value) !std.json.ObjectMap {
    return if (value == .object) value.object else error.InvalidConfiguration;
}

fn uint(o: std.json.ObjectMap, name: []const u8) !u32 {
    const value = o.get(name) orelse return error.InvalidConfiguration;
    if (value != .integer or value.integer < 0 or value.integer > std.math.maxInt(u32))
        return error.InvalidConfiguration;
    return @intCast(value.integer);
}

fn positive(o: std.json.ObjectMap, name: []const u8) !f32 {
    const value = o.get(name) orelse return error.InvalidConfiguration;
    const number: f64 = switch (value) {
        .float => value.float,
        .integer => @floatFromInt(value.integer),
        else => return error.InvalidConfiguration,
    };
    const out: f32 = @floatCast(number);
    if (!std.math.isFinite(out) or out <= 0) return error.InvalidConfiguration;
    return out;
}

fn boolean(o: std.json.ObjectMap, name: []const u8, fallback: bool) !bool {
    const value = o.get(name) orelse return fallback;
    return if (value == .bool) value.bool else error.InvalidConfiguration;
}

const fixture = @embedFile("testdata/config.json");

test "inkling2 config reads the released nested geometry and mixed layers" {
    const c = try Config.parse(std.testing.allocator, fixture);
    try std.testing.expectEqual(@as(u32, 4608), c.hidden);
    try std.testing.expectEqual(@as(u32, 8), c.globalLayers());
    try std.testing.expect(c.local(0) and !c.local(5) and c.local(40) and !c.local(47));
    try std.testing.expectEqual(ExpertFormat.nvfp4, c.expertFormat(39));
    try std.testing.expectEqual(ExpertFormat.mxfp8, c.expertFormat(40));
    try std.testing.expect(c.has_vision and c.has_draft);
    try c.admitReleased(121, 4);
    try std.testing.expectError(error.Inkling2RequiresFourRanks, c.admitReleased(121, 2));
}

test "inkling2 config refuses a different architecture and invalid head grouping" {
    var p = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, fixture, .{});
    defer p.deinit();
    const text = p.value.object.getPtr("text_config").?;
    text.object.getPtr("num_attention_heads").?.* = .{ .integer = 47 };
    try std.testing.expectError(error.UnsupportedGeometry, Config.fromValue(p.value));
    p.value.object.getPtr("model_type").?.* = .{ .string = "llama" };
    try std.testing.expectError(error.WrongModelType, Config.fromValue(p.value));
}

test "inkling2 config refuses duplicate local layers rather than changing cache roles" {
    var p = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, fixture, .{});
    defer p.deinit();
    const text = p.value.object.getPtr("text_config").?;
    const ids = text.object.getPtr("local_layer_ids").?;
    ids.array.items[1] = ids.array.items[0];
    try std.testing.expectError(error.DuplicateLocalLayer, Config.fromValue(p.value));
}

test "released gate rejects metadata whose arithmetic the native kernels do not implement" {
    var c = try Config.parse(std.testing.allocator, fixture);
    c.gate_cap = 5;
    try std.testing.expectError(error.UnsupportedReleasedArithmetic, c.admitReleased(121, 4));
    c.gate_cap = 4;
    c.shared_experts = 1;
    try std.testing.expectError(error.UnsupportedReleasedArithmetic, c.admitReleased(121, 4));
    c.shared_experts = 0;
    c.rel_dim = 32;
    try std.testing.expectError(error.UnsupportedReleasedGeometry, c.admitReleased(121, 4));
}
