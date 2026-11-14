//! Private experimental Inkling2 development targets; this does not register serving capabilities.
const std = @import("std");

pub fn hostTests(b: *std.Build, all: *std.Build.Step) void {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/inkling2_host_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
    });
    const run = b.addRunArtifact(b.addTest(.{ .root_module = m }));
    b.step("test-inkling2-host", "Inkling2 geometry, EP4, stored tensors and original template (no GPU)")
        .dependOn(&run.step);
    all.dependOn(&run.step);
}

pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module) void {
    const m = b.createModule(.{
        .root_source_file = b.path("zig/src/families/inkling2/cuda_probe_main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } },
    });
    const exe = b.addExecutable(.{ .name = "tf-inkling2-cuda", .root_module = m });
    const install = b.addInstallArtifact(exe, .{});
    b.step("tf-inkling2-cuda", "Experimental Inkling2 CUDA ABI and unchanged checkpoint shard probe")
        .dependOn(&install.step);
}
