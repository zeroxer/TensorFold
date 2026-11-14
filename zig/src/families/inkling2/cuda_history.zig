//! Exact convolution-history commits for a target window; rejected KV rows are hidden by the restored position.
const cuda = @import("cuda");
const State = @import("cuda_state.zig").State;

pub const History = struct {
    storage: cuda.DeviceBuffer,
    stride: usize,
    start: u32 = 0,
    rows: u32 = 0,
    pub fn init(d: *const cuda.Driver, s: *const State) !History {
        const stride = s.history_bytes * s.layers.len;
        return .{ .storage = try cuda.DeviceBuffer.alloc(d, 9 * stride), .stride = stride };
    }
    pub fn deinit(h: *History) void {
        h.storage.free();
        h.* = undefined;
    }
    pub fn save(h: *History, s: *const State, stream: cuda.Stream, index: u32) !void {
        if (index > 8) return error.BadHistoryIndex;
        const copy: cuda.abi.Memcpy2D = .{
            .src = s.layers[0].hist_k,
            .src_pitch = s.storage.len / s.layers.len,
            .dst = h.storage.ptr + index * h.stride,
            .dst_pitch = s.history_bytes,
            .width = s.history_bytes,
            .height = s.layers.len,
        };
        try h.storage.d.check(h.storage.d.api.cuMemcpy2DAsync_v2(&copy, stream.handle), "save conv history rows");
    }
    pub fn begin(h: *History, s: *const State, stream: cuda.Stream) !void {
        h.start = s.pos;
        h.rows = 0;
        try h.save(s, stream, 0);
    }
    /// Convolutions write prefix snapshots; rows bounds the accepted path and slot 0 belongs to begin().
    pub fn keep(h: *History, s: *State, stream: cuda.Stream, n: u32) !void {
        if (n == 0 or n > h.rows) return error.BadAcceptedPrefix;
        const copy: cuda.abi.Memcpy2D = .{
            .src = h.storage.ptr + n * h.stride,
            .src_pitch = s.history_bytes,
            .dst = s.layers[0].hist_k,
            .dst_pitch = s.storage.len / s.layers.len,
            .width = s.history_bytes,
            .height = s.layers.len,
        };
        try h.storage.d.check(h.storage.d.api.cuMemcpy2DAsync_v2(&copy, stream.handle), "restore conv history rows");
        s.pos = h.start + n;
        h.rows = n;
    }
};
