//! The ONNX Runtime C API, as much of it as this program needs.
//!
//! Everything in the C API is a function pointer hanging off one `OrtApi`
//! struct, and every call returns an `OrtStatus*` that is null on success and
//! an owned error object otherwise. Both of those are wrapped here once, so the
//! rest of the program can call `session.run(...)` and get a Zig error.
//!
//! The interesting part is `Devices`. The CPU provider is compiled into the
//! runtime, but an Intel NPU is only reachable through OpenVINO, which the
//! runtime dlopens as a separate library at a path given to it. Registering
//! that library makes the NPU (and the iGPU beside it) show up in the runtime's
//! device list, and a session is then pinned to one by handing its
//! `OrtEpDevice` to the session options.

const std = @import("std");

pub const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

/// The API table, resolved once by `init`. Every wrapper below reaches through
/// it, so nothing here works before that call.
var api: *const c.OrtApi = undefined;

/// CPU memory descriptor for the tensors handed in and out. Input and output
/// both live in this process's own memory whatever device the graph runs on --
/// the provider copies across the boundary itself.
var cpu_memory: *c.OrtMemoryInfo = undefined;

pub const Error = error{
    /// The runtime rejected a call. `lastError()` has what it said.
    OnnxRuntime,
    /// The runtime is older than the API version this was compiled against.
    UnsupportedApiVersion,
};

/// The runtime's own message for the most recent failure. Kept in a fixed
/// buffer rather than allocated: it is only ever read to be printed, and a
/// failure path is a bad place to need an allocator.
var error_buf: [1024]u8 = undefined;
var error_len: usize = 0;

pub fn lastError() []const u8 {
    return error_buf[0..error_len];
}

fn check(status: ?*c.OrtStatus) Error!void {
    const s = status orelse return;
    defer api.ReleaseStatus.?(s);

    const message = std.mem.span(api.GetErrorMessage.?(s));
    error_len = @min(message.len, error_buf.len);
    @memcpy(error_buf[0..error_len], message[0..error_len]);
    return Error.OnnxRuntime;
}

/// Resolves the API table and the CPU memory descriptor. Call once, before
/// anything else in this file.
pub fn init() Error!void {
    const base = c.OrtGetApiBase();
    api = base.*.GetApi.?(c.ORT_API_VERSION) orelse return Error.UnsupportedApiVersion;

    var info: ?*c.OrtMemoryInfo = null;
    try check(api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &info));
    cpu_memory = info.?;
}

/// The runtime's version string, e.g. "1.29.0".
pub fn version() []const u8 {
    return std.mem.span(c.OrtGetApiBase().*.GetVersionString.?());
}

/// What a session is running on. `.cpu` is the provider built into the runtime;
/// the other two come from OpenVINO and only exist once its library has been
/// registered.
pub const DeviceKind = enum {
    npu,
    gpu,
    cpu,

    /// What OpenVINO calls this device in its own configuration.
    pub fn openvinoName(self: DeviceKind) [:0]const u8 {
        return switch (self) {
            .npu => "NPU",
            .gpu => "GPU",
            .cpu => "CPU",
        };
    }

    fn hardwareType(self: DeviceKind) c.OrtHardwareDeviceType {
        return switch (self) {
            .npu => c.OrtHardwareDeviceType_NPU,
            .gpu => c.OrtHardwareDeviceType_GPU,
            .cpu => c.OrtHardwareDeviceType_CPU,
        };
    }
};

/// A device the runtime is willing to run a graph on, together with the
/// execution provider that reaches it.
pub const Device = struct {
    kind: DeviceKind,
    /// The device's PCI id. For an Intel NPU this is the only thing that says
    /// which generation of the part it is.
    id: u32,
    handle: *const c.OrtEpDevice,
};

/// One key/value pair configuring an execution provider. Which keys a provider
/// accepts, and what it does with them, is its own business.
pub const Option = struct {
    key: [*:0]const u8,
    value: [*:0]const u8,
};

pub const Env = struct {
    ptr: *c.OrtEnv,

    pub fn init(name: [:0]const u8) Error!Env {
        var ptr: ?*c.OrtEnv = null;
        try check(api.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, name.ptr, &ptr));
        return .{ .ptr = ptr.? };
    }

    pub fn deinit(self: Env) void {
        api.ReleaseEnv.?(self.ptr);
    }

    /// Loads an execution provider library and lets it enumerate its devices.
    /// This is how the OpenVINO provider -- and with it the NPU -- becomes
    /// visible; the runtime dlopens `path` and never links it.
    pub fn registerProvider(self: Env, name: [:0]const u8, path: [:0]const u8) Error!void {
        try check(api.RegisterExecutionProviderLibrary.?(self.ptr, name.ptr, path.ptr));
    }

    /// The first enumerated device of the given kind, or null if the runtime
    /// has none. Registering OpenVINO adds the NPU and the iGPU here; without
    /// it the list is the CPU provider alone.
    pub fn find(self: Env, kind: DeviceKind) Error!?Device {
        var devices: [*c]const ?*const c.OrtEpDevice = undefined;
        var count: usize = 0;
        try check(api.GetEpDevices.?(self.ptr, @ptrCast(&devices), &count));

        for (0..count) |i| {
            const device = devices[i] orelse continue;
            const hardware = api.EpDevice_Device.?(device);
            if (hardware == null) continue;
            if (api.HardwareDevice_Type.?(hardware) != kind.hardwareType()) continue;
            return .{
                .kind = kind,
                .id = api.HardwareDevice_DeviceId.?(hardware),
                .handle = device,
            };
        }
        return null;
    }
};

/// A loaded graph, bound to one device.
pub const Session = struct {
    ptr: *c.OrtSession,
    /// What it was actually pinned to, which is not always what was asked for:
    /// `open` falls back to the CPU when a device refuses the graph.
    device: DeviceKind,

    /// Loads `model_path`, preferring `device` if one is given. A device that
    /// cannot compile the graph -- too large for the NPU, an operator the
    /// provider does not implement -- is not fatal: the session is reopened on
    /// the CPU and `device` records where it ended up. `on_fallback` is handed
    /// the runtime's complaint so the caller can say why.
    pub fn open(
        env: Env,
        model_path: [:0]const u8,
        device: ?Device,
        options: []const Option,
        on_fallback: ?*const fn (message: []const u8) void,
    ) Error!Session {
        if (device) |d| {
            if (tryOpen(env, model_path, d, options)) |ptr| {
                return .{ .ptr = ptr, .device = d.kind };
            } else |_| {
                if (on_fallback) |report| report(lastError());
            }
        }
        return .{ .ptr = try tryOpen(env, model_path, null, &.{}), .device = .cpu };
    }

    fn tryOpen(
        env: Env,
        model_path: [:0]const u8,
        device: ?Device,
        options: []const Option,
    ) Error!*c.OrtSession {
        var session_options: ?*c.OrtSessionOptions = null;
        try check(api.CreateSessionOptions.?(&session_options));
        defer api.ReleaseSessionOptions.?(session_options);

        // The exports carry a symbolic batch dimension. A provider that
        // compiles a graph ahead of time -- which is every provider that is not
        // the CPU -- needs it pinned to a number before it can.
        try check(api.AddFreeDimensionOverrideByName.?(session_options, "batch_size", 1));

        if (device) |d| {
            var keys: [4][*:0]const u8 = undefined;
            var values: [4][*:0]const u8 = undefined;
            std.debug.assert(options.len <= keys.len);
            for (options, 0..) |option, i| {
                keys[i] = option.key;
                values[i] = option.value;
            }

            try check(api.SessionOptionsAppendExecutionProvider_V2.?(
                session_options,
                env.ptr,
                @ptrCast(&d.handle),
                1,
                @ptrCast(&keys),
                @ptrCast(&values),
                options.len,
            ));
        }

        var ptr: ?*c.OrtSession = null;
        try check(api.CreateSession.?(env.ptr, model_path.ptr, session_options, &ptr));
        return ptr.?;
    }

    pub fn deinit(self: Session) void {
        api.ReleaseSession.?(self.ptr);
    }

    /// Runs the graph. `inputs` are borrowed, `outputs` are filled with values
    /// the caller owns and must `deinit`.
    pub fn run(
        self: Session,
        input_names: []const [*:0]const u8,
        inputs: []const Value,
        output_names: []const [*:0]const u8,
        outputs: []Value,
    ) Error!void {
        std.debug.assert(input_names.len == inputs.len);
        std.debug.assert(output_names.len == outputs.len);

        // `Value` wraps the pointer the C API passes around, so both directions
        // are unwrapped into plain arrays of it here rather than relying on the
        // wrapper having the same layout.
        var raw_inputs: [8]?*const c.OrtValue = @splat(null);
        var raw_outputs: [8]?*c.OrtValue = @splat(null);
        std.debug.assert(inputs.len <= raw_inputs.len);
        std.debug.assert(outputs.len <= raw_outputs.len);
        for (inputs, 0..) |value, i| raw_inputs[i] = value.ptr;

        try check(api.Run.?(
            self.ptr,
            null,
            @ptrCast(input_names.ptr),
            &raw_inputs,
            inputs.len,
            @ptrCast(output_names.ptr),
            output_names.len,
            &raw_outputs,
        ));

        for (outputs, raw_outputs[0..outputs.len]) |*out, raw| {
            out.* = .{ .ptr = raw.? };
        }
    }
};

/// A tensor crossing the C boundary. Values built by `borrow*` do not own their
/// data -- the runtime reads straight out of the caller's slice -- while those
/// returned by `Session.run` own theirs.
pub const Value = struct {
    ptr: *c.OrtValue,

    pub fn deinit(self: Value) void {
        api.ReleaseValue.?(self.ptr);
    }

    /// Wraps a caller-owned `f32` slice. `data` must outlive the value.
    pub fn borrowF32(data: []const f32, dims: []const i64) Error!Value {
        return borrow(std.mem.sliceAsBytes(data), dims, c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }

    /// Wraps a caller-owned `i64` slice, which is what the exported prompt
    /// encoder wants for point labels.
    pub fn borrowI64(data: []const i64, dims: []const i64) Error!Value {
        return borrow(std.mem.sliceAsBytes(data), dims, c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    }

    fn borrow(bytes: []const u8, dims: []const i64, element_type: c.ONNXTensorElementDataType) Error!Value {
        var ptr: ?*c.OrtValue = null;
        try check(api.CreateTensorWithDataAsOrtValue.?(
            cpu_memory,
            @constCast(bytes.ptr),
            bytes.len,
            dims.ptr,
            dims.len,
            element_type,
            &ptr,
        ));
        return .{ .ptr = ptr.? };
    }

    /// The tensor's elements, as a view into memory the value owns.
    pub fn dataF32(self: Value) Error![]const f32 {
        var raw: ?*anyopaque = null;
        try check(api.GetTensorMutableData.?(self.ptr, &raw));
        const count = try self.elementCount();
        return @as([*]const f32, @ptrCast(@alignCast(raw.?)))[0..count];
    }

    /// The tensor's shape, written into `buf` and returned as a slice of it.
    pub fn shape(self: Value, buf: []i64) Error![]const i64 {
        var info: ?*c.OrtTensorTypeAndShapeInfo = null;
        try check(api.GetTensorTypeAndShape.?(self.ptr, &info));
        defer api.ReleaseTensorTypeAndShapeInfo.?(info);

        var rank: usize = 0;
        try check(api.GetDimensionsCount.?(info, &rank));
        std.debug.assert(rank <= buf.len);
        try check(api.GetDimensions.?(info, buf.ptr, rank));
        return buf[0..rank];
    }

    fn elementCount(self: Value) Error!usize {
        var info: ?*c.OrtTensorTypeAndShapeInfo = null;
        try check(api.GetTensorTypeAndShape.?(self.ptr, &info));
        defer api.ReleaseTensorTypeAndShapeInfo.?(info);

        var count: usize = 0;
        try check(api.GetTensorShapeElementCount.?(info, &count));
        return count;
    }
};
