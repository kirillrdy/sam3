const std = @import("std");

pub const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

var api: *const c.OrtApi = undefined;

var cpu_memory: *c.OrtMemoryInfo = undefined;

pub const Error = error{
    OnnxRuntime,

    UnsupportedApiVersion,
};

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

pub fn init() Error!void {
    const base = c.OrtGetApiBase();
    api = base.*.GetApi.?(c.ORT_API_VERSION) orelse return Error.UnsupportedApiVersion;

    var info: ?*c.OrtMemoryInfo = null;
    try check(api.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &info));
    cpu_memory = info.?;
}

pub fn version() []const u8 {
    return std.mem.span(c.OrtGetApiBase().*.GetVersionString.?());
}

pub const DeviceKind = enum {
    npu,
    gpu,
    webgpu,
    cpu,

    pub fn openvinoName(self: DeviceKind) [:0]const u8 {
        return switch (self) {
            .npu => "NPU",
            .gpu => "GPU",
            .webgpu => unreachable,
            .cpu => "CPU",
        };
    }

    fn hardwareType(self: DeviceKind) c.OrtHardwareDeviceType {
        return switch (self) {
            .npu => c.OrtHardwareDeviceType_NPU,
            .gpu => c.OrtHardwareDeviceType_GPU,
            .webgpu => c.OrtHardwareDeviceType_GPU,
            .cpu => c.OrtHardwareDeviceType_CPU,
        };
    }
};

pub const Device = struct {
    kind: DeviceKind,

    id: u32,
    handle: *const c.OrtEpDevice,
};

pub const Accelerator = union(enum) {
    device: Device,
    coreml: DeviceKind,

    fn kind(self: Accelerator) DeviceKind {
        return switch (self) {
            .device => |device| device.kind,
            .coreml => |device| device,
        };
    }
};

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

    pub fn registerProvider(self: Env, name: [:0]const u8, path: [:0]const u8) Error!void {
        try check(api.RegisterExecutionProviderLibrary.?(self.ptr, name.ptr, path.ptr));
    }

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

pub const Session = struct {
    ptr: *c.OrtSession,

    device: DeviceKind,

    pub fn open(
        env: Env,
        model_path: [:0]const u8,
        accelerator: ?Accelerator,
        options: []const Option,
        on_fallback: ?*const fn (message: []const u8) void,
    ) Error!Session {
        if (accelerator) |selected| {
            if (tryOpen(env, model_path, selected, options)) |ptr| {
                return .{ .ptr = ptr, .device = selected.kind() };
            } else |_| {
                if (on_fallback) |report| report(lastError());
            }
        }
        return .{ .ptr = try tryOpen(env, model_path, null, &.{}), .device = .cpu };
    }

    fn tryOpen(
        env: Env,
        model_path: [:0]const u8,
        accelerator: ?Accelerator,
        options: []const Option,
    ) Error!*c.OrtSession {
        var session_options: ?*c.OrtSessionOptions = null;
        try check(api.CreateSessionOptions.?(&session_options));
        defer api.ReleaseSessionOptions.?(session_options);

        try check(api.AddFreeDimensionOverrideByName.?(session_options, "batch_size", 1));

        if (accelerator) |selected| {
            if (selected.kind() == .webgpu) {
                // SAM 3 uses static tensor shapes. Exact-size reuse avoids the
                // substantial VRAM overhead of WebGPU's bucketed default.
                try check(api.AddSessionConfigEntry.?(
                    session_options,
                    "ep.webgpuexecutionprovider.storageBufferCacheMode",
                    "simple",
                ));
            }

            var keys: [8][*:0]const u8 = undefined;
            var values: [8][*:0]const u8 = undefined;
            std.debug.assert(options.len <= keys.len);
            for (options, 0..) |option, i| {
                keys[i] = option.key;
                values[i] = option.value;
            }

            switch (selected) {
                .device => |d| try check(api.SessionOptionsAppendExecutionProvider_V2.?(
                    session_options,
                    env.ptr,
                    @ptrCast(&d.handle),
                    1,
                    @ptrCast(&keys),
                    @ptrCast(&values),
                    options.len,
                )),
                .coreml => try check(api.SessionOptionsAppendExecutionProvider.?(
                    session_options,
                    "CoreML",
                    @ptrCast(&keys),
                    @ptrCast(&values),
                    options.len,
                )),
            }
        }

        var ptr: ?*c.OrtSession = null;
        try check(api.CreateSession.?(env.ptr, model_path.ptr, session_options, &ptr));
        return ptr.?;
    }

    pub fn deinit(self: Session) void {
        api.ReleaseSession.?(self.ptr);
    }

    pub fn run(
        self: Session,
        input_names: []const [*:0]const u8,
        inputs: []const Value,
        output_names: []const [*:0]const u8,
        outputs: []Value,
    ) Error!void {
        std.debug.assert(input_names.len == inputs.len);
        std.debug.assert(output_names.len == outputs.len);

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

pub const Value = struct {
    ptr: *c.OrtValue,

    pub fn deinit(self: Value) void {
        api.ReleaseValue.?(self.ptr);
    }

    pub fn borrowF32(data: []const f32, dims: []const i64) Error!Value {
        return borrow(std.mem.sliceAsBytes(data), dims, c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    }

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

    pub fn dataF32(self: Value) Error![]const f32 {
        var raw: ?*anyopaque = null;
        try check(api.GetTensorMutableData.?(self.ptr, &raw));
        const count = try self.elementCount();
        return @as([*]const f32, @ptrCast(@alignCast(raw.?)))[0..count];
    }

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
