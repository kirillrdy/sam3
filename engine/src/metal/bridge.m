#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "bridge.h"

struct SamMetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> last;
};

struct SamMetalModule {
    SamMetalContext *context;
    id<MTLLibrary> library;
};

struct SamMetalFunction {
    SamMetalContext *context;
    id<MTLComputePipelineState> pipeline;
};

static __thread char error_text[8192];

static void set_error(NSString *text) {
    const char *utf8 = text ? text.UTF8String : "unknown Metal error";
    snprintf(error_text, sizeof(error_text), "%s", utf8);
}

const char *sam_metal_last_error(void) { return error_text; }

SamMetalContext *sam_metal_context_create(uint32_t ordinal) {
    @autoreleasepool {
        id<MTLDevice> device = ordinal == 0 ? MTLCreateSystemDefaultDevice() : nil;
        if (!device) {
            set_error(@"requested Metal device ordinal not found");
            return NULL;
        }
        SamMetalContext *context = calloc(1, sizeof(*context));
        context->device = device;
        context->queue = [context->device newCommandQueue];
        if (!context->queue) {
            set_error(@"could not create a Metal command queue");
            free(context);
            return NULL;
        }
        return context;
    }
}

void sam_metal_context_destroy(SamMetalContext *context) { free(context); }

int sam_metal_context_synchronize(SamMetalContext *context) {
    id<MTLCommandBuffer> command = context->last;
    if (!command) return 1;
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        set_error(command.error.localizedDescription);
        return 0;
    }
    context->last = nil;
    return 1;
}

SamMetalModule *sam_metal_module_create(SamMetalContext *context, const char *source) {
    @autoreleasepool {
        NSError *error = nil;
        NSString *text = [NSString stringWithUTF8String:source];
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = YES;
        id<MTLLibrary> library = [context->device newLibraryWithSource:text options:options error:&error];
        if (!library) {
            set_error(error.localizedDescription);
            return NULL;
        }
        SamMetalModule *module = calloc(1, sizeof(*module));
        module->context = context;
        module->library = library;
        return module;
    }
}

void sam_metal_module_destroy(SamMetalModule *module) { free(module); }

SamMetalFunction *sam_metal_function_create(SamMetalModule *module, const char *name) {
    @autoreleasepool {
        NSString *symbol = [NSString stringWithUTF8String:name];
        id<MTLFunction> function = [module->library newFunctionWithName:symbol];
        if (!function) {
            set_error([NSString stringWithFormat:@"Metal function '%@' was not found", symbol]);
            return NULL;
        }
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline =
            [module->context->device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            set_error(error.localizedDescription);
            return NULL;
        }
        SamMetalFunction *result = calloc(1, sizeof(*result));
        result->context = module->context;
        result->pipeline = pipeline;
        return result;
    }
}

void sam_metal_function_destroy(SamMetalFunction *function) { free(function); }

int sam_metal_launch(SamMetalFunction *function, SamMetalDim grid, SamMetalDim block,
                     const SamMetalArg *args, size_t arg_count) {
    @autoreleasepool {
        SamMetalContext *context = function->context;
        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:function->pipeline];
        for (size_t i = 0; i < arg_count; ++i) {
            if (args[i].kind == SAM_METAL_BUFFER) {
                [encoder setBuffer:(__bridge id<MTLBuffer>)args[i].buffer.buffer
                            offset:args[i].buffer.offset atIndex:i];
            } else {
                [encoder setBytes:args[i].bytes length:args[i].size atIndex:i];
            }
        }
        [encoder dispatchThreadgroups:MTLSizeMake(grid.x, grid.y, grid.z)
                threadsPerThreadgroup:MTLSizeMake(block.x, block.y, block.z)];
        [encoder endEncoding];
        [command commit];
        context->last = command;
        return 1;
    }
}

void *sam_metal_buffer_create(SamMetalContext *context, size_t bytes) {
    id<MTLBuffer> buffer = [context->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) set_error(@"could not allocate a Metal buffer");
    return (__bridge_retained void *)buffer;
}

void sam_metal_buffer_destroy(void *buffer) {
    if (buffer) CFRelease(buffer);
}

void sam_metal_buffer_upload(SamMetalBufferRef buffer, const void *source, size_t bytes) {
    id<MTLBuffer> object = (__bridge id<MTLBuffer>)buffer.buffer;
    memcpy((uint8_t *)object.contents + buffer.offset, source, bytes);
}

int sam_metal_buffer_upload_async(SamMetalContext *context, SamMetalBufferRef buffer,
                                  const void *source, size_t bytes) {
    @autoreleasepool {
        id<MTLBuffer> staging = [context->device newBufferWithBytes:source
                                                            length:bytes
                                                           options:MTLResourceStorageModeShared];
        if (!staging) {
            set_error(@"could not allocate a Metal upload buffer");
            return 0;
        }
        id<MTLCommandBuffer> command = [context->queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        [encoder copyFromBuffer:staging sourceOffset:0
                       toBuffer:(__bridge id<MTLBuffer>)buffer.buffer destinationOffset:buffer.offset
                           size:bytes];
        [encoder endEncoding];
        [command commit];
        context->last = command;
        return 1;
    }
}

void sam_metal_buffer_download(SamMetalBufferRef buffer, void *destination, size_t bytes) {
    id<MTLBuffer> object = (__bridge id<MTLBuffer>)buffer.buffer;
    memcpy(destination, (uint8_t *)object.contents + buffer.offset, bytes);
}
