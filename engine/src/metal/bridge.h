#include <stddef.h>
#include <stdint.h>

typedef struct SamMetalContext SamMetalContext;
typedef struct SamMetalModule SamMetalModule;
typedef struct SamMetalFunction SamMetalFunction;

typedef struct {
    void *buffer;
    size_t offset;
} SamMetalBufferRef;

typedef struct {
    uint32_t x, y, z;
} SamMetalDim;

typedef enum {
    SAM_METAL_BUFFER,
    SAM_METAL_BYTES,
} SamMetalArgKind;

typedef struct {
    SamMetalArgKind kind;
    SamMetalBufferRef buffer;
    const void *bytes;
    size_t size;
} SamMetalArg;

SamMetalContext *sam_metal_context_create(uint32_t ordinal);
void sam_metal_context_destroy(SamMetalContext *context);
int sam_metal_context_synchronize(SamMetalContext *context);

SamMetalModule *sam_metal_module_create(SamMetalContext *context, const char *source);
void sam_metal_module_destroy(SamMetalModule *module);
SamMetalFunction *sam_metal_function_create(SamMetalModule *module, const char *name);
void sam_metal_function_destroy(SamMetalFunction *function);
int sam_metal_launch(SamMetalFunction *function, SamMetalDim grid, SamMetalDim block,
                     const SamMetalArg *args, size_t arg_count);

void *sam_metal_buffer_create(SamMetalContext *context, size_t bytes);
void sam_metal_buffer_destroy(void *buffer);
void sam_metal_buffer_upload(SamMetalBufferRef buffer, const void *source, size_t bytes);
int sam_metal_buffer_upload_async(SamMetalContext *context, SamMetalBufferRef buffer,
                                  const void *source, size_t bytes);
void sam_metal_buffer_download(SamMetalBufferRef buffer, void *destination, size_t bytes);

const char *sam_metal_last_error(void);
