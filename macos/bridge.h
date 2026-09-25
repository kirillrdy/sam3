#ifndef SAM_MACOS_BRIDGE_H
#define SAM_MACOS_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void (*on_open_file)(const char *path);
    void (*on_sample_click)(void);
    void (*on_mode_change)(int mode); // 1 = add, 0 = cut
    void (*on_clear_points)(void);
    void (*on_find_text)(const char *text);
    void (*on_canvas_click)(float norm_x, float norm_y, int is_positive);
    void (*on_select_mask)(int mask_index);
} SamCallbacks;

typedef struct {
    float score;
    float coverage;
} SamMaskInfo;

int sam_macos_init(const SamCallbacks *callbacks);
void sam_macos_run(void);
void sam_macos_set_status(const char *text);
void sam_macos_set_image(const uint8_t *rgba_pixels, int width, int height);
void sam_macos_set_masks(int count, const SamMaskInfo *masks, int best_index, int selected_index);
void sam_macos_set_busy(int is_busy);
void sam_macos_dispatch_main(void (*fn)(void *ctx), void *ctx);

#ifdef __cplusplus
}
#endif

#endif // SAM_MACOS_BRIDGE_H
