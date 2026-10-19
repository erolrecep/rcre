/**
 * @file rcre_window.h
 * @brief RCRE Lightweight Native GUI Window Companion
 *
 * Provides a minimal, zero-external-dependency hardware-accelerated GUI window
 * to preview live video tensors and computer vision filter outputs.
 * Native macOS Cocoa with cross-platform fallback.
 */

#ifndef RCRE_WINDOW_H
#define RCRE_WINDOW_H

#include "rcre.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RCRE_Window RCRE_Window;

/**
 * @brief Checks if native desktop GUI display is supported in current build.
 * @return 1 if supported, 0 otherwise.
 */
int rcre_window_is_supported(void);

/**
 * @brief Creates a native GUI window for live video display.
 * @param title Window title text.
 * @param width Window content width in pixels.
 * @param height Window content height in pixels.
 * @return Pointer to RCRE_Window instance, or NULL if unsupported/failed.
 */
RCRE_Window* rcre_window_create(const char* title, size_t width, size_t height);

/**
 * @brief Updates window with an RGB24 frame buffer (width * height * 3 bytes).
 * Pumps UI events (mouse, keyboard, window close).
 * @param win Window instance.
 * @param rgb_data Pointer to RGB24 packed pixel buffer.
 * @param width Frame width.
 * @param height Frame height.
 * @return 1 if active; 0 if user requested close (ESC, 'q', or window closed); 2 if view toggle requested (Spacebar/Tab).
 */
int rcre_window_update_rgb(RCRE_Window* win, const uint8_t* rgb_data, size_t width, size_t height);

/**
 * @brief Updates the window title bar (e.g. for dynamic FPS / telemetry display).
 */
void rcre_window_set_title(RCRE_Window* win, const char* title);

/**
 * @brief Resizes window frame to adapt to new content dimensions.
 */
void rcre_window_resize(RCRE_Window* win, size_t width, size_t height);

/**
 * @brief Closes and destroys the window.
 */
void rcre_window_destroy(RCRE_Window* win);

#ifdef __cplusplus
}
#endif

#endif /* RCRE_WINDOW_H */
