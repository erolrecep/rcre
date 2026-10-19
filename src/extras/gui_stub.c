/**
 * @file gui_stub.c
 * @brief Fallback GUI Window Stub for platforms without native desktop window
 */

#include "extras/rcre_window.h"
#include <stddef.h>

int rcre_window_is_supported(void) {
    return 0;
}

RCRE_Window* rcre_window_create(const char* title, size_t width, size_t height) {
    (void)title;
    (void)width;
    (void)height;
    return NULL;
}

int rcre_window_update_rgb(RCRE_Window* win, const uint8_t* rgb_data, size_t width, size_t height) {
    (void)win;
    (void)rgb_data;
    (void)width;
    (void)height;
    return 0;
}

void rcre_window_set_title(RCRE_Window* win, const char* title) {
    (void)win;
    (void)title;
}

void rcre_window_resize(RCRE_Window* win, size_t width, size_t height) {
    (void)win;
    (void)width;
    (void)height;
}

void rcre_window_destroy(RCRE_Window* win) {
    (void)win;
}
