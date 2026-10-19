/**
 * @file gui_mac.m
 * @brief macOS Cocoa Native Video Window Implementation
 */

#import <Cocoa/Cocoa.h>
#include "extras/rcre_window.h"
#include <stdlib.h>

@interface RCREWindowDelegate : NSObject <NSWindowDelegate, NSApplicationDelegate>
@property (nonatomic, assign) BOOL shouldClose;
@end

@implementation RCREWindowDelegate
- (BOOL)windowShouldClose:(NSWindow *)sender {
    (void)sender;
    self.shouldClose = YES;
    return YES;
}
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    (void)sender;
    return NO;
}
@end

struct RCRE_Window {
    NSWindow *window;
    NSImageView *imageView;
    RCREWindowDelegate *delegate;
    size_t width;
    size_t height;
    BOOL shouldClose;
};

int rcre_window_is_supported(void) {
    return 1;
}

RCRE_Window* rcre_window_create(const char* title, size_t width, size_t height) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        // Center on screen
        NSScreen *screen = [NSScreen mainScreen];
        NSRect screenRect = screen ? [screen visibleFrame] : NSMakeRect(0, 0, 1920, 1080);
        CGFloat posX = (screenRect.size.width - (CGFloat)width) / 2.0;
        CGFloat posY = (screenRect.size.height - (CGFloat)height) / 2.0;
        if (posX < 0) posX = 50;
        if (posY < 0) posY = 50;

        NSRect frame = NSMakeRect(posX, posY, (CGFloat)width, (CGFloat)height);
        NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                             styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
                                               backing:NSBackingStoreBuffered
                                                 defer:NO];
        if (!window) return NULL;

        NSString *nsTitle = [NSString stringWithUTF8String:title ? title : "RCRE Video Stream"];
        [window setTitle:nsTitle];

        RCREWindowDelegate *delegate = [[RCREWindowDelegate alloc] init];
        [window setDelegate:delegate];
        [NSApp setDelegate:delegate];

        NSImageView *imageView = [[NSImageView alloc] initWithFrame:[[window contentView] bounds]];
        [imageView setImageScaling:NSImageScaleProportionallyUpOrDown];
        [imageView setAutoresizingMask:NSViewWidthSizable | NSViewHeightSizable];
        [[window contentView] addSubview:imageView];

        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        RCRE_Window* win = (RCRE_Window*)malloc(sizeof(RCRE_Window));
        if (!win) return NULL;

        win->window = window;
        win->imageView = imageView;
        win->delegate = delegate;
        win->width = width;
        win->height = height;
        win->shouldClose = NO;

        return win;
    }
}

int rcre_window_update_rgb(RCRE_Window* win, const uint8_t* rgb_data, size_t width, size_t height) {
    if (!win || win->shouldClose) return 0;

    @autoreleasepool {
        if (win->delegate.shouldClose) {
            win->shouldClose = YES;
            return 0;
        }

        // Pump events
        NSEvent *event;
        while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                          untilDate:[NSDate distantPast]
                                             inMode:NSDefaultRunLoopMode
                                            dequeue:YES])) {
            if ([event type] == NSEventTypeKeyDown) {
                NSString *chars = [event charactersIgnoringModifiers];
                if ([chars length] > 0) {
                    unichar c = [chars characterAtIndex:0];
                    if (c == 27 /* ESC */ || c == 'q' || c == 'Q') {
                        win->shouldClose = YES;
                        return 0;
                    }
                }
            }
            [NSApp sendEvent:event];
        }

        if (win->delegate.shouldClose) {
            win->shouldClose = YES;
            return 0;
        }

        if (!rgb_data) return 1;

        // Render RGB24
        unsigned char *planes[1] = { (unsigned char *)rgb_data };
        NSBitmapImageRep *rep = [[NSBitmapImageRep alloc]
            initWithBitmapDataPlanes:planes
                          pixelsWide:(NSInteger)width
                          pixelsHigh:(NSInteger)height
                       bitsPerSample:8
                     samplesPerPixel:3
                            hasAlpha:NO
                            isPlanar:NO
                      colorSpaceName:NSDeviceRGBColorSpace
                         bytesPerRow:(NSInteger)(width * 3)
                        bitsPerPixel:24];

        if (rep) {
            NSImage *img = [[NSImage alloc] initWithSize:NSMakeSize((CGFloat)width, (CGFloat)height)];
            [img addRepresentation:rep];
            [win->imageView setImage:img];
        }

        return 1;
    }
}

void rcre_window_set_title(RCRE_Window* win, const char* title) {
    if (!win || !win->window || !title) return;
    @autoreleasepool {
        NSString *nsTitle = [NSString stringWithUTF8String:title];
        [win->window setTitle:nsTitle];
    }
}

void rcre_window_destroy(RCRE_Window* win) {
    if (!win) return;
    @autoreleasepool {
        [NSApp setDelegate:nil];
        if (win->window) {
            [win->window setDelegate:nil];
            [win->window orderOut:nil];
            win->window = nil;
        }
        win->imageView = nil;
        win->delegate = nil;
        free(win);
    }
}
