#import <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#import "bridge.h"

@interface SamCanvasView : NSView
@property (nonatomic, assign) CGImageRef currentImage;
@property (nonatomic, assign) int imgWidth;
@property (nonatomic, assign) int imgHeight;
@property (nonatomic, assign) NSRect imageRect;
@property (nonatomic, assign) BOOL hasImage;
@property (nonatomic, assign) BOOL isBusy;
@property (nonatomic, assign) BOOL clickModeAdd;
@property (nonatomic, assign) const SamCallbacks *callbacks;

- (void)setImage:(CGImageRef)image width:(int)width height:(int)height;
@end

@implementation SamCanvasView

- (instancetype)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        _clickModeAdd = YES;
        self.wantsLayer = YES;
        self.layer.backgroundColor = [NSColor colorWithCalibratedRed:0.11 green:0.12 blue:0.15 alpha:1.0].CGColor;
        self.layer.borderColor = [NSColor colorWithCalibratedRed:0.17 green:0.19 blue:0.22 alpha:1.0].CGColor;
        self.layer.borderWidth = 1.0;
        self.layer.cornerRadius = 8.0;
        self.layer.masksToBounds = YES;
        [self registerForDraggedTypes:@[NSPasteboardTypeFileURL]];
    }
    return self;
}

- (void)dealloc {
    if (_currentImage) {
        CGImageRelease(_currentImage);
        _currentImage = NULL;
    }
}

- (BOOL)isFlipped {
    return YES;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)setImage:(CGImageRef)image width:(int)width height:(int)height {
    if (_currentImage) {
        CGImageRelease(_currentImage);
        _currentImage = NULL;
    }
    if (image && width > 0 && height > 0) {
        _currentImage = CGImageRetain(image);
        _imgWidth = width;
        _imgHeight = height;
        _hasImage = YES;
    } else {
        _imgWidth = 0;
        _imgHeight = 0;
        _hasImage = NO;
    }
    [self setNeedsDisplay:YES];
    if (self.window) {
        [self.window invalidateCursorRectsForView:self];
    }
}

- (void)resetCursorRects {
    [super resetCursorRects];
    if (_hasImage && _imageRect.size.width > 0 && _imageRect.size.height > 0) {
        [self addCursorRect:_imageRect cursor:[NSCursor crosshairCursor]];
    }
}

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];

    NSRect bounds = self.bounds;

    NSColor *bgColor = [NSColor colorWithCalibratedRed:0.11 green:0.12 blue:0.15 alpha:1.0];
    [bgColor setFill];
    NSRectFill(bounds);

    if (_hasImage && _currentImage && _imgWidth > 0 && _imgHeight > 0) {
        CGFloat scaleX = bounds.size.width / (CGFloat)_imgWidth;
        CGFloat scaleY = bounds.size.height / (CGFloat)_imgHeight;
        CGFloat scale = fmin(scaleX, scaleY);

        CGFloat drawW = _imgWidth * scale;
        CGFloat drawH = _imgHeight * scale;
        CGFloat drawX = bounds.origin.x + (bounds.size.width - drawW) * 0.5;
        CGFloat drawY = bounds.origin.y + (bounds.size.height - drawH) * 0.5;
        _imageRect = NSMakeRect(drawX, drawY, drawW, drawH);

        CGContextRef ctx = [[NSGraphicsContext currentContext] CGContext];
        CGContextSaveGState(ctx);
        CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
        CGContextTranslateCTM(ctx, _imageRect.origin.x, _imageRect.origin.y + _imageRect.size.height);
        CGContextScaleCTM(ctx, 1.0, -1.0);
        CGContextDrawImage(ctx, CGRectMake(0, 0, _imageRect.size.width, _imageRect.size.height), _currentImage);
        CGContextRestoreGState(ctx);
    } else {
        _imageRect = NSZeroRect;
        NSString *placeholder = @"Click “Open Image…” or “Sample Image”, or drag an image file here.";
        NSDictionary *attrs = @{
            NSFontAttributeName: [NSFont systemFontOfSize:14],
            NSForegroundColorAttributeName: [NSColor colorWithCalibratedRed:0.59 green:0.61 blue:0.65 alpha:1.0]
        };
        NSSize textSize = [placeholder sizeWithAttributes:attrs];
        NSRect textRect = NSMakeRect(
            bounds.origin.x + (bounds.size.width - textSize.width) * 0.5,
            bounds.origin.y + (bounds.size.height - textSize.height) * 0.5,
            textSize.width,
            textSize.height
        );
        [placeholder drawInRect:textRect withAttributes:attrs];
    }
}

- (void)mouseDown:(NSEvent *)event {
    if (_isBusy || !_hasImage) return;
    NSPoint p = [self convertPoint:[event locationInWindow] fromView:nil];
    if (!NSPointInRect(p, _imageRect)) return;

    float normX = (float)((p.x - _imageRect.origin.x) / _imageRect.size.width);
    float normY = (float)((p.y - _imageRect.origin.y) / _imageRect.size.height);
    if (normX < 0.0f) normX = 0.0f; else if (normX > 1.0f) normX = 1.0f;
    if (normY < 0.0f) normY = 0.0f; else if (normY > 1.0f) normY = 1.0f;

    BOOL shift = (event.modifierFlags & NSEventModifierFlagShift) != 0;
    int isPositive = (_clickModeAdd && !shift) ? 1 : 0;

    if (_callbacks && _callbacks->on_canvas_click) {
        _callbacks->on_canvas_click(normX, normY, isPositive);
    }
}

- (void)rightMouseDown:(NSEvent *)event {
    if (_isBusy || !_hasImage) return;
    NSPoint p = [self convertPoint:[event locationInWindow] fromView:nil];
    if (!NSPointInRect(p, _imageRect)) return;

    float normX = (float)((p.x - _imageRect.origin.x) / _imageRect.size.width);
    float normY = (float)((p.y - _imageRect.origin.y) / _imageRect.size.height);
    if (normX < 0.0f) normX = 0.0f; else if (normX > 1.0f) normX = 1.0f;
    if (normY < 0.0f) normY = 0.0f; else if (normY > 1.0f) normY = 1.0f;

    if (_callbacks && _callbacks->on_canvas_click) {
        _callbacks->on_canvas_click(normX, normY, 0); // Always negative on right-click
    }
}

- (NSDragOperation)draggingEntered:(id<NSDraggingInfo>)sender {
    NSPasteboard *pboard = [sender draggingPasteboard];
    if ([[pboard types] containsObject:NSPasteboardTypeFileURL]) {
        return NSDragOperationCopy;
    }
    return NSDragOperationNone;
}

- (BOOL)performDragOperation:(id<NSDraggingInfo>)sender {
    NSPasteboard *pboard = [sender draggingPasteboard];
    if ([[pboard types] containsObject:NSPasteboardTypeFileURL]) {
        NSURL *fileURL = [NSURL URLFromPasteboard:pboard];
        if (fileURL && _callbacks && _callbacks->on_open_file) {
            _callbacks->on_open_file([fileURL.path UTF8String]);
            return YES;
        }
    }
    return NO;
}

@end

@interface SamAppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property (nonatomic, strong) NSWindow *window;
@property (nonatomic, strong) SamCanvasView *canvasView;
@property (nonatomic, strong) NSTextField *statusLabel;
@property (nonatomic, strong) NSTextField *conceptField;
@property (nonatomic, strong) NSButton *openBtn;
@property (nonatomic, strong) NSButton *sampleBtn;
@property (nonatomic, strong) NSSegmentedControl *modeSeg;
@property (nonatomic, strong) NSButton *clearBtn;
@property (nonatomic, strong) NSButton *findBtn;
@property (nonatomic, strong) NSProgressIndicator *spinner;
@property (nonatomic, strong) NSStackView *masksStackView;
@property (nonatomic, strong) NSScrollView *masksScrollView;
@property (nonatomic, strong) NSMutableArray<NSButton *> *maskButtons;
@property (nonatomic, assign) const SamCallbacks *callbacks;

- (instancetype)initWithCallbacks:(const SamCallbacks *)callbacks;
- (void)createWindow;
- (void)updateMasks:(NSArray<NSDictionary *> *)masks bestIndex:(int)bestIndex selectedIndex:(int)selectedIndex;
- (void)setBusy:(BOOL)busy;
@end

static SamAppDelegate *g_delegate = nil;

@implementation SamAppDelegate

- (instancetype)initWithCallbacks:(const SamCallbacks *)callbacks {
    self = [super init];
    if (self) {
        _callbacks = callbacks;
        _maskButtons = [NSMutableArray array];
    }
    return self;
}

- (void)createWindow {
    NSRect screenRect = [[NSScreen mainScreen] visibleFrame];
    CGFloat winWidth = fmin(1060.0, screenRect.size.width - 100.0);
    CGFloat winHeight = fmin(800.0, screenRect.size.height - 100.0);
    NSRect winRect = NSMakeRect(0, 0, winWidth, winHeight);

    NSUInteger styleMask = NSWindowStyleMaskTitled |
                           NSWindowStyleMaskClosable |
                           NSWindowStyleMaskMiniaturizable |
                           NSWindowStyleMaskResizable;

    _window = [[NSWindow alloc] initWithContentRect:winRect
                                          styleMask:styleMask
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = @"SAM 3";
    _window.delegate = self;
    _window.minSize = NSMakeSize(780, 520);
    _window.appearance = [NSAppearance appearanceNamed:NSAppearanceNameDarkAqua];
    _window.backgroundColor = [NSColor colorWithCalibratedRed:0.08 green:0.09 blue:0.10 alpha:1.0];

    NSView *contentView = _window.contentView;

    // Row 1
    _openBtn = [NSButton buttonWithTitle:@"Open Image…" target:self action:@selector(openFile:)];
    _sampleBtn = [NSButton buttonWithTitle:@"Sample Image" target:self action:@selector(sampleClick:)];

    _modeSeg = [NSSegmentedControl segmentedControlWithLabels:@[@"Clicks add to mask", @"Clicks cut from mask"]
                                                trackingMode:NSSegmentSwitchTrackingSelectOne
                                                      target:self
                                                      action:@selector(modeChanged:)];
    _modeSeg.selectedSegment = 0;

    _clearBtn = [NSButton buttonWithTitle:@"Clear Points" target:self action:@selector(clearClick:)];

    NSStackView *row1 = [NSStackView stackViewWithViews:@[_openBtn, _sampleBtn, _modeSeg, _clearBtn]];
    row1.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    row1.spacing = 8.0;
    row1.alignment = NSLayoutAttributeCenterY;

    // Row 2
    _conceptField = [[NSTextField alloc] init];
    _conceptField.placeholderString = @"Find objects, e.g. cat or red car";
    _conceptField.target = self;
    _conceptField.action = @selector(findClick:);
    _conceptField.maximumNumberOfLines = 1;
    _conceptField.font = [NSFont systemFontOfSize:13];

    _findBtn = [NSButton buttonWithTitle:@"Find by Word" target:self action:@selector(findClick:)];

    _spinner = [[NSProgressIndicator alloc] init];
    _spinner.style = NSProgressIndicatorStyleSpinning;
    _spinner.controlSize = NSControlSizeSmall;
    _spinner.displayedWhenStopped = NO;

    NSStackView *row2 = [NSStackView stackViewWithViews:@[_conceptField, _findBtn, _spinner]];
    row2.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    row2.spacing = 8.0;
    row2.alignment = NSLayoutAttributeCenterY;

    // Row 3 (Status)
    _statusLabel = [NSTextField labelWithString:@"Initializing SAM 3…"];
    _statusLabel.textColor = [NSColor colorWithCalibratedRed:0.59 green:0.61 blue:0.65 alpha:1.0];
    _statusLabel.font = [NSFont systemFontOfSize:13];

    // Canvas
    _canvasView = [[SamCanvasView alloc] initWithFrame:NSZeroRect];
    _canvasView.callbacks = _callbacks;

    // Masks Bar
    _masksStackView = [NSStackView stackViewWithViews:@[]];
    _masksStackView.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    _masksStackView.spacing = 8.0;
    _masksStackView.alignment = NSLayoutAttributeCenterY;
    _masksStackView.translatesAutoresizingMaskIntoConstraints = NO;

    _masksScrollView = [[NSScrollView alloc] init];
    _masksScrollView.hasHorizontalScroller = YES;
    _masksScrollView.hasVerticalScroller = NO;
    _masksScrollView.drawsBackground = NO;
    _masksScrollView.documentView = _masksStackView;

    // Layout
    for (NSView *v in @[row1, row2, _statusLabel, _canvasView, _masksScrollView]) {
        v.translatesAutoresizingMaskIntoConstraints = NO;
        [contentView addSubview:v];
    }

    [NSLayoutConstraint activateConstraints:@[
        // Row 1
        [row1.topAnchor constraintEqualToAnchor:contentView.topAnchor constant:14.0],
        [row1.leadingAnchor constraintEqualToAnchor:contentView.leadingAnchor constant:16.0],
        [row1.trailingAnchor constraintLessThanOrEqualToAnchor:contentView.trailingAnchor constant:-16.0],

        // Row 2
        [row2.topAnchor constraintEqualToAnchor:row1.bottomAnchor constant:10.0],
        [row2.leadingAnchor constraintEqualToAnchor:contentView.leadingAnchor constant:16.0],
        [row2.trailingAnchor constraintLessThanOrEqualToAnchor:contentView.trailingAnchor constant:-16.0],
        [_conceptField.widthAnchor constraintGreaterThanOrEqualToConstant:280.0],

        // Status
        [_statusLabel.topAnchor constraintEqualToAnchor:row2.bottomAnchor constant:8.0],
        [_statusLabel.leadingAnchor constraintEqualToAnchor:contentView.leadingAnchor constant:16.0],
        [_statusLabel.trailingAnchor constraintEqualToAnchor:contentView.trailingAnchor constant:-16.0],
        [_statusLabel.heightAnchor constraintEqualToConstant:20.0],

        // Canvas
        [_canvasView.topAnchor constraintEqualToAnchor:_statusLabel.bottomAnchor constant:10.0],
        [_canvasView.leadingAnchor constraintEqualToAnchor:contentView.leadingAnchor constant:16.0],
        [_canvasView.trailingAnchor constraintEqualToAnchor:contentView.trailingAnchor constant:-16.0],
        [_canvasView.bottomAnchor constraintEqualToAnchor:_masksScrollView.topAnchor constant:-10.0],

        // Masks Scroll View
        [_masksScrollView.leadingAnchor constraintEqualToAnchor:contentView.leadingAnchor constant:16.0],
        [_masksScrollView.trailingAnchor constraintEqualToAnchor:contentView.trailingAnchor constant:-16.0],
        [_masksScrollView.bottomAnchor constraintEqualToAnchor:contentView.bottomAnchor constant:-12.0],
        [_masksScrollView.heightAnchor constraintEqualToConstant:46.0],

        // Inner stack view in scroll view
        [_masksStackView.topAnchor constraintEqualToAnchor:_masksScrollView.contentView.topAnchor],
        [_masksStackView.bottomAnchor constraintEqualToAnchor:_masksScrollView.contentView.bottomAnchor],
        [_masksStackView.leadingAnchor constraintEqualToAnchor:_masksScrollView.contentView.leadingAnchor],
        [_masksStackView.heightAnchor constraintEqualToAnchor:_masksScrollView.contentView.heightAnchor]
    ]];

    [_window center];
    [_window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)openFile:(id)sender {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = YES;
    panel.canChooseDirectories = NO;
    panel.allowsMultipleSelection = NO;
    if (@available(macOS 11.0, *)) {
        panel.allowedContentTypes = @[UTTypeImage];
    } else {
        panel.allowedFileTypes = @[@"png", @"jpg", @"jpeg", @"webp", @"bmp", @"tiff", @"gif"];
    }

    if ([panel runModal] == NSModalResponseOK) {
        NSURL *url = panel.URLs.firstObject;
        if (url && _callbacks && _callbacks->on_open_file) {
            _callbacks->on_open_file([url.path UTF8String]);
        }
    }
}

- (void)sampleClick:(id)sender {
    if (_callbacks && _callbacks->on_sample_click) {
        _callbacks->on_sample_click();
    }
}

- (void)modeChanged:(id)sender {
    int mode = (_modeSeg.selectedSegment == 0) ? 1 : 0;
    _canvasView.clickModeAdd = (mode == 1);
    if (_callbacks && _callbacks->on_mode_change) {
        _callbacks->on_mode_change(mode);
    }
}

- (void)clearClick:(id)sender {
    if (_callbacks && _callbacks->on_clear_points) {
        _callbacks->on_clear_points();
    }
}

- (void)findClick:(id)sender {
    NSString *text = [_conceptField.stringValue stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if (text.length > 0 && _callbacks && _callbacks->on_find_text) {
        _callbacks->on_find_text([text UTF8String]);
    }
}

- (void)maskClicked:(NSButton *)sender {
    int index = (int)sender.tag;
    if (_callbacks && _callbacks->on_select_mask) {
        _callbacks->on_select_mask(index);
    }
}

- (void)updateMasks:(NSArray<NSDictionary *> *)masks bestIndex:(int)bestIndex selectedIndex:(int)selectedIndex {
    for (NSButton *btn in _maskButtons) {
        [btn removeFromSuperview];
    }
    [_maskButtons removeAllObjects];

    for (int i = 0; i < (int)masks.count; i++) {
        float score = [masks[i][@"score"] floatValue];
        float coverage = [masks[i][@"coverage"] floatValue];
        NSString *star = (i == bestIndex) ? @" ★" : @"";
        NSString *title = [NSString stringWithFormat:@"Mask %d%@ (%.3f · %.1f%%)",
                           i, star, score, coverage * 100.0f];

        NSButton *btn = [NSButton buttonWithTitle:title target:self action:@selector(maskClicked:)];
        btn.tag = i;
        btn.bezelStyle = NSBezelStyleRounded;
        if (i == selectedIndex) {
            btn.contentTintColor = [NSColor colorWithCalibratedRed:0.0 green:0.86 blue:0.39 alpha:1.0];
        } else {
            btn.contentTintColor = [NSColor colorWithCalibratedRed:0.90 green:0.91 blue:0.93 alpha:1.0];
        }

        [_masksStackView addArrangedSubview:btn];
        [_maskButtons addObject:btn];
    }
}

- (void)setBusy:(BOOL)busy {
    _canvasView.isBusy = busy;
    _openBtn.enabled = !busy;
    _sampleBtn.enabled = !busy;
    _clearBtn.enabled = !busy;
    _findBtn.enabled = !busy;
    _conceptField.enabled = !busy;

    if (busy) {
        [_spinner startAnimation:nil];
    } else {
        [_spinner stopAnimation:nil];
    }
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

- (void)windowDidResize:(NSNotification *)notification {
    [_canvasView setNeedsDisplay:YES];
    [_window invalidateCursorRectsForView:_canvasView];
}

@end

int sam_macos_init(const SamCallbacks *callbacks) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        g_delegate = [[SamAppDelegate alloc] initWithCallbacks:callbacks];
        [NSApp setDelegate:g_delegate];

        NSMenu *menubar = [[NSMenu alloc] init];

        // App Menu
        NSMenuItem *appMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:appMenuItem];
        NSMenu *appMenu = [[NSMenu alloc] init];
        [appMenu addItemWithTitle:@"About SAM 3" action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
        [appMenu addItem:[NSMenuItem separatorItem]];
        [appMenu addItemWithTitle:@"Hide SAM 3" action:@selector(hide:) keyEquivalent:@"h"];
        [appMenu addItemWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
        [appMenu addItemWithTitle:@"Show All" action:@selector(unhideAllApplications:) keyEquivalent:@""];
        [appMenu addItem:[NSMenuItem separatorItem]];
        [appMenu addItemWithTitle:@"Quit SAM 3" action:@selector(terminate:) keyEquivalent:@"q"];
        [appMenuItem setSubmenu:appMenu];

        // File Menu
        NSMenuItem *fileMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:fileMenuItem];
        NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
        [fileMenu addItemWithTitle:@"Open Image…" action:@selector(openFile:) keyEquivalent:@"o"];
        [fileMenu addItem:[NSMenuItem separatorItem]];
        [fileMenu addItemWithTitle:@"Close Window" action:@selector(performClose:) keyEquivalent:@"w"];
        [fileMenuItem setSubmenu:fileMenu];

        // Edit Menu
        NSMenuItem *editMenuItem = [[NSMenuItem alloc] init];
        [menubar addItem:editMenuItem];
        NSMenu *editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
        [editMenu addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
        [editMenu addItemWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"Z"];
        [editMenu addItem:[NSMenuItem separatorItem]];
        [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
        [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
        [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
        [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
        [editMenuItem setSubmenu:editMenu];

        [NSApp setMainMenu:menubar];

        [g_delegate createWindow];
    }
    return 0;
}

void sam_macos_run(void) {
    @autoreleasepool {
        [NSApp run];
    }
}

void sam_macos_set_status(const char *text) {
    NSString *str = text ? [NSString stringWithUTF8String:text] : @"";
    dispatch_async(dispatch_get_main_queue(), ^{
        if (g_delegate && g_delegate.statusLabel) {
            [g_delegate.statusLabel setStringValue:str];
        }
    });
}

void sam_macos_set_image(const uint8_t *rgba_pixels, int width, int height) {
    if (!rgba_pixels || width <= 0 || height <= 0) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (g_delegate && g_delegate.canvasView) {
                [g_delegate.canvasView setImage:NULL width:0 height:0];
            }
        });
        return;
    }

    NSData *data = [NSData dataWithBytes:rgba_pixels length:(NSUInteger)(width * height * 4)];
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGImageRef cgImage = CGImageCreate(
        width,
        height,
        8,
        32,
        width * 4,
        colorSpace,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big,
        provider,
        NULL,
        NO,
        kCGRenderingIntentDefault
    );
    CGColorSpaceRelease(colorSpace);
    CGDataProviderRelease(provider);

    dispatch_async(dispatch_get_main_queue(), ^{
        if (g_delegate && g_delegate.canvasView) {
            [g_delegate.canvasView setImage:cgImage width:width height:height];
        }
        CGImageRelease(cgImage);
    });
}

void sam_macos_set_masks(int count, const SamMaskInfo *masks, int best_index, int selected_index) {
    NSMutableArray *list = [NSMutableArray arrayWithCapacity:count];
    for (int i = 0; i < count; i++) {
        [list addObject:@{
            @"score": @(masks[i].score),
            @"coverage": @(masks[i].coverage)
        }];
    }
    dispatch_async(dispatch_get_main_queue(), ^{
        if (g_delegate) {
            [g_delegate updateMasks:list bestIndex:best_index selectedIndex:selected_index];
        }
    });
}

void sam_macos_set_busy(int is_busy) {
    dispatch_async(dispatch_get_main_queue(), ^{
        if (g_delegate) {
            [g_delegate setBusy:(is_busy != 0)];
        }
    });
}

void sam_macos_dispatch_main(void (*fn)(void *ctx), void *ctx) {
    if ([NSThread isMainThread]) {
        fn(ctx);
    } else {
        dispatch_async(dispatch_get_main_queue(), ^{
            fn(ctx);
        });
    }
}
