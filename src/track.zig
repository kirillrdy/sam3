//! Following objects through a video with a model that only ever sees one
//! frame.
//!
//! SAM 3 carries a video in a memory bank: a memory encoder folds each frame's
//! mask back into features, and memory attention lets the next frame read
//! them. The exports published for the tracker are the vision encoder and the
//! prompt encoder / mask decoder only -- neither memory graph is among them --
//! so that state has nowhere to live inside the model.
//!
//! It lives here instead. Every frame is segmented on its own and this stitches
//! the results into tracks, in the two ways the prompts allow:
//!
//!   * A phrase names the objects, and the text decoder finds all of them on
//!     every frame. Frames are tied together by matching this frame's
//!     detections to last frame's tracks.
//!
//!   * A click names one object on one frame, and the track carries itself
//!     forward: the mask it left on the previous frame says where to prompt
//!     the next one.
//!
//! What that buys is identity -- an object keeps its number while it is on
//! screen, survives the frames it is missed on, and is forgotten once it is
//! gone. What it does not buy is the memory bank's recall: an object that
//! leaves and comes back is a new object here, and appearance is only ever
//! carried one frame at a time.

const std = @import("std");
const sam3 = @import("sam3.zig");

pub const Config = struct {
    /// How much of a track a detection has to cover to be the same object.
    match_iou: f32 = 0.2,
    /// Frames an object has to be seen on before it is reported. Above one,
    /// a detection that flickers on a single frame never reaches the screen.
    min_hits: u32 = 1,
    /// Frames a track survives unmatched before it is forgotten. Bridges the
    /// object passing behind something.
    max_misses: u32 = 6,
    /// How far a carried mask may change size in one frame and still be taken
    /// for the same object, as a factor either way.
    max_area_change: f32 = 3.0,
    /// How good the decoder has to think its own mask is for that mask to
    /// carry a track. Below this it is guessing, usually because the object it
    /// was prompted for is no longer there, and a few frames of guessing in a
    /// row is what retires the track.
    min_carried_score: f32 = 0.5,
};

/// What one frame says is there. The mask is borrowed for the duration of the
/// update; the tracker copies whatever it decides to keep.
pub const Detection = struct {
    box: sam3.Box,
    score: f32,
    plane: []const f32,
};

pub const Track = struct {
    /// Stable for as long as the object is followed, and never reused.
    id: u32,
    box: sam3.Box,
    score: f32,
    /// The mask from the last frame this track was seen on, which is both what
    /// gets drawn and what prompts the next frame.
    mask: []f32,
    /// Frames this track has been seen on.
    hits: u32 = 1,
    /// Frames since it last was.
    misses: u32 = 0,

    /// True once the track has been seen often enough to be worth reporting,
    /// and while it is not currently missing.
    pub fn isVisible(self: Track, config: Config) bool {
        return self.misses == 0 and self.hits >= config.min_hits;
    }
};

pub const Tracker = struct {
    allocator: std.mem.Allocator,
    config: Config,

    tracks: std.ArrayList(Track) = .empty,
    next_id: u32 = 1,

    /// The mask resolution every track is held at. Set by the first frame; a
    /// frame that disagrees starts the tracker over.
    width: usize = 0,
    height: usize = 0,

    pub fn init(allocator: std.mem.Allocator, config: Config) Tracker {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *Tracker) void {
        self.clear();
        self.tracks.deinit(self.allocator);
        self.* = undefined;
    }

    /// Forgets every track, and the numbers they were given. A new video, or a
    /// new prompt for the same one, starts from one again.
    pub fn reset(self: *Tracker) void {
        self.clear();
        self.next_id = 1;
        self.width = 0;
        self.height = 0;
    }

    fn clear(self: *Tracker) void {
        for (self.tracks.items) |track| self.allocator.free(track.mask);
        self.tracks.clearRetainingCapacity();
    }

    pub fn isEmpty(self: Tracker) bool {
        return self.tracks.items.len == 0;
    }

    /// Takes one frame's detections and folds them into the tracks.
    ///
    /// Every detection is either an object already being followed or a new
    /// one, and every track is either matched this frame or has missed it.
    pub fn update(
        self: *Tracker,
        detections: []const Detection,
        width: usize,
        height: usize,
    ) !void {
        if (width != self.width or height != self.height) {
            self.reset();
            self.width = width;
            self.height = height;
        }
        const stride = width * height;

        var matched_track = try self.allocator.alloc(bool, self.tracks.items.len);
        defer self.allocator.free(matched_track);
        @memset(matched_track, false);

        var matched_detection = try self.allocator.alloc(bool, detections.len);
        defer self.allocator.free(matched_detection);
        @memset(matched_detection, false);

        // Greedy assignment: take the best-agreeing pair still on the table,
        // then the next, until nothing left agrees enough. With the handful of
        // objects a frame holds this lands on the same answer as a full
        // assignment would, for none of the machinery.
        while (true) {
            var best_similarity = self.config.match_iou;
            var best_track: ?usize = null;
            var best_detection: usize = 0;

            for (self.tracks.items, 0..) |track, t| {
                if (matched_track[t]) continue;
                for (detections, 0..) |detection, d| {
                    if (matched_detection[d]) continue;
                    const similarity = agreement(track, detection);
                    if (similarity > best_similarity) {
                        best_similarity = similarity;
                        best_track = t;
                        best_detection = d;
                    }
                }
            }

            const t = best_track orelse break;
            matched_track[t] = true;
            matched_detection[best_detection] = true;

            const detection = detections[best_detection];
            const track = &self.tracks.items[t];
            @memcpy(track.mask, detection.plane[0..stride]);
            track.box = detection.box;
            track.score = detection.score;
            track.hits += 1;
            track.misses = 0;
        }

        for (detections, 0..) |detection, d| {
            if (matched_detection[d]) continue;
            const mask = try self.allocator.dupe(f32, detection.plane[0..stride]);
            errdefer self.allocator.free(mask);
            try self.tracks.append(self.allocator, .{
                .id = self.next_id,
                .box = detection.box,
                .score = detection.score,
                .mask = mask,
            });
            self.next_id += 1;
        }

        // Walk backwards so that dropping a track does not move the ones still
        // to be looked at.
        var i = matched_track.len;
        while (i > 0) {
            i -= 1;
            if (matched_track[i]) continue;
            const track = &self.tracks.items[i];
            track.misses += 1;
            if (track.misses <= self.config.max_misses) continue;
            self.allocator.free(track.mask);
            _ = self.tracks.orderedRemove(i);
        }
    }

    /// Takes what became of each track on this frame, in the order the tracks
    /// are held in: a detection where the track was found again, and null
    /// where it was not.
    ///
    /// Carrying is not matching. A detection produced by prompting with one
    /// track belongs to that track by construction, so pairing them up again
    /// afterwards could only ever get it wrong -- and getting it wrong means
    /// inventing an object that was never there.
    pub fn carry(self: *Tracker, carried: []const ?Detection) void {
        std.debug.assert(carried.len == self.tracks.items.len);
        const stride = self.width * self.height;

        // Backwards, so that forgetting a track does not move the ones still
        // to be looked at.
        var i = carried.len;
        while (i > 0) {
            i -= 1;
            const track = &self.tracks.items[i];
            if (carried[i]) |detection| {
                @memcpy(track.mask, detection.plane[0..stride]);
                track.box = detection.box;
                track.score = detection.score;
                track.hits += 1;
                track.misses = 0;
                continue;
            }
            track.misses += 1;
            if (track.misses <= self.config.max_misses) continue;
            self.allocator.free(track.mask);
            _ = self.tracks.orderedRemove(i);
        }
    }

    /// Which of the readings the decoder offered is this track, carried onto
    /// this frame -- or none of them, when the object is not there any more.
    ///
    /// Prompting is not bounded. Nothing stops the decoder answering with the
    /// wall behind the object, and once the object has left the frame that is
    /// all it can answer with. So a reading has to earn the track three times
    /// over: the decoder has to rate its own mask, the mask has to be the size
    /// the object was, and it has to overlap where the object was. What is
    /// left of those is the reading that overlaps most.
    pub fn carriedFrom(self: Tracker, track: Track, masks: sam3.Masks) ?Detection {
        var carried: ?Detection = null;
        var most_overlap: f32 = 0.0;

        for (0..masks.count) |i| {
            const candidate: Detection = .{
                .box = masks.boxes[i],
                .score = masks.scores[i],
                .plane = masks.plane(i),
            };
            if (candidate.score < self.config.min_carried_score) continue;
            if (!self.plausible(track, candidate)) continue;

            const overlap = sam3.maskIou(candidate.plane, track.mask);
            if (overlap <= most_overlap) continue;
            most_overlap = overlap;
            carried = candidate;
        }
        return carried;
    }

    /// Whether what came back from carrying a track forward can be the same
    /// object at all. An object does not triple in size between two frames, so
    /// a mask that does is a wrong answer, and taking it would only make the
    /// next frame's prompt worse.
    pub fn plausible(self: Tracker, track: Track, detection: Detection) bool {
        const before = track.box.area();
        const after = detection.box.area();
        if (before <= 0.0 or after <= 0.0) return false;
        const change = if (after > before) after / before else before / after;
        return change <= self.config.max_area_change;
    }

    /// Where to prompt the next frame for this track: the point deepest inside
    /// the mask it left on the last one.
    ///
    /// Depth is what matters, not merely being inside. The object will have
    /// moved by the time the next frame is prompted, and a point near the edge
    /// of where it was lands on the background -- whereupon the decoder
    /// faithfully answers with the background, the track takes it, and the
    /// object is lost while still plainly on screen. The deepest point has
    /// half the object's width to give up before that happens.
    pub fn promptFrom(self: Tracker, track: Track) !?sam3.Point {
        if (track.box.isEmpty()) return null;
        const depth = try self.allocator.alloc(u32, self.width * self.height);
        defer self.allocator.free(depth);
        return deepest(track.mask, self.width, self.height, depth);
    }

    /// Replaces what a track is with a sharper reading of the same object.
    /// This is a viewer adding a point to a prompt they have already made:
    /// the answer changes, but the object -- and its number -- does not.
    pub fn refine(self: *Tracker, id: u32, detection: Detection) bool {
        for (self.tracks.items) |*track| {
            if (track.id != id) continue;
            @memcpy(track.mask, detection.plane[0 .. self.width * self.height]);
            track.box = detection.box;
            track.score = detection.score;
            track.misses = 0;
            return true;
        }
        return false;
    }

    /// Adds an object the viewer pointed at, as its own track. Anything
    /// already being followed that this is plainly the same object as is
    /// replaced, so clicking twice on one thing refines it rather than
    /// doubling it.
    pub fn seed(
        self: *Tracker,
        detection: Detection,
        width: usize,
        height: usize,
    ) !u32 {
        if (width != self.width or height != self.height) {
            self.reset();
            self.width = width;
            self.height = height;
        }

        for (self.tracks.items) |*track| {
            if (sam3.maskIou(track.mask, detection.plane) <= self.config.match_iou) continue;
            @memcpy(track.mask, detection.plane[0 .. width * height]);
            track.box = detection.box;
            track.score = detection.score;
            track.hits += 1;
            track.misses = 0;
            return track.id;
        }

        const mask = try self.allocator.dupe(f32, detection.plane[0 .. width * height]);
        errdefer self.allocator.free(mask);
        try self.tracks.append(self.allocator, .{
            .id = self.next_id,
            .box = detection.box,
            .score = detection.score,
            .mask = mask,
            // An object the viewer has just pointed at does not have to prove
            // itself over several frames; they have already seen it.
            .hits = self.config.min_hits,
        });
        self.next_id += 1;
        return self.next_id - 1;
    }

    /// How much a track and a detection look like the same object.
    ///
    /// The masks decide it, because two objects that pass in front of one
    /// another share a box long before they share any pixels. The box only
    /// gates the comparison, to skip the pixel walk for the pairs that are
    /// nowhere near each other.
    fn agreement(track: Track, detection: Detection) f32 {
        if (track.box.iou(detection.box) <= 0.0) return 0.0;
        return sam3.maskIou(track.mask, detection.plane);
    }
};

/// The point of the mask furthest from anything the mask does not claim.
///
/// `depth` is scratch the size of the mask. Every pixel gets its distance to
/// the nearest one the mask does not claim, in two sweeps -- down-right, then
/// up-left -- and the furthest of those is the answer. Everything past the
/// edge of the frame counts as not claimed, so an object hanging half out of
/// shot is prompted at the middle of the half that is in it.
///
/// There is deliberately no box in the prompt either. A box tells SAM the
/// object fills it and SAM answers with a mask that does; read that mask's box
/// back, prompt the next frame with it, and the object grows a little every
/// frame until it is the whole picture. A point does not compound that way.
fn deepest(plane: []const f32, width: usize, height: usize, depth: []u32) ?sam3.Point {
    std.debug.assert(depth.len == width * height);
    const unreachable_distance: u32 = @intCast(width + height);

    for (0..height) |y| {
        for (0..width) |x| {
            const i = y * width + x;
            if (plane[i] <= 0.0) {
                depth[i] = 0;
                continue;
            }
            // The frame's own edge is a boundary like any other.
            var nearest: u32 = if (x == 0 or y == 0) 1 else unreachable_distance;
            if (y > 0) nearest = @min(nearest, depth[i - width] + 1);
            if (x > 0) nearest = @min(nearest, depth[i - 1] + 1);
            depth[i] = nearest;
        }
    }

    var furthest: u32 = 0;
    var y = height;
    while (y > 0) {
        y -= 1;
        var x = width;
        while (x > 0) {
            x -= 1;
            const i = y * width + x;
            if (depth[i] == 0) continue;
            var nearest = depth[i];
            if (x + 1 == width or y + 1 == height) nearest = @min(nearest, 1);
            if (y + 1 < height) nearest = @min(nearest, depth[i + width] + 1);
            if (x + 1 < width) nearest = @min(nearest, depth[i + 1] + 1);
            depth[i] = nearest;
            furthest = @max(furthest, nearest);
        }
    }
    if (furthest == 0) return null;

    // A long thin object is equally deep all along its length, and any of
    // those pixels would do for depth. Take the middle one, so that whichever
    // way the object goes it has the length as well as the depth to give.
    var sum_x: u64 = 0;
    var sum_y: u64 = 0;
    var deepest_count: u64 = 0;
    for (depth, 0..) |distance, i| {
        if (distance != furthest) continue;
        sum_x += i % width;
        sum_y += i / width;
        deepest_count += 1;
    }
    const mean_x = sum_x / deepest_count;
    const mean_y = sum_y / deepest_count;

    var index: usize = 0;
    var closest: u64 = std.math.maxInt(u64);
    for (depth, 0..) |distance, i| {
        if (distance != furthest) continue;
        const dx = @abs(@as(i64, @intCast(i % width)) - @as(i64, @intCast(mean_x)));
        const dy = @abs(@as(i64, @intCast(i / width)) - @as(i64, @intCast(mean_y)));
        const away: u64 = @intCast(dx * dx + dy * dy);
        if (away >= closest) continue;
        closest = away;
        index = i;
    }
    return .{
        .x = (@as(f32, @floatFromInt(index % width)) + 0.5) / @as(f32, @floatFromInt(width)),
        .y = (@as(f32, @floatFromInt(index / width)) + 0.5) / @as(f32, @floatFromInt(height)),
        .label = 1,
    };
}

const testing = std.testing;

/// A mask that claims one rectangle of a `size` by `size` frame.
fn rectangle(plane: []f32, size: usize, x0: usize, y0: usize, x1: usize, y1: usize) void {
    @memset(plane, -1.0);
    for (y0..y1) |y| {
        for (x0..x1) |x| plane[y * size + x] = 1.0;
    }
}

fn detectionOf(plane: []const f32, size: usize, score: f32) Detection {
    return .{ .box = sam3.maskBox(plane, size, size), .score = score, .plane = plane };
}

test "an object that moves keeps the number it was given" {
    const size = 8;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    var plane: [size * size]f32 = undefined;
    rectangle(&plane, size, 1, 1, 4, 4);
    try tracker.update(&.{detectionOf(&plane, size, 0.9)}, size, size);

    try testing.expectEqual(@as(usize, 1), tracker.tracks.items.len);
    const id = tracker.tracks.items[0].id;

    // The same object, one pixel to the right on each of the next two frames.
    for (0..2) |step| {
        rectangle(&plane, size, 2 + step, 1, 5 + step, 4);
        try tracker.update(&.{detectionOf(&plane, size, 0.9)}, size, size);
    }

    try testing.expectEqual(@as(usize, 1), tracker.tracks.items.len);
    try testing.expectEqual(id, tracker.tracks.items[0].id);
    try testing.expectEqual(@as(u32, 3), tracker.tracks.items[0].hits);
    try testing.expectEqual(@as(u32, 0), tracker.tracks.items[0].misses);
    try testing.expectEqual(sam3.maskBox(&plane, size, size), tracker.tracks.items[0].box);
}

test "an object somewhere else is a different object" {
    const size = 8;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    var here: [size * size]f32 = undefined;
    rectangle(&here, size, 0, 0, 3, 3);
    try tracker.update(&.{detectionOf(&here, size, 0.9)}, size, size);

    var there: [size * size]f32 = undefined;
    rectangle(&there, size, 5, 5, 8, 8);
    try tracker.update(&.{detectionOf(&there, size, 0.9)}, size, size);

    try testing.expectEqual(@as(usize, 2), tracker.tracks.items.len);
    try testing.expectEqual(@as(u32, 1), tracker.tracks.items[0].id);
    try testing.expectEqual(@as(u32, 2), tracker.tracks.items[1].id);
    // The first one was not seen this frame, but is not given up on yet.
    try testing.expectEqual(@as(u32, 1), tracker.tracks.items[0].misses);
    try testing.expect(!tracker.tracks.items[0].isVisible(tracker.config));
}

test "two objects that cross keep their own masks" {
    const size = 16;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    var tall: [size * size]f32 = undefined;
    var wide: [size * size]f32 = undefined;
    rectangle(&tall, size, 6, 0, 10, 16);
    rectangle(&wide, size, 0, 6, 16, 10);

    try tracker.update(&.{
        detectionOf(&tall, size, 0.9),
        detectionOf(&wide, size, 0.8),
    }, size, size);
    try testing.expectEqual(@as(usize, 2), tracker.tracks.items.len);

    // Their boxes overlap in the middle, so only the pixels tell them apart.
    // Offering them in the other order must not swap the two tracks.
    try tracker.update(&.{
        detectionOf(&wide, size, 0.8),
        detectionOf(&tall, size, 0.9),
    }, size, size);

    try testing.expectEqual(@as(usize, 2), tracker.tracks.items.len);
    try testing.expectEqual(@as(u32, 1), tracker.tracks.items[0].id);
    try testing.expectEqual(@as(f32, 1.0), sam3.maskIou(tracker.tracks.items[0].mask, &tall));
    try testing.expectEqual(@as(u32, 2), tracker.tracks.items[1].id);
    try testing.expectEqual(@as(f32, 1.0), sam3.maskIou(tracker.tracks.items[1].mask, &wide));
}

test "a track survives being missed, but not forever" {
    const size = 8;
    var tracker: Tracker = .init(testing.allocator, .{ .max_misses = 2 });
    defer tracker.deinit();

    var plane: [size * size]f32 = undefined;
    rectangle(&plane, size, 1, 1, 4, 4);
    try tracker.update(&.{detectionOf(&plane, size, 0.9)}, size, size);

    for (1..3) |misses| {
        try tracker.update(&.{}, size, size);
        try testing.expectEqual(@as(usize, 1), tracker.tracks.items.len);
        try testing.expectEqual(@as(u32, @intCast(misses)), tracker.tracks.items[0].misses);
    }

    // Back on screen before it was given up on, and still itself.
    try tracker.update(&.{detectionOf(&plane, size, 0.9)}, size, size);
    try testing.expectEqual(@as(u32, 1), tracker.tracks.items[0].id);
    try testing.expectEqual(@as(u32, 0), tracker.tracks.items[0].misses);

    for (0..3) |_| try tracker.update(&.{}, size, size);
    try testing.expectEqual(@as(usize, 0), tracker.tracks.items.len);

    // The number it had is not handed to the next object.
    try tracker.update(&.{detectionOf(&plane, size, 0.9)}, size, size);
    try testing.expectEqual(@as(u32, 2), tracker.tracks.items[0].id);
}

test "adding a point to a prompt changes the answer, not the object" {
    const size = 8;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    var first: [size * size]f32 = undefined;
    rectangle(&first, size, 3, 3, 5, 5);
    const id = try tracker.seed(detectionOf(&first, size, 0.5), size, size);

    // A second point turned a scrap of the object into the whole of it. The
    // two masks barely overlap, and it is still the same object.
    var whole: [size * size]f32 = undefined;
    rectangle(&whole, size, 0, 0, 7, 7);
    try testing.expect(tracker.refine(id, detectionOf(&whole, size, 0.95)));

    try testing.expectEqual(@as(usize, 1), tracker.tracks.items.len);
    try testing.expectEqual(id, tracker.tracks.items[0].id);
    try testing.expectEqual(@as(f32, 0.95), tracker.tracks.items[0].score);
    try testing.expectEqual(@as(f32, 1.0), sam3.maskIou(tracker.tracks.items[0].mask, &whole));

    // A track that is no longer there cannot be refined.
    try testing.expect(!tracker.refine(id + 99, detectionOf(&whole, size, 0.95)));
}

test "clicking the same object again refines it instead of doubling it" {
    const size = 8;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    var small: [size * size]f32 = undefined;
    rectangle(&small, size, 1, 1, 4, 4);
    const first = try tracker.seed(detectionOf(&small, size, 0.5), size, size);

    var larger: [size * size]f32 = undefined;
    rectangle(&larger, size, 1, 1, 5, 5);
    const again = try tracker.seed(detectionOf(&larger, size, 0.7), size, size);

    try testing.expectEqual(first, again);
    try testing.expectEqual(@as(usize, 1), tracker.tracks.items.len);
    try testing.expectEqual(@as(f32, 0.7), tracker.tracks.items[0].score);

    var elsewhere: [size * size]f32 = undefined;
    rectangle(&elsewhere, size, 6, 6, 8, 8);
    const other = try tracker.seed(detectionOf(&elsewhere, size, 0.6), size, size);
    try testing.expect(other != first);
    try testing.expectEqual(@as(usize, 2), tracker.tracks.items.len);
}

test "a frame at a different resolution starts the tracker over" {
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    var small: [4 * 4]f32 = undefined;
    rectangle(&small, 4, 0, 0, 2, 2);
    try tracker.update(&.{detectionOf(&small, 4, 0.9)}, 4, 4);
    try testing.expectEqual(@as(usize, 1), tracker.tracks.items.len);

    var large: [8 * 8]f32 = undefined;
    rectangle(&large, 8, 0, 0, 4, 4);
    try tracker.update(&.{detectionOf(&large, 8, 0.9)}, 8, 8);

    try testing.expectEqual(@as(usize, 1), tracker.tracks.items.len);
    try testing.expectEqual(@as(usize, 8), tracker.width);
    try testing.expectEqual(@as(u32, 1), tracker.tracks.items[0].id);
}

/// Where a prompt point falls, as pixel indices into a `size` by `size` frame.
fn promptPixel(tracker: Tracker, size: usize) !struct { x: usize, y: usize } {
    const point = (try tracker.promptFrom(tracker.tracks.items[0])).?;
    try testing.expectEqual(@as(i64, 1), point.label);
    return .{
        .x = @intFromFloat(point.x * @as(f32, @floatFromInt(size))),
        .y = @intFromFloat(point.y * @as(f32, @floatFromInt(size))),
    };
}

test "a track is prompted where its mask is deepest, not merely where it is" {
    const size = 16;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    // A wide, shallow bar: ten across and four down. Plenty of it is inside,
    // but only the middle two rows are as far from an edge as it gets.
    var plane: [size * size]f32 = undefined;
    rectangle(&plane, size, 2, 6, 12, 10);
    _ = try tracker.seed(detectionOf(&plane, size, 0.9), size, size);

    // Every pixel along the middle of the bar is equally deep; the one taken
    // is the one in the middle of those, not whichever end came first.
    const at = try promptPixel(tracker, size);
    try testing.expect(plane[at.y * size + at.x] > 0.0);
    try testing.expect(at.y == 7 or at.y == 8);
    try testing.expect(at.x >= 5 and at.x <= 8);
}

test "a mask hard against the frame's edge is prompted inside what is showing" {
    const size = 16;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    // An object half out of shot: the frame's edge cuts it off at x = 0.
    var plane: [size * size]f32 = undefined;
    rectangle(&plane, size, 0, 4, 8, 12);
    _ = try tracker.seed(detectionOf(&plane, size, 0.9), size, size);

    // The edge of the frame bounds it like any other edge, so the point sits
    // in the middle of the part that is showing rather than at x = 0.
    const at = try promptPixel(tracker, size);
    try testing.expect(plane[at.y * size + at.x] > 0.0);
    try testing.expect(at.x >= 3);
    try testing.expect(at.y >= 6 and at.y < 10);
}

test "a track takes the reading that is still the object, and otherwise none" {
    const size = 8;
    const stride = size * size;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    var was: [stride]f32 = undefined;
    rectangle(&was, size, 2, 2, 6, 6);
    _ = try tracker.seed(detectionOf(&was, size, 0.9), size, size);
    const track = tracker.tracks.items[0];

    // Three readings of the next frame: the object a pixel along, the whole
    // frame, and the object again but with the decoder unsure of it.
    var logits: [3 * stride]f32 = undefined;
    rectangle(logits[0..stride], size, 3, 2, 7, 6);
    rectangle(logits[stride .. 2 * stride], size, 0, 0, size, size);
    rectangle(logits[2 * stride ..], size, 3, 2, 7, 6);

    var boxes: [3]sam3.Box = undefined;
    for (&boxes, 0..) |*box, i| box.* = sam3.maskBox(logits[i * stride ..][0..stride], size, size);
    var scores = [_]f32{ 0.9, 0.95, 0.2 };

    const masks: sam3.Masks = .{
        .allocator = testing.allocator,
        .logits = &logits,
        .scores = &scores,
        .boxes = &boxes,
        .count = 3,
        .width = size,
        .height = size,
        .object_score = 0,
    };

    // The whole frame is too big to be the object and the unsure one is not
    // worth taking, which leaves the one that is both.
    const carried = tracker.carriedFrom(track, masks).?;
    try testing.expectEqual(@as(f32, 0.9), carried.score);
    try testing.expectEqual(boxes[0], carried.box);

    // With the object gone, every reading is of something else.
    scores = .{ 0.2, 0.95, 0.2 };
    try testing.expectEqual(null, tracker.carriedFrom(track, masks));

    // As is a reading that shares no ground with where the object was.
    scores = .{ 0.9, 0.1, 0.1 };
    rectangle(logits[0..stride], size, 6, 6, 8, 8);
    boxes[0] = sam3.maskBox(logits[0..stride], size, size);
    try testing.expectEqual(null, tracker.carriedFrom(track, masks));
}

test "a track that is carried forward stays itself" {
    const size = 8;
    var tracker: Tracker = .init(testing.allocator, .{ .max_misses = 1 });
    defer tracker.deinit();

    var first: [size * size]f32 = undefined;
    var second: [size * size]f32 = undefined;
    rectangle(&first, size, 1, 1, 4, 4);
    rectangle(&second, size, 4, 4, 7, 7);
    _ = try tracker.seed(detectionOf(&first, size, 0.9), size, size);
    _ = try tracker.seed(detectionOf(&second, size, 0.8), size, size);
    try testing.expectEqual(@as(usize, 2), tracker.tracks.items.len);

    // The first one moved right across the frame, further than it overlaps
    // itself. Because it was carried rather than matched, it is still itself.
    var moved: [size * size]f32 = undefined;
    rectangle(&moved, size, 5, 1, 8, 4);
    tracker.carry(&.{ detectionOf(&moved, size, 0.7), null });

    try testing.expectEqual(@as(usize, 2), tracker.tracks.items.len);
    try testing.expectEqual(@as(u32, 1), tracker.tracks.items[0].id);
    try testing.expectEqual(@as(f32, 1.0), sam3.maskIou(tracker.tracks.items[0].mask, &moved));
    try testing.expectEqual(@as(u32, 0), tracker.tracks.items[0].misses);

    // The second was not found on this frame, or the next one.
    try testing.expectEqual(@as(u32, 1), tracker.tracks.items[1].misses);
    tracker.carry(&.{ null, null });
    try testing.expectEqual(@as(usize, 1), tracker.tracks.items.len);
    try testing.expectEqual(@as(u32, 1), tracker.tracks.items[0].id);
}

test "a mask that changes size all at once is not the object it came from" {
    const size = 8;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    var plane: [size * size]f32 = undefined;
    rectangle(&plane, size, 2, 2, 4, 4);
    _ = try tracker.seed(detectionOf(&plane, size, 0.9), size, size);
    const track = tracker.tracks.items[0];

    var nudged: [size * size]f32 = undefined;
    rectangle(&nudged, size, 3, 2, 5, 5);
    try testing.expect(tracker.plausible(track, detectionOf(&nudged, size, 0.9)));

    // The decoder answered with the whole frame instead of the object.
    var everything: [size * size]f32 = undefined;
    rectangle(&everything, size, 0, 0, size, size);
    try testing.expect(!tracker.plausible(track, detectionOf(&everything, size, 0.9)));

    // Or with a sliver of it.
    var sliver: [size * size]f32 = undefined;
    rectangle(&sliver, size, 2, 2, 3, 3);
    try testing.expect(!tracker.plausible(track, detectionOf(&sliver, size, 0.9)));

    // Or with nothing at all.
    const nothing: [size * size]f32 = @splat(-1.0);
    try testing.expect(!tracker.plausible(track, detectionOf(&nothing, size, 0.9)));
}

test "a mask the centre of mass falls outside is prompted on its own pixels" {
    const size = 8;
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();

    // A ring: its centre of mass is the hole in the middle, and prompting
    // there would ask the decoder for the hole.
    var plane: [size * size]f32 = @splat(-1.0);
    for (1..7) |y| {
        for (1..7) |x| {
            if (x > 1 and x < 6 and y > 1 and y < 6) continue;
            plane[y * size + x] = 1.0;
        }
    }
    _ = try tracker.seed(detectionOf(&plane, size, 0.9), size, size);

    const at = try promptPixel(tracker, size);
    try testing.expect(plane[at.y * size + at.x] > 0.0);
}

test "a track with nothing left of it has no prompt to give" {
    var tracker: Tracker = .init(testing.allocator, .{});
    defer tracker.deinit();
    tracker.width = 4;
    tracker.height = 4;

    var plane: [16]f32 = @splat(-1.0);
    const track: Track = .{ .id = 1, .box = .{}, .score = 0, .mask = &plane };
    try testing.expectEqual(null, try tracker.promptFrom(track));

    // A box says there is something, but the mask has nothing left in it.
    const hollow: Track = .{
        .id = 1,
        .box = .{ .x0 = 0, .y0 = 0, .x1 = 1, .y1 = 1 },
        .score = 0,
        .mask = &plane,
    };
    try testing.expectEqual(null, try tracker.promptFrom(hollow));
}
