const std = @import("std");

pub const max_tokens = 32;
pub const bos_token: i64 = 49406;
pub const eos_token: i64 = 49407;

pub const Encoding = struct {
    ids: [max_tokens]i64,
    attention: [max_tokens]i64,
};

pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    /// Owns every key the two maps point at. Held by pointer because a
    /// `Tokenizer` is returned and stored by value, and an arena cannot be
    /// allocated from once it has been copied.
    strings: *std.heap.ArenaAllocator,
    vocab: std.StringHashMapUnmanaged(i64) = .empty,
    merges: std.StringHashMapUnmanaged(usize) = .empty,

    /// Reads the vocabulary and the merge ranks straight off the token stream.
    /// Building a `std.json.Value` of the whole file instead costs a few
    /// hundred megabytes for the life of the process, against the ~1.5 MiB of
    /// keys that are actually wanted.
    pub fn init(allocator: std.mem.Allocator, bytes: []const u8) !Tokenizer {
        const strings = try allocator.create(std.heap.ArenaAllocator);
        strings.* = .init(allocator);

        // Owns `strings` from here on, so releasing it is `deinit`'s alone.
        var self: Tokenizer = .{ .allocator = allocator, .strings = strings };
        errdefer self.deinit();

        var scratch: std.heap.ArenaAllocator = .init(allocator);
        defer scratch.deinit();

        var scanner: std.json.Scanner = .initCompleteInput(allocator, bytes);
        defer scanner.deinit();

        try expect(&scanner, .object_begin);
        while (try nextKey(&scanner, scratch.allocator())) |key| {
            if (!std.mem.eql(u8, key, "model")) {
                try scanner.skipValue();
                continue;
            }
            try expect(&scanner, .object_begin);
            while (try nextKey(&scanner, scratch.allocator())) |field| {
                if (std.mem.eql(u8, field, "vocab")) {
                    try self.readVocab(&scanner, scratch.allocator());
                } else if (std.mem.eql(u8, field, "merges")) {
                    try self.readMerges(&scanner, scratch.allocator());
                } else {
                    try scanner.skipValue();
                }
            }
        }

        if (self.vocab.count() == 0 or self.merges.count() == 0) return error.BadTokenizer;
        return self;
    }

    fn readVocab(self: *Tokenizer, scanner: *std.json.Scanner, scratch: std.mem.Allocator) !void {
        try expect(scanner, .object_begin);
        while (try nextKey(scanner, scratch)) |token| {
            const id = try nextInteger(scanner, scratch);
            const owned = try self.strings.allocator().dupe(u8, token);
            try self.vocab.put(self.allocator, owned, id);
        }
    }

    fn readMerges(self: *Tokenizer, scanner: *std.json.Scanner, scratch: std.mem.Allocator) !void {
        try expect(scanner, .array_begin);
        var rank: usize = 0;
        while (true) : (rank += 1) {
            switch (try scanner.next()) {
                .array_begin => {},
                .array_end => return,
                else => return error.BadTokenizer,
            }
            const left = try nextString(scanner, scratch);
            const right = try nextString(scanner, scratch);
            try expect(scanner, .array_end);

            const key = try std.fmt.allocPrint(self.strings.allocator(), "{s}\x00{s}", .{ left, right });
            try self.merges.put(self.allocator, key, rank);
        }
    }

    pub fn deinit(self: *Tokenizer) void {
        self.merges.deinit(self.allocator);
        self.vocab.deinit(self.allocator);
        self.strings.deinit();
        self.allocator.destroy(self.strings);
        self.* = undefined;
    }

    pub fn encode(self: *const Tokenizer, text: []const u8) !Encoding {
        var result: Encoding = .{
            .ids = @splat(eos_token),
            .attention = @splat(0),
        };
        result.ids[0] = bos_token;
        result.attention[0] = 1;

        var arena_state = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const normalized = try arena.dupe(u8, text);
        for (normalized) |*byte| byte.* = std.ascii.toLower(byte.*);

        var cursor: usize = 0;
        var count: usize = 1;
        while (nextPiece(normalized, &cursor)) |piece| {
            const ids = try self.encodePiece(arena, piece);
            for (ids) |id| {
                if (count == max_tokens - 1) break;
                result.ids[count] = id;
                result.attention[count] = 1;
                count += 1;
            }
            if (count == max_tokens - 1) break;
        }
        result.ids[count] = eos_token;
        result.attention[count] = 1;
        return result;
    }

    fn encodePiece(self: *const Tokenizer, allocator: std.mem.Allocator, piece: []const u8) ![]const i64 {
        var symbols: std.ArrayList([]const u8) = .empty;
        for (piece, 0..) |byte, i| {
            var encoded: [4]u8 = undefined;
            const len = try std.unicode.utf8Encode(byteCodepoint(byte), &encoded);
            const suffix = if (i + 1 == piece.len) "</w>" else "";
            try symbols.append(allocator, try std.fmt.allocPrint(allocator, "{s}{s}", .{ encoded[0..len], suffix }));
        }

        while (symbols.items.len > 1) {
            var best_rank: usize = std.math.maxInt(usize);
            var best_left: ?[]const u8 = null;
            var best_right: ?[]const u8 = null;
            for (symbols.items[0 .. symbols.items.len - 1], symbols.items[1..]) |left, right| {
                const key = try std.fmt.allocPrint(allocator, "{s}\x00{s}", .{ left, right });
                if (self.merges.get(key)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_left = left;
                        best_right = right;
                    }
                }
            }
            const left = best_left orelse break;
            const right = best_right.?;

            var merged: std.ArrayList([]const u8) = .empty;
            var i: usize = 0;
            while (i < symbols.items.len) {
                if (i + 1 < symbols.items.len and
                    std.mem.eql(u8, symbols.items[i], left) and
                    std.mem.eql(u8, symbols.items[i + 1], right))
                {
                    try merged.append(allocator, try std.fmt.allocPrint(
                        allocator,
                        "{s}{s}",
                        .{ symbols.items[i], symbols.items[i + 1] },
                    ));
                    i += 2;
                } else {
                    try merged.append(allocator, symbols.items[i]);
                    i += 1;
                }
            }
            symbols = merged;
        }

        const ids = try allocator.alloc(i64, symbols.items.len);
        for (symbols.items, ids) |symbol, *id| id.* = self.vocab.get(symbol) orelse eos_token;
        return ids;
    }
};

fn expect(scanner: *std.json.Scanner, want: std.meta.Tag(std.json.Token)) !void {
    if (std.meta.activeTag(try scanner.next()) != want) return error.BadTokenizer;
}

/// The next object key, or null once the object has ended. The slice is only
/// valid until `scratch` is released.
fn nextKey(scanner: *std.json.Scanner, scratch: std.mem.Allocator) !?[]const u8 {
    return switch (try scanner.nextAlloc(scratch, .alloc_if_needed)) {
        .string, .allocated_string => |text| text,
        .object_end => null,
        else => error.BadTokenizer,
    };
}

fn nextString(scanner: *std.json.Scanner, scratch: std.mem.Allocator) ![]const u8 {
    return switch (try scanner.nextAlloc(scratch, .alloc_if_needed)) {
        .string, .allocated_string => |text| text,
        else => error.BadTokenizer,
    };
}

fn nextInteger(scanner: *std.json.Scanner, scratch: std.mem.Allocator) !i64 {
    const digits = switch (try scanner.nextAlloc(scratch, .alloc_if_needed)) {
        .number, .allocated_number => |text| text,
        else => return error.BadTokenizer,
    };
    return std.fmt.parseInt(i64, digits, 10) catch error.BadTokenizer;
}

fn nextPiece(text: []const u8, cursor: *usize) ?[]const u8 {
    while (cursor.* < text.len and std.ascii.isWhitespace(text[cursor.*])) cursor.* += 1;
    if (cursor.* == text.len) return null;

    const start = cursor.*;
    const first = text[cursor.*];
    if (first == '\'' and contractionLength(text[start..]) != 0) {
        cursor.* += contractionLength(text[start..]);
    } else if (isLetter(first)) {
        cursor.* += 1;
        while (cursor.* < text.len and isLetter(text[cursor.*])) cursor.* += 1;
    } else if (std.ascii.isDigit(first)) {
        cursor.* += 1;
    } else {
        cursor.* += 1;
        while (cursor.* < text.len and
            !std.ascii.isWhitespace(text[cursor.*]) and
            !isLetter(text[cursor.*]) and
            !std.ascii.isDigit(text[cursor.*]) and
            text[cursor.*] != '\'') cursor.* += 1;
    }
    return text[start..cursor.*];
}

fn contractionLength(text: []const u8) usize {
    for ([_][]const u8{ "'re", "'ve", "'ll", "'s", "'t", "'m", "'d" }) |suffix| {
        if (std.mem.startsWith(u8, text, suffix)) return suffix.len;
    }
    return 0;
}

fn isLetter(byte: u8) bool {
    return std.ascii.isAlphabetic(byte) or byte >= 0x80;
}

fn byteCodepoint(byte: u8) u21 {
    if ((byte >= '!' and byte <= '~') or
        (byte >= 0xa1 and byte <= 0xac) or
        (byte >= 0xae)) return byte;

    var offset: u21 = 0;
    var candidate: u16 = 0;
    while (candidate < byte) : (candidate += 1) {
        const value: u8 = @intCast(candidate);
        if (!((value >= '!' and value <= '~') or
            (value >= 0xa1 and value <= 0xac) or
            value >= 0xae)) offset += 1;
    }
    return 256 + offset;
}

test "the vocabulary and the merge ranks are read past everything else" {
    const json =
        \\{
        \\  "version": "1.0",
        \\  "added_tokens": [{"id": 0, "content": "<|startoftext|>"}],
        \\  "normalizer": {"type": "Sequence", "normalizers": []},
        \\  "model": {
        \\    "type": "BPE",
        \\    "dropout": null,
        \\    "vocab": {"a</w>": 7, "in": 11, "\u00e9": 12, "\"": 13},
        \\    "merges": [["i", "n"], ["t", "h"]]
        \\  }
        \\}
    ;

    var tok = try Tokenizer.init(std.testing.allocator, json);
    defer tok.deinit();

    try std.testing.expectEqual(4, tok.vocab.count());
    try std.testing.expectEqual(@as(i64, 7), tok.vocab.get("a</w>").?);
    try std.testing.expectEqual(@as(i64, 11), tok.vocab.get("in").?);
    // Escapes have to survive the stream, since keys are matched byte for byte.
    try std.testing.expectEqual(@as(i64, 12), tok.vocab.get("\u{e9}").?);
    try std.testing.expectEqual(@as(i64, 13), tok.vocab.get("\"").?);

    try std.testing.expectEqual(2, tok.merges.count());
    try std.testing.expectEqual(@as(usize, 0), tok.merges.get("i\x00n").?);
    try std.testing.expectEqual(@as(usize, 1), tok.merges.get("t\x00h").?);
}

test "a document with no BPE model in it is refused" {
    try std.testing.expectError(
        error.BadTokenizer,
        Tokenizer.init(std.testing.allocator, "{\"version\": \"1.0\"}"),
    );
}

test "CLIP byte encoding uses the GPT-2 alphabet" {
    try std.testing.expectEqual(@as(u21, '!'), byteCodepoint('!'));
    try std.testing.expectEqual(@as(u21, 256), byteCodepoint(0));
    try std.testing.expectEqual(@as(u21, 288), byteCodepoint(' '));
}

test "pre-tokenization separates words, digits, punctuation, and contractions" {
    const text = "Red cars, don't!";
    var cursor: usize = 0;
    for ([_][]const u8{ "Red", "cars", ",", "don", "'t", "!" }) |want| {
        try std.testing.expectEqualStrings(want, nextPiece(text, &cursor).?);
    }
    try std.testing.expectEqual(null, nextPiece(text, &cursor));
}
