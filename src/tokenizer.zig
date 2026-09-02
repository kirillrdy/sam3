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
    arena: std.heap.ArenaAllocator,
    vocab: std.StringHashMapUnmanaged(i64) = .empty,
    merges: std.StringHashMapUnmanaged(usize) = .empty,

    pub fn init(allocator: std.mem.Allocator, bytes: []const u8) !Tokenizer {
        const TokenizerData = struct {
            model: ?struct {
                vocab: std.json.ArrayHashMap(i64),
                merges: []const [2][]const u8,
            } = null,
        };

        var parsed = std.json.parseFromSlice(TokenizerData, allocator, bytes, .{
            .ignore_unknown_fields = true,
        }) catch return error.BadTokenizer;
        defer parsed.deinit();

        const model = parsed.value.model orelse return error.BadTokenizer;
        if (model.vocab.map.count() == 0 or model.merges.len == 0) return error.BadTokenizer;

        var self: Tokenizer = .{
            .allocator = allocator,
            .arena = .init(allocator),
        };
        errdefer self.deinit();

        const strings = self.arena.allocator();

        try self.vocab.ensureTotalCapacity(allocator, @intCast(model.vocab.map.count()));
        for (model.vocab.map.keys(), model.vocab.map.values()) |k, v| {
            const owned = try strings.dupe(u8, k);
            self.vocab.putAssumeCapacity(owned, v);
        }

        try self.merges.ensureTotalCapacity(allocator, @intCast(model.merges.len));
        for (model.merges, 0..) |m, rank| {
            const key = try std.fmt.allocPrint(strings, "{s}\x00{s}", .{ m[0], m[1] });
            self.merges.putAssumeCapacity(key, rank);
        }

        return self;
    }

    pub fn deinit(self: *Tokenizer) void {
        self.merges.deinit(self.allocator);
        self.vocab.deinit(self.allocator);
        self.arena.deinit();
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
