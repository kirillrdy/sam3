// Rebuild the embedded atlas with a Source Code Pro Regular OTF and FreeType:
// cc generate_font.c $(pkg-config --cflags --libs freetype2) -o generate_font
// ./generate_font SourceCodePro-Regular.otf font_atlas.bin
#include <ft2build.h>
#include FT_FREETYPE_H
#include <stdio.h>
#include <string.h>

enum { first = 32, count = 95, width = 9, height = 18, baseline = 14 };

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    FT_Library library;
    FT_Face face;
    if (FT_Init_FreeType(&library)) return 1;
    if (FT_New_Face(library, argv[1], 0, &face)) return 1;
    if (FT_Set_Pixel_Sizes(face, 0, 15)) return 1;

    FILE *out = fopen(argv[2], "wb");
    if (!out) return 1;
    for (int ch = first; ch < first + count; ch++) {
        unsigned char cell[width * height] = {0};
        if (FT_Load_Char(face, ch, FT_LOAD_RENDER)) return 1;
        FT_GlyphSlot glyph = face->glyph;
        FT_Bitmap bitmap = glyph->bitmap;
        for (unsigned int y = 0; y < bitmap.rows; y++) {
            for (unsigned int x = 0; x < bitmap.width; x++) {
                int dx = glyph->bitmap_left + (int)x;
                int dy = baseline - glyph->bitmap_top + (int)y;
                if (dx >= 0 && dx < width && dy >= 0 && dy < height)
                    cell[dy * width + dx] = bitmap.buffer[y * bitmap.pitch + x];
            }
        }
        if (fwrite(cell, 1, sizeof cell, out) != sizeof cell) return 1;
    }
    fclose(out);
    FT_Done_Face(face);
    FT_Done_FreeType(library);
    return 0;
}
