# -*- coding: UTF-8 -*-
import os
import struct
import zlib


WIDTH = 32
HEIGHT = 32

GLYPHS = {
    "H": [
        "10001",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "T": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "U": [
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11111",
    ],
}


def make_canvas():
    return [[255 for _ in range(WIDTH)] for _ in range(HEIGHT)]


def set_pixel(canvas, x, y, value=0):
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        canvas[y][x] = value


def fill_rect(canvas, x0, y0, x1, y1, value=0):
    for y in range(max(0, y0), min(HEIGHT, y1)):
        for x in range(max(0, x0), min(WIDTH, x1)):
            canvas[y][x] = value


def blit_glyph(canvas, ch, origin_x, origin_y, scale_x, scale_y, slant=0.0, embolden=0):
    glyph = GLYPHS[ch]
    for row_idx, row in enumerate(glyph):
        row_shift = int(round((row_idx - len(glyph) / 2) * slant))
        for col_idx, bit in enumerate(row):
            if bit != "1":
                continue
            x0 = origin_x + col_idx * scale_x + row_shift
            y0 = origin_y + row_idx * scale_y
            fill_rect(canvas, x0, y0, x0 + scale_x + embolden, y0 + scale_y + embolden)


def outline(canvas):
    source = [row[:] for row in canvas]
    outlined = [row[:] for row in canvas]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if source[y][x] == 0:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                            outlined[ny][nx] = 0
    return outlined


def add_top_bar(canvas, y, thickness=1):
    fill_rect(canvas, 3, y, WIDTH - 3, y + thickness)


def add_bottom_bar(canvas, y, thickness=1):
    fill_rect(canvas, 4, y, WIDTH - 4, y + thickness)


def add_side_accents(canvas):
    fill_rect(canvas, 2, 10, 3, 22)
    fill_rect(canvas, WIDTH - 3, 10, WIDTH - 2, 22)


def render_style(style):
    canvas = make_canvas()
    x = style["start_x"]
    y = style["start_y"]
    for ch in "HTU":
        blit_glyph(
            canvas,
            ch,
            x,
            y,
            style["scale_x"],
            style["scale_y"],
            slant=style.get("slant", 0.0),
            embolden=style.get("embolden", 0),
        )
        x += style["advance"]

    if style.get("outline"):
        canvas = outline(canvas)
    if style.get("top_bar"):
        add_top_bar(canvas, style["top_bar_y"], style.get("bar_thickness", 1))
    if style.get("bottom_bar"):
        add_bottom_bar(canvas, style["bottom_bar_y"], style.get("bar_thickness", 1))
    if style.get("side_accents"):
        add_side_accents(canvas)
    return canvas


def png_chunk(tag, data):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def save_png(canvas, path):
    raw = bytearray()
    for row in canvas:
        raw.append(0)
        raw.extend(row)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(
        png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 0, 0, 0, 0),
        )
    )
    png.extend(png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    png.extend(png_chunk(b"IEND", b""))

    with open(path, "wb") as f:
        f.write(png)


def main():
    styles = [
        {"start_x": 2, "start_y": 9, "scale_x": 1, "scale_y": 2, "advance": 9},
        {"start_x": 1, "start_y": 9, "scale_x": 1, "scale_y": 2, "advance": 10, "slant": 0.15},
        {"start_x": 2, "start_y": 8, "scale_x": 2, "scale_y": 2, "advance": 9},
        {"start_x": 3, "start_y": 10, "scale_x": 1, "scale_y": 1, "advance": 8, "outline": True},
        {"start_x": 2, "start_y": 8, "scale_x": 1, "scale_y": 2, "advance": 9, "embolden": 1},
        {"start_x": 2, "start_y": 7, "scale_x": 1, "scale_y": 2, "advance": 9, "top_bar": True, "top_bar_y": 5},
        {"start_x": 2, "start_y": 8, "scale_x": 1, "scale_y": 2, "advance": 9, "bottom_bar": True, "bottom_bar_y": 25},
        {"start_x": 2, "start_y": 8, "scale_x": 1, "scale_y": 2, "advance": 9, "side_accents": True},
        {"start_x": 3, "start_y": 9, "scale_x": 1, "scale_y": 2, "advance": 8, "slant": -0.2},
        {
            "start_x": 2,
            "start_y": 9,
            "scale_x": 1,
            "scale_y": 2,
            "advance": 9,
            "outline": True,
            "top_bar": True,
            "top_bar_y": 4,
            "bar_thickness": 2,
        },
    ]

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "my_trigger_images",
    )
    os.makedirs(output_dir, exist_ok=True)

    for idx, style in enumerate(styles):
        canvas = render_style(style)
        save_png(canvas, os.path.join(output_dir, f"htu_{idx:02d}.png"))

    print(f"Generated {len(styles)} PNG images in {output_dir}")


if __name__ == "__main__":
    main()
