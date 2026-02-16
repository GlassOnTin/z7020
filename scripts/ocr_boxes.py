#!/usr/bin/env python3
"""OCR bounding box tool — extracts text strings with pixel coordinates from screenshots.

Usage:
    python3 scripts/ocr_boxes.py /tmp/screenshot.png
    python3 scripts/ocr_boxes.py /tmp/screenshot.png --min-conf 60
    python3 scripts/ocr_boxes.py /tmp/screenshot.png --json
    python3 scripts/ocr_boxes.py /tmp/screenshot.png --filter "button"
    python3 scripts/ocr_boxes.py /tmp/screenshot.png --scale 1.58  # for HiDPI screenshots
    python3 scripts/ocr_boxes.py /tmp/screenshot.png --crop 100,200,800,600

    # Chrome MCP workflow — coordinates match Chrome MCP click targets:
    python3 scripts/ocr_boxes.py --capture "Colab" --chrome-mcp

Outputs each detected text string with its bounding box (x, y, w, h) in pixel coordinates.
The --scale flag divides coordinates by the given factor (useful when browser screenshots
are captured at a different resolution than CSS pixels).
The --crop flag crops the image to (left,top,right,bottom) before OCR, and adjusts
returned coordinates to be relative to the original image.
The --chrome-mcp flag applies scale=1.578 and y-offset=78 to convert native window
coordinates to Chrome MCP viewport coordinates (calibrate with --calibrate if needed).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytesseract
from PIL import Image


def ocr_with_boxes(image_path, min_conf=40, scale=1.0, crop=None):
    """Run OCR on image and return list of {text, x, y, w, h, conf, center_x, center_y}.

    crop: (left, top, right, bottom) tuple to crop before OCR. Coordinates in
          returned results are adjusted back to original image space.
    """
    img = Image.open(image_path)
    crop_offset_x, crop_offset_y = 0, 0
    if crop:
        left, top, right, bottom = crop
        img = img.crop((left, top, right, bottom))
        crop_offset_x, crop_offset_y = left, top
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    results = []
    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if not text or conf < min_conf:
            continue

        x = data['left'][i] + crop_offset_x
        y = data['top'][i] + crop_offset_y
        w = data['width'][i]
        h = data['height'][i]

        # Apply scale factor (for HiDPI → CSS pixel conversion)
        if scale != 1.0:
            x = round(x / scale)
            y = round(y / scale)
            w = round(w / scale)
            h = round(h / scale)

        cx = x + w // 2
        cy = y + h // 2

        results.append({
            'text': text,
            'x': x, 'y': y, 'w': w, 'h': h,
            'center_x': cx, 'center_y': cy,
            'conf': conf,
        })

    return results


def merge_words_to_lines(results, y_tolerance=5, x_gap_max=30):
    """Merge individual words into lines based on vertical proximity and horizontal gap."""
    if not results:
        return []

    # Sort by y then x
    sorted_results = sorted(results, key=lambda r: (r['y'], r['x']))

    lines = []
    current_line = [sorted_results[0]]

    for item in sorted_results[1:]:
        prev = current_line[-1]
        # Same line if y-centers are close and x gap is small
        prev_cy = prev['y'] + prev['h'] // 2
        item_cy = item['y'] + item['h'] // 2
        prev_right = prev['x'] + prev['w']

        if abs(prev_cy - item_cy) <= y_tolerance and (item['x'] - prev_right) <= x_gap_max:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]

    lines.append(current_line)

    # Merge each line into a single entry
    merged = []
    for line_words in lines:
        text = ' '.join(w['text'] for w in line_words)
        x = min(w['x'] for w in line_words)
        y = min(w['y'] for w in line_words)
        x2 = max(w['x'] + w['w'] for w in line_words)
        y2 = max(w['y'] + w['h'] for w in line_words)
        avg_conf = sum(w['conf'] for w in line_words) // len(line_words)
        merged.append({
            'text': text,
            'x': x, 'y': y, 'w': x2 - x, 'h': y2 - y,
            'center_x': (x + x2) // 2, 'center_y': (y + y2) // 2,
            'conf': avg_conf,
        })

    return merged


def capture_window(window_name=None, output_path='/tmp/ocr_capture.png'):
    """Capture a window screenshot using ImageMagick import.

    If window_name is given, finds window by name using xdotool.
    Otherwise captures the active/focused window.
    Returns the output path.
    """
    env = {**os.environ, 'DISPLAY': ':0'}
    if window_name:
        result = subprocess.run(
            ['xdotool', 'search', '--name', window_name],
            capture_output=True, text=True, env=env)
        wids = result.stdout.strip().split('\n')
        if not wids or not wids[0]:
            print(f"Error: no window matching '{window_name}'", file=sys.stderr)
            sys.exit(1)
        wid = wids[0]
    else:
        result = subprocess.run(
            ['xdotool', 'getactivewindow'],
            capture_output=True, text=True, env=env)
        wid = result.stdout.strip()

    subprocess.run(
        ['import', '-window', wid, output_path],
        env=env, check=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='OCR with bounding boxes')
    parser.add_argument('image', nargs='?', default=None,
                        help='Path to screenshot image (omit to capture active window)')
    parser.add_argument('--min-conf', type=int, default=40,
                        help='Minimum confidence threshold (0-100, default: 40)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor to divide coordinates by (e.g. 1.58 for HiDPI)')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--filter', type=str, default=None,
                        help='Filter results containing this text (case-insensitive)')
    parser.add_argument('--crop', type=str, default=None,
                        help='Crop region: left,top,right,bottom (pixel coordinates)')
    parser.add_argument('--capture', type=str, default=None, metavar='WINDOW_NAME',
                        help='Capture window by name (uses xdotool + import)')
    parser.add_argument('--capture-active', action='store_true',
                        help='Capture the currently active window')
    parser.add_argument('--chrome-mcp', action='store_true',
                        help='Apply Chrome MCP viewport transform (scale=1.578, y-offset=78)')
    parser.add_argument('--y-offset', type=int, default=0,
                        help='Subtract this from all y coordinates (for viewport offset)')
    parser.add_argument('--words', action='store_true',
                        help='Show individual words instead of merged lines')
    args = parser.parse_args()

    # Capture mode: take screenshot of a window
    if args.capture or args.capture_active:
        image_path = capture_window(
            window_name=args.capture if args.capture else None)
        print(f"Captured: {image_path}", file=sys.stderr)
    elif args.image:
        image_path = args.image
    else:
        parser.error("Provide an image path, --capture WINDOW_NAME, or --capture-active")

    if not Path(image_path).exists():
        print(f"Error: {image_path} not found", file=sys.stderr)
        sys.exit(1)

    # Chrome MCP preset
    if args.chrome_mcp:
        if args.scale == 1.0:
            args.scale = 1.578
        if args.y_offset == 0:
            args.y_offset = 78

    crop = None
    if args.crop:
        parts = [int(x) for x in args.crop.split(',')]
        if len(parts) != 4:
            print("Error: --crop needs 4 values: left,top,right,bottom", file=sys.stderr)
            sys.exit(1)
        crop = tuple(parts)

    results = ocr_with_boxes(image_path, args.min_conf, args.scale, crop)

    # Apply y-offset
    if args.y_offset:
        for r in results:
            r['y'] -= args.y_offset
            r['center_y'] -= args.y_offset

    if not args.words:
        results = merge_words_to_lines(results)

    if args.filter:
        filt = args.filter.lower()
        results = [r for r in results if filt in r['text'].lower()]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print("No text detected.")
            return
        # Table format
        print(f"{'Text':<50} {'Center':>12} {'Box (x,y,w,h)':>20} {'Conf':>5}")
        print('-' * 92)
        for r in results:
            text = r['text'][:48]
            center = f"({r['center_x']},{r['center_y']})"
            box = f"({r['x']},{r['y']},{r['w']},{r['h']})"
            print(f"{text:<50} {center:>12} {box:>20} {r['conf']:>5}")


if __name__ == '__main__':
    main()
