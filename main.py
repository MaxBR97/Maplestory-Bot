import os

import cv2
import numpy as np
import time
import random
import pydirectinput
from mss import mss
from pathlib import Path
import easyocr
import re

# --- CONFIGURATION ---
# Adjust these to match your Mini-map location
SCREEN = {"top": 0, "left": 0, "width": 800, "height": 625}
BAR_REGION = {"top": 590, "left": 0, "width": 560, "height": 35}
NOTICE_REGION = {"top": 370, "left": 500, "width": 300, "height": 110}

ACTIONS = ['left', 'right', 'up', 'down', 'alt', 'ctrl','idle']
CHARACTER_TEMPLATES_DIR = Path("objects/My_Character")
MONSTER_TEMPLATES_DIR = Path("Objects/Monsters")

MATCH_THRESHOLD = 0.5
CHARACTER_MATCH_THRESHOLD = 0.5
WINDOW_NAME = "MapleStory Detections"
NOTICE_TEMPLATES_DIR = Path("Objects/Notice")
NOTICE_MATCH_THRESHOLD = 0.35
EASY_OCR_READER = easyocr.Reader(["en"], gpu=False)
BAR_TEMPLATES_DIR = Path("Objects/Bars")
BAR_MATCH_THRESHOLD = 0.5

BAR_PATTERNS = {
    "Level": re.compile(r"\b(\d{1,3})\b"),
    "HP": re.compile(r"\[(\d+/\d+)\]"),
    "MP": re.compile(r"\[(\d+/\d+)\]"),
    "EXP": re.compile(r"(\d+(?:\.\d+)?)"),
}

def load_templates(directory):
    templates = []
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        templates.append((tpl, tpl.shape[::-1]))  # (template, (width,height))
    return templates

def load_templates_recursive(directory):
    templates = []
    for entry in sorted(directory.rglob("*")):
        if not entry.is_file() or entry.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        templates.append((tpl, tpl.shape[::-1]))
    return templates

PLAYER_TEMPLATES = load_templates(CHARACTER_TEMPLATES_DIR)
NOTICE_TEMPLATES = load_templates(NOTICE_TEMPLATES_DIR)
BAR_TEMPLATES = {
    category.name: load_templates(category)
    for category in sorted(BAR_TEMPLATES_DIR.iterdir())
    if category.is_dir()
}
MONSTER_TEMPLATES = load_templates_recursive(MONSTER_TEMPLATES_DIR)

def get_player_coords(sct):
    monitor = sct.grab(SCREEN)
    img = np.array(monitor)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)

    best_center = None
    best_score = CHARACTER_MATCH_THRESHOLD
    for template, (w, h) in PLAYER_TEMPLATES:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_center = (int(max_loc[0] + w / 2), int(max_loc[1] + h / 2))

    return best_center

def detect_monsters(sct):
    monitor = sct.grab(SCREEN)
    img = np.array(monitor)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    monsters = []
    for template, (w, h) in MONSTER_TEMPLATES:
        if h > gray.shape[0] or w > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= MATCH_THRESHOLD)
        for y, x in zip(ys, xs):
            center = (int(x + w / 2), int(y + h / 2))
            if any(np.hypot(center[0] - m[0], center[1] - m[1]) < 20 for m in monsters):
                continue
            monsters.append(center)

    return monsters

def preprocess_for_ocr(snippet):
    if snippet.size == 0:
        return snippet
    scaled = cv2.resize(snippet, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(scaled)
    denoised = cv2.medianBlur(equalized, 3)
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        7,
    )
    return thresh

def preprocess_notice_snippet(gray_snip, color_snip):
    hsv = cv2.cvtColor(color_snip, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 60, 255]))
    focused = cv2.bitwise_and(gray_snip, white_mask)
    return preprocess_for_ocr(focused)

def detect_notices(sct):
    monitor = sct.grab(NOTICE_REGION)
    img = np.array(monitor)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    matches = []
    for template, (w, h) in NOTICE_TEMPLATES:
        if h > gray.shape[0] or w > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= NOTICE_MATCH_THRESHOLD)
        for y, x in zip(ys, xs):
            snippet_gray = gray[y : y + h, x : x + w]
            snippet_color = img_bgr[y : y + h, x : x + w]
            ocr_input = preprocess_notice_snippet(snippet_gray, snippet_color)

            preview = cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR)
            preview = cv2.resize(preview, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Notice Preview", preview)

            parsed = EASY_OCR_READER.readtext(
                ocr_input, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()+ "
            )
            text = " ".join(item.strip() for item in parsed if item.strip())
            if text:
                matches.append(text)
    return matches

def normalize_bar_text(category, candidate):
    pattern = BAR_PATTERNS.get(category)
    if not pattern:
        return None
    match = pattern.search(candidate)
    return match.group(1) if match else None

def preprocess_bar_snippet(gray_snip, color_snip):
    hsv = cv2.cvtColor(color_snip, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 60, 255]))
    green_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([95, 255, 255]))
    mask = cv2.bitwise_or(white_mask, green_mask)
    focused = cv2.bitwise_and(gray_snip, mask)
    return preprocess_for_ocr(focused)

def detect_bars(sct):
    monitor = sct.grab(BAR_REGION)
    img = np.array(monitor)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bars = {}
    for category, templates in BAR_TEMPLATES.items():
        parsed_texts = []
        seen_centers = []
        found = False
        for template, (w, h) in templates:
            if found or h > gray.shape[0] or w > gray.shape[1]:
                continue
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= BAR_MATCH_THRESHOLD)
            for y, x in zip(ys, xs):
                snippet_gray = gray[y : y + h, x : x + w]
                snippet_color = img_bgr[y : y + h, x : x + w]
                ocr_input = preprocess_bar_snippet(snippet_gray, snippet_color)

                preview = cv2.cvtColor(ocr_input, cv2.COLOR_GRAY2BGR)
                preview = cv2.resize(preview, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(f"Bar Preview - {category}", preview)

                parsed = EASY_OCR_READER.readtext(
                    ocr_input, detail=0, allowlist="0123456789.%[]/"
                )
                text = " ".join(item.strip() for item in parsed if item.strip())
                normalized = text #normalize_bar_text(category, text)
                if not normalized:
                    continue
                center = (x + w // 2, y + h // 2)
                if any(np.hypot(center[0] - c[0], center[1] - c[1]) < 8 for c in seen_centers):
                    continue
                seen_centers.append(center)
                parsed_texts.append({"text": normalized, "rect": (x, y, w, h)})
                found = True
                break
        if parsed_texts:
            bars[category] = parsed_texts
    return bars

def perform_random_action():
    #action = random.choice(ACTIONS)
    action = 'idle' 
    if action != "idle":
        pydirectinput.press(action)
    
    return action

def annotate_frame(frame, player_coord, monsters, bars):
    annotated = frame.copy()
    if player_coord:
        cv2.circle(annotated, player_coord, 12, (255, 0, 0), 2)
    for m in monsters:
        cv2.drawMarker(annotated, m, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
    for items in bars.values():
        for item in items:
            x, y, w, h = item["rect"]
            top_left = (x + BAR_REGION["left"], y + BAR_REGION["top"])
            bottom_right = (
                x + BAR_REGION["left"] + w,
                y + BAR_REGION["top"] + h,
            )
            cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 255), 2)
    return annotated

# --- MAIN LOOP ---
print("AI Agent Active. Focus the MapleStory window NOW!")
#time.sleep(2)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, 800, 0)

with mss() as sct:
    try:
        while True:
            coords = get_player_coords(sct)
            monsters = detect_monsters(sct)
            notices = detect_notices(sct)
            bars = detect_bars(sct)
            bars_summary = {cat: items[0]["text"] for cat, items in bars.items() if items}
            captured = np.array(sct.grab(SCREEN))
            display_frame = cv2.cvtColor(captured, cv2.COLOR_BGRA2BGR)

            last_action = perform_random_action()
            coord_str = f"({coords[0]}, {coords[1]})" if coords else "Not Found"
            print(
                f"Time: {time.strftime('%H:%M:%S')} | Char: {coord_str} | "
                f"Monsters: {monsters} | Notices: {notices} | Bars: {bars_summary} | Action: {last_action}"
            )

            annotated = annotate_frame(display_frame, coords, monsters, bars)

            window_rect = cv2.getWindowImageRect(WINDOW_NAME)
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(random.uniform(0.1, 0.35))

    except KeyboardInterrupt:
        print("\nStopping Agent.")
    finally:
        cv2.destroyAllWindows()