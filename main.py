import cv2
import numpy as np
import time
import random
import pydirectinput
from mss import mss
import os
from pathlib import Path
import pytesseract
import easyocr

# --- CONFIGURATION ---
# Adjust these to match your Mini-map location
#MAP_REGION = {"top": 40, "left": 15, "width": 150, "height":150 }
SCREEN = {"top": 0, "left": 0, "width": 800, "height": 640}

ACTIONS = ['left', 'right', 'up', 'down', 'alt', 'ctrl','idle']
GREEN_SNAIL_DIR = Path("objects/monsters/green_snail")
CHARACTER_TEMPLATES_DIR = Path("objects/My_Character")
MATCH_THRESHOLD = 0.5
CHARACTER_MATCH_THRESHOLD = 0.5
WINDOW_NAME = "MapleStory Detections"
NOTICE_TEMPLATES_DIR = Path("Objects/Notice")
NOTICE_MATCH_THRESHOLD = 0.5
EASY_OCR_READER = easyocr.Reader(["en"], gpu=False)
BAR_TEMPLATES_DIR = Path("Objects/Bars")
BAR_MATCH_THRESHOLD = 0.5

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

GREEN_SNAIL_TEMPLATES = load_templates(GREEN_SNAIL_DIR)
PLAYER_TEMPLATES = load_templates(CHARACTER_TEMPLATES_DIR)
NOTICE_TEMPLATES = load_templates(NOTICE_TEMPLATES_DIR)
BAR_TEMPLATES = {
    category.name: load_templates(category)
    for category in sorted(BAR_TEMPLATES_DIR.iterdir())
    if category.is_dir()
}

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
    for template, (w, h) in GREEN_SNAIL_TEMPLATES:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= MATCH_THRESHOLD)
        for y, x in zip(ys, xs):
            center = (int(x + w / 2), int(y + h / 2))
            if any(np.hypot(center[0] - m[0], center[1] - m[1]) < 20 for m in monsters):
                continue
            monsters.append(center)

    return monsters

def detect_notices(sct):
    monitor = sct.grab(SCREEN)
    img = np.array(monitor)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    notices = []
    for template, (w, h) in NOTICE_TEMPLATES:
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= NOTICE_MATCH_THRESHOLD)
        for y, x in zip(ys, xs):
            snippet = gray[y : y + h, x : x + w]
            parsed = EASY_OCR_READER.readtext(snippet, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()+ ")
            text = " ".join(item.strip() for item in parsed if item.strip())
            if text:
                notices.append(text)
    return notices

def detect_bars(sct):
    monitor = sct.grab(SCREEN)
    img = np.array(monitor)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bars = {}
    for category, templates in BAR_TEMPLATES.items():
        parsed_texts = []
        for template, (w, h) in templates:
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= BAR_MATCH_THRESHOLD)
            for y, x in zip(ys, xs):
                snippet = gray[y : y + h, x : x + w]
                parsed = EASY_OCR_READER.readtext(
                    snippet, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./% "
                )
                text = " ".join(item.strip() for item in parsed if item.strip())
                if text:
                    parsed_texts.append(text)
        if parsed_texts:
            bars[category] = parsed_texts
    return bars

def perform_random_action():
    #action = random.choice(ACTIONS)
    action = 'idle' 
    if action != "idle":
        pydirectinput.press(action)
    
    return action

def annotate_frame(frame, player_coord, monsters):
    annotated = frame.copy()
    if player_coord:
        cv2.circle(annotated, player_coord, 12, (255, 0, 0), 2)
    for m in monsters:
        cv2.drawMarker(annotated, m, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
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
            notices =  []#detect_notices(sct)
            bars = detect_bars(sct)
            captured = np.array(sct.grab(SCREEN))
            display_frame = cv2.cvtColor(captured, cv2.COLOR_BGRA2BGR)

            last_action = perform_random_action()
            coord_str = f"({coords[0]}, {coords[1]})" if coords else "Not Found"
            print(
                f"Time: {time.strftime('%H:%M:%S')} | Char: {coord_str} | "
                f"Monsters: {monsters} | Notices: {notices} | Bars: {bars} | Action: {last_action}"
            )

            annotated = annotate_frame(display_frame, coords, monsters)

            window_rect = cv2.getWindowImageRect(WINDOW_NAME)
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(random.uniform(0.33, 1.12))

    except KeyboardInterrupt:
        print("\nStopping Agent.")
    finally:
        cv2.destroyAllWindows()