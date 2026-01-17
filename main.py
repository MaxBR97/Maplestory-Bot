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

ACTIONS = ['left', 'right', 'up', 'down', 'alt', 'ctrl','idle', 'space', 'Home', 'End']
CHARACTER_TEMPLATES_DIR = Path("objects/My_Character")
CHARACTER_CLIMBING_TEMPLATES_DIR = Path("objects/My_Character/Climbing") # New
MONSTER_TEMPLATES_DIR = Path("Objects/Monsters")
CLIMBING_TEMPLATES_DIR = Path("Objects/Climbing")

MONSTER_MATCH_THRESHOLD = 0.65
CHARACTER_MATCH_THRESHOLD = 0.5
ATTACK_RANGE_X = 250
ATTACK_RANGE_Y = 80 # Vertical distance to engage monsters
ATTACK_KEYS = {'ctrl', 'space'} # Define all possible attack keys here
WINDOW_NAME = "MapleStory Detections"
NOTICE_TEMPLATES_DIR = Path("Objects/Notice")
NOTICE_MATCH_THRESHOLD = 0.4
EASY_OCR_READER = easyocr.Reader(["en"], gpu=False)
BAR_TEMPLATES_DIR = Path("Objects/Bars")
BAR_MATCH_THRESHOLD = 0.5

BAR_PATTERNS = {
    "Level": re.compile(r"\b(\d{1,3})\b"),
    "HP": re.compile(r"\[(\d+/\d+)\]"),
    "MP": re.compile(r"\[(\d+/\d+)\]"),
    "EXP": re.compile(r"(\d+(?:\.\d+)?)"),
}

def load_player_templates(base_dir, climbing_dir):
    """Loads all player templates and tags them as climbing or not."""
    templates = []
    # Load normal templates from the base directory
    for entry in sorted(base_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
            if tpl is not None:
                # Tag: is_climbing = False
                templates.append((tpl, tpl.shape[::-1], False))
    
    # Load climbing templates recursively from the climbing subdirectory
    for entry in sorted(climbing_dir.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
            if tpl is not None:
                # Tag: is_climbing = True
                templates.append((tpl, tpl.shape[::-1], True))
    return templates

def load_templates(directory):
    templates = []
    for entry in sorted(directory.iterdir()):
        if entry.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        templates.append((tpl, tpl.shape[::-1]))
    return templates

def load_templates_recursive(directory):
    """Recursively loads all image templates from a directory and its subdirectories."""
    templates = []
    for entry in sorted(directory.rglob("*")):
        if entry.is_file() and entry.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
            if tpl is None:
                continue
            templates.append((tpl, tpl.shape[::-1]))
    return templates

PLAYER_TEMPLATES = load_player_templates(CHARACTER_TEMPLATES_DIR, CHARACTER_CLIMBING_TEMPLATES_DIR)
NOTICE_TEMPLATES = load_templates(NOTICE_TEMPLATES_DIR)
BAR_TEMPLATES = {
    category.name: load_templates(category)
    for category in sorted(BAR_TEMPLATES_DIR.iterdir())
    if category.is_dir()
}
MONSTER_TEMPLATES = load_templates_recursive(MONSTER_TEMPLATES_DIR)
CLIMBING_TEMPLATES = load_templates_recursive(CLIMBING_TEMPLATES_DIR) # New

def check_hp_status(sct):
    """Checks a single pixel to see if HP is low (grey) or high (red)."""
    # Define the single pixel region to check
    hp_pixel_region = {"top": 615, "left": 270, "width": 1, "height": 1}
    
    # Grab the pixel
    img = np.array(sct.grab(hp_pixel_region))
    
    # Convert from BGRA to BGR, then to HSV
    bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv_pixel = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)[0, 0]
    
    hue, saturation, value = hsv_pixel
    
    # Grey colors have very low saturation.
    # If saturation is low, we assume the bar is depleted.
    is_grey = saturation < 30 and value > 50
    
    # As requested: returns True if grey (HP needed), False if red.
    return is_grey

def check_mp_status(sct):
    """Checks a single pixel to see if MP is low (grey) or high (blue)."""
    # Define the single pixel region to check
    mp_pixel_region = {"top": 615, "left": 350, "width": 1, "height": 1}
    
    # Grab the pixel
    img = np.array(sct.grab(mp_pixel_region))
    
    # Convert from BGRA to BGR, then to HSV
    bgr_img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv_pixel = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)[0, 0]
    
    hue, saturation, value = hsv_pixel
    
    # Grey colors have very low saturation.
    is_grey = saturation < 30 and value > 50
    
    # Returns True if grey (MP needed), False if blue.
    return is_grey

def get_player_state(sct):
    """
    Finds the player on screen, returning their coordinates and climbing status.
    It checks all player templates (normal and climbing) and finds the single best match.
    """
    monitor = sct.grab(SCREEN)
    img = np.array(monitor)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)

    best_score = CHARACTER_MATCH_THRESHOLD
    best_match = None # Will store (coords, is_climbing_state)

    for template, (w, h), is_climbing_template in PLAYER_TEMPLATES:
        if h > gray.shape[0] or w > gray.shape[1]:
            continue
        
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_score:
            best_score = max_val
            coords = (int(max_loc[0] + w / 2), int(max_loc[1] + h / 2))
            best_match = (coords, is_climbing_template)

    if best_match:
        print(best_match, " score: " ,best_score)
        return best_match # Returns (coords, is_climbing)
    else:
        return (None, False) # Player not found

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
        ys, xs = np.where(res >= MONSTER_MATCH_THRESHOLD)
        for y, x in zip(ys, xs):
            center = (int(x + w / 2), int(y + h / 2))
            if any(np.hypot(center[0] - m[0], center[1] - m[1]) < 20 for m in monsters):
                continue
            monsters.append(center)

    return monsters

def detect_climbing_objects(sct):
    """Detects ropes and ladders on the screen."""
    monitor = sct.grab(SCREEN)
    img = np.array(monitor)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)

    climbing_objects = []
    for template, (w, h) in CLIMBING_TEMPLATES:
        if h > gray.shape[0] or w > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= 0.7) # Using a slightly higher threshold for static objects
        for y, x in zip(ys, xs):
            center = (int(x + w / 2), int(y + h / 2))
            # De-duplicate
            if not any(np.hypot(center[0] - c[0], center[1] - c[1]) < 20 for c in climbing_objects):
                climbing_objects.append(center)
    return climbing_objects

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

def decide_action(player_coords, monsters, climbing_objects, is_climbing, need_HP, need_MP):
    """Decides the next set of actions based on game state, including climbing."""
    if need_HP:
        return ['Home']
    if need_MP:
        return ['End']

    if not player_coords or not monsters:
        return ['idle']

    player_x, player_y = player_coords
    target_monster = min(monsters, key=lambda m: np.hypot(m[0] - player_x, m[1] - player_y))
    monster_x, monster_y = target_monster

    # --- NEW: Climbing State Logic ---
    if is_climbing:
        if monster_y < player_y:
            return ['up'] # Monster is above, keep climbing up
        else:
            return ['down'] # Monster is below, climb down

    # --- Existing Pathfinding Logic ---
    actions = []
    #if random.uniform(0,1) > 0.5:
    #    actions.append('z')
    # Is the target monster on the same platform?
    if abs(monster_y - player_y) < ATTACK_RANGE_Y:
        # --- SAME PLATFORM LOGIC (Existing Logic) ---
        if abs(monster_x - player_x) < ATTACK_RANGE_X:
            # In attack range
            direction = 'left' if monster_x < player_x else 'right'
            actions.extend([direction, 'space'])
        else:
            # Out of attack range, move towards it
            direction = 'left' if monster_x < player_x else 'right'
            actions.append(direction)
    else:
        # --- DIFFERENT PLATFORM LOGIC (New Climbing Logic) ---
        # The monster is above or below. We need to find a rope/ladder.
        if not climbing_objects:
            return ['idle'] # No way to get there

        # Find the closest climbing object to the player
        closest_climb = min(climbing_objects, key=lambda c: np.hypot(c[0] - player_x, c[1] - player_y))
        climb_x, climb_y = closest_climb

        # 1. Align horizontally with the climbing object
        if abs(player_x - climb_x) > (random.random() * 40 + 6): # Not aligned with the rope/ladder
            # Move towards the rope/ladder
            direction = 'left' if climb_x < player_x else 'right'
            actions.append(direction)
        else:
            # 2. We are at the rope/ladder, now climb
            if monster_y < player_y:
                # Monster is above, climb up
                actions.extend(['alt', 'up'])
            else:
                # Monster is below, climb down
                actions.append('alt')
                actions.append('down')
    
    return actions if actions else ['idle']

def perform_action(next_actions, current_actions):
    """
    Manages continuous key presses efficiently. Re-presses attack keys to ensure continuous attacks.
    """
    # --- Wandering Logic ---
    # If the bot decides to be idle, there's a 40% chance it will move or jump down.
    if next_actions == ['idle'] and random.random() < 0.2:
        # Define the possible random actions
        possible_wandering_actions = [
            ['left'], 
            ['right'], 
            ['down', 'alt']
        ]
        # Randomly choose one of the defined actions
        next_actions = random.choice(possible_wandering_actions)
 
    # --- Handle Single-Press Actions (Potions) ---
    if 'Home' in next_actions or 'End' in next_actions:
        for key in current_actions:
            if key != 'idle':
                pydirectinput.keyUp(key)
        for key in next_actions:
            if key in ['Home', 'End']:
                pydirectinput.press(key)
        return ['idle']

    # --- Handle Continuous Actions (Movement/Attack/Climbing) ---
    next_actions_set = set(next_actions)
    current_actions_set = set(current_actions)

    # 1. Release keys that are no longer needed.
    keys_to_release = current_actions_set - next_actions_set
    for key in keys_to_release:
        if key != 'idle':
            pydirectinput.keyUp(key)
    
    # 2. Handle pressing keys, with special logic for attacks.
    for key in next_actions:
        # If the key is an attack key, always perform a quick press.
        # This ensures the attack is re-triggered on every loop.
        if key in ATTACK_KEYS or key == 'alt':
            pydirectinput.press(key)
        # If it's a movement/climbing key and it's NOT already held down, press it.
        elif key not in current_actions_set and key != 'idle':
            pydirectinput.keyDown(key)
    
    return next_actions

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

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.moveWindow(WINDOW_NAME, 800, 0)

with mss() as sct:
    try:
        current_actions = ['idle']
        while True:
            # Single call to get both player coordinates and climbing state
            coords, is_climbing = get_player_state(sct)
            
            monsters = detect_monsters(sct)
            climbs = detect_climbing_objects(sct)
            notices = []#detect_notices(sct)
            bars = {}#detect_bars(sct)
            bars_summary = {cat: items[0]["text"] for cat, items in bars.items() if items}
            captured = np.array(sct.grab(SCREEN))
            display_frame = cv2.cvtColor(captured, cv2.COLOR_BGRA2BGR)

            # Check HP and MP status
            need_hp = check_hp_status(sct)
            need_mp = check_mp_status(sct)

            # Decide and perform the next action
            next_actions = decide_action(coords, monsters, climbs, is_climbing, need_hp, need_mp)
            current_actions = perform_action(next_actions, current_actions)

            coord_str = f"{'Climbing ' if is_climbing else ''}({coords[0]}, {coords[1]})" if coords else "Not Found"
            print(
                f"Time: {time.strftime('%H:%M:%S')} | Char: {coord_str} | "
                f"Monsters: {len(monsters)} | Notices: {notices} | Bars: {bars_summary} | Action: {current_actions}"
            )

            annotated = annotate_frame(display_frame, coords, monsters, bars)

            window_rect = cv2.getWindowImageRect(WINDOW_NAME)
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(random.uniform(0.04, 0.25))

    except KeyboardInterrupt:
        print("\nStopping Agent.")
    finally:
        # Ensure all keys are released on exit
        for key in ACTIONS:
            if key != 'idle':
                pydirectinput.keyUp(key)
        cv2.destroyAllWindows()