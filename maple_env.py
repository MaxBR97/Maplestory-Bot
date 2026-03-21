import cv2
import numpy as np
import mss
import pydirectinput
import time
import math
from pathlib import Path
import random

# --- CONFIGURATION ---
SCREEN = {"top": 0, "left": 0, "width": 800, "height": 625}
CHARACTER_TEMPLATES_DIR = Path("Objects/My_Character")
CHARACTER_CLIMBING_DIR = Path("Objects/My_Character/Climbing")
MONSTER_TEMPLATES_DIR = Path("Objects/Monsters")
DAMAGE_TEMPLATES_DIR = Path("Objects/Damage")

# Detection Thresholds
MONSTER_MATCH_THRESHOLD = 0.65
CHARACTER_MATCH_THRESHOLD = 0.60
DAMAGE_MATCH_THRESHOLD = 0.55
HP_POTION_KEY = 'Home'
MP_POTION_KEY = 'End'

# Max monsters to track in the state vector
MAX_MONSTERS = 1

# Action Mapping (Indices -> Actions)
# Simple strings are single keys. Tuples are sequences (key1, delay, key2).
ACTION_MAP = {
    #0: 'idle',
    1: 'left',
    2: 'right',
    #3: 'up',
    #4: 'down',
    #5: 'alt',             # Jump
    0: 'ctrl',            # Attack
    #7: ('down', 'alt'),   # Jump Down (Composite)
    #8: ('left', 'alt'),   # Jump Left (Composite)
    #9: ('right', 'alt'), # Jump Right (Composite)
}

class MapleStoryEnv:
    def __init__(self):
        self.sct = mss.mss()
        
        # Load Templates
        self.player_templates = self._load_player_templates(CHARACTER_TEMPLATES_DIR, CHARACTER_CLIMBING_DIR)
        self.monster_templates = self._load_templates_recursive(MONSTER_TEMPLATES_DIR)
        self.damage_templates = self._load_templates_recursive(DAMAGE_TEMPLATES_DIR)
        
        # State definitions
        # [PlayerX, PlayerY, IsClimbing, LowHP, LowMP] + [Mon1_dx, Mon1_dy, ..., MonN_dx, MonN_dy]
        self.base_state_dim = 5
        self.state_dim = self.base_state_dim + (MAX_MONSTERS * 2)
        self.action_dim = len(ACTION_MAP)
        
        # For Visualization & Logging
        self.last_frame = None
        self.last_player_pos = None
        self.last_monsters = []
        self.last_damage_count = 0
        self.last_bars = {}

    def _load_templates_recursive(self, directory):
        templates = []
        if not directory.exists(): return templates
        for entry in sorted(directory.rglob("*")):
            if entry.is_file() and entry.suffix.lower() in {".png", ".jpg"}:
                tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
                if tpl is not None:
                    templates.append((tpl, tpl.shape[::-1]))
        return templates

    def _load_player_templates(self, base_dir, climbing_dir):
        templates = []
        if base_dir.exists():
            for entry in sorted(base_dir.iterdir()):
                if entry.is_file() and entry.suffix.lower() in {".png", ".jpg"}:
                    tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
                    if tpl is not None: templates.append((tpl, tpl.shape[::-1], False))
        if climbing_dir.exists():
            for entry in sorted(climbing_dir.rglob("*")):
                if entry.is_file() and entry.suffix.lower() in {".png", ".jpg"}:
                    tpl = cv2.imread(str(entry), cv2.IMREAD_GRAYSCALE)
                    if tpl is not None: templates.append((tpl, tpl.shape[::-1], True))
        return templates

    def reset(self):
        pydirectinput.keyUp('left')
        pydirectinput.keyUp('right')
        pydirectinput.keyUp('down')
        state, _ = self._get_state()
        return state

    def step(self, action_idx):
        # 1. Execute Action
        action_def = ACTION_MAP.get(action_idx, 'idle')
        
        if isinstance(action_def, tuple):
            # Composite Action (e.g., Down + Alt)
            k1, k2 = action_def
            pydirectinput.keyDown(k1)
            time.sleep(0.05) # Short delay holding the direction
            pydirectinput.press(k2)
            time.sleep(0.05)
            pydirectinput.keyUp(k1)
            
        elif action_def in ['left', 'right', 'up', 'down']:
            pydirectinput.keyDown(action_def)
            time.sleep(0.15)
            pydirectinput.keyUp(action_def)
            
        elif action_def == 'idle':
            time.sleep(0.05)
            
        else:
            # Single press (Attack, Jump, Loot, Potions)
            pydirectinput.press(action_def)
            time.sleep(0.1)

        # 2. Observe State
        state, internal_data = self._get_state()
        damage_count = self._detect_damage()
        self.last_damage_count = damage_count # For rendering/logging

        # 3. Calculate Reward
        # +1 per damage detected, small time penalty
        reward = (damage_count * 1.0) - 0.05
        
        # Penalty for missing HP/MP when low
        if internal_data['low_hp'] and action_def != HP_POTION_KEY:
            reward -= 0.5
        if internal_data['low_mp'] and action_def != MP_POTION_KEY:
            reward -= 0.2

        done = False
        
        # Pack info for the logger
        info = {
            'player_pos': internal_data['player_pos'],
            'is_climbing': internal_data['is_climbing'],
            'monsters': internal_data['monsters'],
            'damage': damage_count,
            'action_name': str(action_def),
            'low_hp': internal_data['low_hp'],
            'low_mp': internal_data['low_mp']
        }
        
        return state, reward, done, info

    def _get_state(self):
        # Capture Screen
        monitor = self.sct.grab(SCREEN)
        img = np.array(monitor)
        self.last_frame = img.copy() # Save for rendering
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)

        # 1. Find Player
        player_pos = None
        is_climbing = False
        best_score = CHARACTER_MATCH_THRESHOLD
        
        for template, (w, h), climbing_flag in self.player_templates:
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                player_pos = (max_loc[0] + w // 2, max_loc[1] + h // 2)
                is_climbing = climbing_flag

        self.last_player_pos = player_pos
        self.last_climbing = is_climbing

        # 2. Find Monsters
        monsters = []
        for template, (w, h) in self.monster_templates:
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= MONSTER_MATCH_THRESHOLD)
            for y, x in zip(ys, xs):
                center = (int(x + w / 2), int(y + h / 2))
                # De-duplicate
                if not any(math.hypot(center[0]-m[0], center[1]-m[1]) < 20 for m in monsters):
                    monsters.append(center)
        
        self.last_monsters = monsters

        # 3. Process Monsters for State Vector
        # Sort by distance to player
        monster_features = []
        if player_pos:
            px, py = player_pos
            # Calculate distance for sorting
            monsters_with_dist = []
            for m in monsters:
                dist = math.hypot(m[0]-px, m[1]-py)
                monsters_with_dist.append((dist, m))
            
            monsters_with_dist.sort(key=lambda x: x[0])
            
            # Fill features for closest MAX_MONSTERS
            for i in range(MAX_MONSTERS):
                if i < len(monsters_with_dist):
                    m = monsters_with_dist[i][1]
                    # Normalize relative distance
                    dx = (m[0] - px) / 800.0
                    dy = (m[1] - py) / 600.0
                    monster_features.extend([dx, dy])
                else:
                    # Padding if fewer monsters than limit
                    monster_features.extend([1.0, 1.0]) # "Far away"
        else:
            # Player not found? Pad everything as far away
            monster_features = [1.0] * (MAX_MONSTERS * 2)

        # 4. Check HP/MP
        low_hp = self._check_pixel_grey(img, 615, 270)
        low_mp = self._check_pixel_grey(img, 615, 350)
        
        # Build Vector
        if player_pos:
            px_norm = player_pos[0] / 800.0
            py_norm = player_pos[1] / 600.0
        else:
            px_norm, py_norm = 0.5, 0.5

        state_vector = [
            px_norm, 
            py_norm, 
            1.0 if is_climbing else 0.0, 
            1.0 if low_hp else 0.0, 
            1.0 if low_mp else 0.0
        ] + monster_features

        internal_data = {
            'player_pos': player_pos,
            'is_climbing': is_climbing,
            'monsters': monsters,
            'low_hp': low_hp,
            'low_mp': low_mp
        }

        return np.array(state_vector, dtype=np.float32), internal_data

    def _detect_damage(self):
        monitor = self.sct.grab(SCREEN)
        img = np.array(monitor)
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2GRAY)
        count = 0
        for template, (w, h) in self.damage_templates:
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= DAMAGE_MATCH_THRESHOLD)
            count += len(ys)
        return min(count, 5)

    def _check_pixel_grey(self, img, y, x):
        if y >= img.shape[0] or x >= img.shape[1]: return False
        b, g, r, a = img[y, x]
        mx, mn = max(r, g, b), min(r, g, b)
        return (mx - mn < 15) and (mx > 50)

    def render(self):
        """Draws the detection panel."""
        if self.last_frame is None: return

        display = cv2.cvtColor(self.last_frame, cv2.COLOR_BGRA2BGR)
        
        # Draw Player
        if self.last_player_pos:
            color = (0, 0, 255) if self.last_climbing else (255, 0, 0) # Red if climbing, Blue normal
            cv2.circle(display, self.last_player_pos, 15, color, 2)
            cv2.putText(display, "Player", (self.last_player_pos[0]-20, self.last_player_pos[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw Monsters
        for m in self.last_monsters:
            cv2.drawMarker(display, m, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 20, 2)

        # Draw Damage Info
        cv2.putText(display, f"Damage Detected: {self.last_damage_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("MapleStory AI Detections", display)
        cv2.waitKey(1)