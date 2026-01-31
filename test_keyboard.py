#!/usr/bin/env python
"""
Debug script to discover what key names the keyboard library reports.
Press keys and watch the output.
"""
import keyboard
import time

print("Keyboard debug mode: Press keys to see their names")
print("Press ESC to exit\n")

try:
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            print(f"KEY DOWN: '{event.name}'")
        if event.name == 'esc':
            print("Exiting...")
            break
except KeyboardInterrupt:
    pass
