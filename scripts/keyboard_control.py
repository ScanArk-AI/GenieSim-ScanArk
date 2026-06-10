#!/usr/bin/env python3
# VLN Keyboard Controller - sends discrete actions via UDP
# Run in a second terminal inside the container:
#   python3 /geniesim/main/scripts/keyboard_control.py

import socket
import sys
import tty
import termios

KEY_MAP = {
    "w": "forward",
    "s": "backward",
    "a": "turn_left",
    "d": "turn_right",
    " ": "stop",
}

HELP_TEXT = """
=== VLN Keyboard Controller (Habitat-style) ===
  W - Forward  (0.25m)
  S - Backward (0.25m)
  A - Turn Left  (15 deg)
  D - Turn Right (15 deg)
  Space - Stop
  Ctrl+C - Exit
================================================
"""


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    print(HELP_TEXT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = ("127.0.0.1", 12346)
    print(f"Sending actions to {target[0]}:{target[1]} via UDP\n")

    try:
        while True:
            key = get_key()
            if key == "\x03":  # Ctrl+C
                break
            if key in KEY_MAP:
                action = KEY_MAP[key]
                sock.sendto(action.encode(), target)
                print(f"Action: {action}")
    finally:
        sock.close()
        print("\nExited.")


if __name__ == "__main__":
    main()
