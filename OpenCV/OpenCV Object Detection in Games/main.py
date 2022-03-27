import cv2
import time
import curses
import win32com.client
from windowcapture import WindowCapture
from vision import Vision, VisionBonus
from hsvfilter import HsvFilter
from enemy import Enemy, Bonus

# Typer
wsh = win32com.client.Dispatch("WScript.Shell")
wsh.AppActivate("Typer Shark Deluxe 1.02")

# Screen
stdscr = curses.initscr()

# WindowCapture.list_window_names()
# Vision.init_control_gui()

shark = Enemy('Shark', 'src/shark.jpg', 47, 127, 8, 28)
piranha = Enemy('Piranha', 'src/piranha.jpg', 15, 31, 17, 35)
bonus = Bonus('Bonus', 245, 285, 150, 35)
# Needles
n_shark = Vision(enemy = shark)
n_piranha = Vision(enemy = piranha)
n_bonus = VisionBonus(bonus = bonus)

filter_sea = HsvFilter(h_min = 115, h_max = 116)

wincap = WindowCapture('Typer Shark Deluxe 1.02')

while (True):
    loop_time = time.time()
    screenshot = wincap.capture()
    
    
    haystack = Vision.applyHsvFilter(screenshot.copy(), filter_sea, mode = 'out')
    haystack = cv2.cvtColor(haystack, cv2.COLOR_BGR2GRAY)

    rect_shark = n_shark.find(haystack, threshold = 0.6, max_results = 10)
    rect_piranha = n_piranha.find(haystack, threshold = 0.75, max_results = 10)

    # OCR
    text_shark = n_shark.ocr(haystack, rect_shark)
    text_piranha = n_piranha.ocr(haystack, rect_piranha)
    text_bonus = n_bonus.ocr(haystack)
    
    
    stdscr.erase()
    stdscr.addstr(0, 0, f'FPS: {1/(time.time() - loop_time):.2f}')
    stdscr.addstr(1, 0, 'Shark: ')
    stdscr.addstr(2, 0, 'Piranha: ')
    stdscr.addstr(3, 0, 'Bonus: ')
    for i, text in enumerate([text_shark, text_piranha, text_bonus]):
        if text:
            stdscr.addstr((i + 1), 9, f'{text}')
            for key in text:
                wsh.SendKeys(key.lower())
                time.sleep(0.03)
    
    display = Vision.draw_rectangles(screenshot, rect_shark)
    display = Vision.draw_rectangles(screenshot, rect_piranha)
    stdscr.refresh()
    cv2.imshow('Result', display)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break