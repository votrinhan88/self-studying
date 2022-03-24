import cv2
import numpy as np
import win32gui, win32ui, win32con

class WindowCapture:
    # Properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f'Window {window_name} not found.')

        # Get window size
        windowRect = win32gui.GetWindowRect(self.hwnd)
        self.w = windowRect[2] - windowRect[0]
        self.h = windowRect[3] - windowRect[1]
        # Crop border (measured with Paint)
        border_pixel = 3
        titlebar_pixel = 26
        self.w = self.w - border_pixel * 2
        self.h = self.h - titlebar_pixel - border_pixel 
        self.cropped_x = border_pixel
        self.cropped_y = titlebar_pixel
        # Set cropped coordiantes offset so we can translate screenshot images into actual screen positions
        self.offset_x = windowRect[0] + self.cropped_x
        self.offset_y = windowRect[1] + self.cropped_y

    def capture(self):
        # bmpfilenamename = "window_capture.bmp"
        # Thanks StackOverflow
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0,0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)
        
        # Do not save .bmp
        # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # Optimize (drop alpha channel, feed to cv2)
        img = np.fromstring(signedIntsArray, dtype = 'uint8')
        img.shape = (self.h, self.w, 4)
        img = img[:, :, 0:3]
        img = np.ascontiguousarray(img)
        return img

    def get_screen_position(self, pos):
        # Do not move window around
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)

    def list_window_names(self):
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)