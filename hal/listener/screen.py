"""Whether the Mac's screen is locked. Standard library only (ctypes).

macOS describes the login session in CGSessionCopyCurrentDictionary():
"CGSSessionScreenIsLocked" is true while the screen is locked (the key is
absent otherwise), and "kCGSSessionOnConsoleKey" is false while another user
has the screen (fast user switching). Either means nobody is at this keyboard.
Anywhere else, or if the call fails, the answer is None: not known.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

_UTF8 = 0x08000100
_lib: tuple | None = None


def _libs():
    global _lib
    if _lib is None:
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics") or ctypes.util.find_library("ApplicationServices"))
        cg.CGSessionCopyCurrentDictionary.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFDictionaryGetValue.restype = ctypes.c_void_p
        cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        cf.CFGetTypeID.restype = ctypes.c_ulong
        cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
        cf.CFBooleanGetTypeID.restype = ctypes.c_ulong
        cf.CFBooleanGetValue.restype = ctypes.c_bool
        cf.CFBooleanGetValue.argtypes = [ctypes.c_void_p]
        _lib = (cf, cg)
    return _lib


def _flag(cf, session, name: bytes) -> bool | None:
    key = cf.CFStringCreateWithCString(None, name, _UTF8)
    if not key:
        return None
    try:
        value = cf.CFDictionaryGetValue(session, key)
        if not value or cf.CFGetTypeID(value) != cf.CFBooleanGetTypeID():
            return None
        return bool(cf.CFBooleanGetValue(value))
    finally:
        cf.CFRelease(key)


def locked() -> bool | None:
    """True if the screen is locked or another user has it; None if unknown."""
    if sys.platform != "darwin":
        return None
    try:
        cf, cg = _libs()
        session = cg.CGSessionCopyCurrentDictionary()
        if not session:       # no window-server session (over ssh, say)
            return None
        try:
            if _flag(cf, session, b"CGSSessionScreenIsLocked"):
                return True
            return _flag(cf, session, b"kCGSSessionOnConsoleKey") is False
        finally:
            cf.CFRelease(session)
    except (OSError, AttributeError, TypeError):
        return None
