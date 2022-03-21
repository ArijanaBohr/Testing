#It'll be 4 for 32 bit or 8 for 64 bit.
import ctypes
print(ctypes.sizeof(ctypes.c_voidp))
