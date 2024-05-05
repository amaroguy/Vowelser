import time 
import vgamepad as vg

gamepad = vg.VX360Gamepad()
gamepad.reset()
gamepad.update()

A_BUTTON = vg.XUSB_BUTTON.XUSB_GAMEPAD_A
B_BUTTON = vg.XUSB_BUTTON.XUSB_GAMEPAD_B
X_BUTTON = vg.XUSB_BUTTON.XUSB_GAMEPAD_X
SLEEP_TIME = 3

def press_button(button):
    
    gamepad.press_button(button=button)
    gamepad.update()
    
    time.sleep(1)

    gamepad.release_button(button=button)
    gamepad.reset()
    gamepad.update()

def move_stick(x_dir, y_dir):

    gamepad.left_joystick_float(x_value_float=x_dir, y_value_float=y_dir)
    gamepad.update()

    time.sleep(1)

    gamepad.reset()
    gamepad.update()





print(f"Starting Controller Syncing in {SLEEP_TIME}s")
time.sleep(SLEEP_TIME)

print("Pressing A")
press_button(A_BUTTON)
print("Pressed A")

print(f"Pressing B in {SLEEP_TIME}s")
time.sleep(SLEEP_TIME)
press_button(B_BUTTON)
print("Pressed B")

print(f"Pressing X in {SLEEP_TIME}s")
time.sleep(SLEEP_TIME)
press_button(X_BUTTON)
print("Done!")

print(f"Pressing up on the control stick in {SLEEP_TIME}s")
time.sleep(SLEEP_TIME)
move_stick(x_dir=0.0, y_dir=1.0)

print(f"Pressing down on the control stick in {SLEEP_TIME}s")
time.sleep(SLEEP_TIME)
move_stick(x_dir=0.0, y_dir=-1.0)

print(f"Pressing left on the control stick in {SLEEP_TIME}s")
time.sleep(SLEEP_TIME)
move_stick(x_dir=-1.0, y_dir=0.0)

print(f"Pressing right on the control stick in {SLEEP_TIME}s")
time.sleep(SLEEP_TIME)
move_stick(x_dir=1.0, y_dir=0.0)




