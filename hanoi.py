#!/usr/bin/env python3

def solve_hanoi(disks, from_pin, to_pin, aux_pin):
    if disks == 1:
        print("Move disk from pin", from_pin, "to", to_pin)
        return

    solve_hanoi(disks - 1, from_pin, aux_pin, to_pin) # Move all disks except the last to the auxiliary pin
    print("Move disk from pin", from_pin, "to", to_pin) # Move the last disk to the target pin
    solve_hanoi(disks - 1, aux_pin, to_pin, from_pin) # Move all disks from the auxiliary pin to the target pin


solve_hanoi(10, 1, 3, 2)
