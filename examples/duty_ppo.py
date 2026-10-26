#!/usr/bin/env python3
"""PPO for duty-cycle control.

on_off switches 100% and TurnOff. duty_mode walks 100/75/50/25/0/TurnOff
by one level per step. The ASM component state machine is asm_ppo.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import asm_ppo


def main():
    asm_ppo.main(
        asm_ppo.parse_args(
            default_kind="duty_mode",
            choices=("on_off", "duty_mode", "duty", "both"),
        )
    )


if __name__ == "__main__":
    main()
