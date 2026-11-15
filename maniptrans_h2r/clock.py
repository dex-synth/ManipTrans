"""The method's own clocks, in seconds.

Training keeps the native 60 Hz control over 120 Hz physics in a private model; the task's
nominal step is never inherited. Deployment keeps the control period, while the evaluator owns
the evaluation physics step and announces it in hello["dt_s"].
"""

CONTROL_DT_S = 1/60
PHYSICS_DT_S = 1/120
DECIMATION = round(CONTROL_DT_S/PHYSICS_DT_S)  # training physics steps per control period


def training_ticks(seconds):
    """Whole training physics steps in a duration; anything else is rejected, never rounded."""
    ticks = round(seconds/PHYSICS_DT_S)
    if ticks < 1 or abs(ticks*PHYSICS_DT_S-seconds) > 1e-9:
        raise ValueError(f"{seconds:g} s is not a whole number of {PHYSICS_DT_S:g} s training physics steps")
    return ticks
