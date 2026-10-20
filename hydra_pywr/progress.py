"""
Progress reporting for HWI.

HWI (via the job queue) reads a job's stdout looking for lines starting `!!Progress x/y`, which
drive the progress bar, and `!!Output text`, which are shown to the user. Every progress update
here is paired with an output message describing it. The total is always `TOTAL` because HWI
takes the total from the first progress line it sees.

A run is split into stages, each ending at a fixed progress value; the model run itself spans
`RUN_START` to `RUN_END` and is scaled by the percentage of timesteps completed.
"""
from hydra_client.output import write_output, write_progress

TOTAL = 100

# Progress reached on completing each stage
EXPORT_DONE = 10
RUN_START = 10
RUN_END = 90
SAVE_DONE = 99
COMPLETE = 100


def report(progress, message):
    """ Write a progress update and the output message that goes with it. """
    write_progress(int(progress), TOTAL)
    write_output(message)


def report_run_percentage(percentage, message):
    """ Report `percentage` (0-100) of the model run, scaled into the run stage's range. """
    report(RUN_START + (RUN_END - RUN_START) * percentage / 100, message)


def hydra_progress_recorder_class():
    """
        A Pywr v1 `ProgressRecorder` that reports through `report_run_percentage`.
        Defined lazily so this module imports without Pywr v1.
    """
    from pywr.recorders.progress import ProgressRecorder

    class HydraProgressRecorder(ProgressRecorder):
        def __init__(self, model, **kwargs):
            kwargs.setdefault("print_func", lambda msg: None)
            # Recorders are saved to Hydra under their name, so keep the stock recorder's name
            # rather than a new attribute (`simulated_hydraprogressrecorder_value`) appearing.
            kwargs.setdefault("name", "progressrecorder")
            super().__init__(model, **kwargs)

        def update_progress(self, progress, speed=None):
            if speed is not None:
                message = f"Model run {progress}% complete, {speed:.0f} steps/second"
            else:
                message = f"Model run {progress}% complete"
            report_run_percentage(progress, message)

    return HydraProgressRecorder
