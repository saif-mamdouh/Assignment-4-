import time
from datetime import timedelta

class TrainingLogger:
    def __init__(self, log_path, print_every=50):
        self.log_path = log_path
        self.print_every = print_every
        self.epoch_start_time = None
        self.last_log_time = None
        self.samples_seen = 0
        self.total_samples = 0
        self.current_epoch = 0  # حفظ الإبوك الحالي
        open(self.log_path, "w").close()  # clear file

    def start_epoch(self, epoch, total_samples):
        self.epoch_start_time = time.time()
        self.last_log_time = self.epoch_start_time
        self.samples_seen = 0
        self.total_samples = total_samples
        self.current_epoch = epoch
        self._write(f"\nEpoch {epoch} : Starting training ({total_samples} samples)\n")

    def step_samples(self, n):
        self.samples_seen += n
        if self.samples_seen % self.print_every < n:
            now = time.time()
            time_last_interval = now - self.last_log_time
            total_elapsed = now - self.epoch_start_time
            samples_left = max(self.total_samples - self.samples_seen, 0)
            est_time_left = (time_last_interval / self.print_every) * samples_left

            def fmt_hours(sec):
                # تحويل ثواني إلى H:M:S
                td = timedelta(seconds=int(sec))
                # صياغة H:M:S
                h, remainder = divmod(td.seconds, 3600)
                m, s = divmod(remainder, 60)
                return f"{h}:{m:02d}:{s:02d} hours"

            self._write(
                f"Epoch {self.current_epoch} : {self.samples_seen} / {self.total_samples} samples ,\n"
                f"time for last {self.print_every} samples : {fmt_hours(time_last_interval)} ,\n"
                f"time since beginning : {fmt_hours(total_elapsed)} ,\n"
                f"time left to finish the epoch : {fmt_hours(est_time_left)}\n\n"
            )
            self.last_log_time = now

    def log_metrics(self, epoch, step, **kwargs):
        msg = f"[Epoch {epoch}][Step {step}] "
        msg += " ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
        self._write(msg + "\n")

    def _write(self, msg):
        print(msg.strip())
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
