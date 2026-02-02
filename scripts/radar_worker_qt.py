# radar_worker_qt.py
import os
import time
import datetime
import numpy as np
from PyQt5 import QtCore

from processing_utils import spectrogram


class SpectrogramWorker(QtCore.QObject):
    """
    Qt worker that:
      1) Initializes the radar (USB connect + configure)
      2) Collects up to n_frames (or until stop() is called)
      3) Closes the device
      4) Generates + saves a spectrogram PNG using your existing processing_utils.spectrogram()
      5) Emits the saved PNG path
    """
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(str)   # emits saved png path
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        radar,
        cfg_seq: dict,
        cfg_chirp: dict,
        params: dict,
        n_frames: int = 128,
        save_dir: str = "./data/test/spectrogram",
    ):
        super().__init__()
        self.radar = radar
        self.cfg_seq = cfg_seq
        self.cfg_chirp = cfg_chirp
        self.params = params
        self.n_frames = int(n_frames)
        self.save_dir = save_dir

        self._running = False
        self._frames = []

    @QtCore.pyqtSlot()
    def run(self):
        """
        Runs inside a QThread. Do NOT touch UI widgets from here.
        """
        self._running = True
        self._frames = []

        try:
            # Ensure output directory exists (important in GUI apps)
            os.makedirs(self.save_dir, exist_ok=True)

            self.status.emit("Initializing radar...")
            self.radar.init_device_fmcw(self.cfg_seq, self.cfg_chirp)

            self.status.emit(f"Collecting frames (target={self.n_frames})...")

            # Collect frames (batch-style, like your original worker.py)
            while self._running and len(self._frames) < self.n_frames:
                frame_contents = self.radar.device.get_next_frame()
                self._frames.append(frame_contents[0])

            collected = len(self._frames)
            self.status.emit(f"Stopping capture ({collected} frames)...")

            # Close radar right after capture (matches your original pipeline)
            self.radar.close()

            if collected == 0:
                raise RuntimeError("No frames collected (collected=0).")

            data = np.asarray(self._frames)  # (n_frame, n_rx, n_chirp, n_sample)
            duration = float(data.shape[0]) * float(self.cfg_seq["frame_repetition_time_s"])

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            png_path = os.path.join(self.save_dir, f"{timestamp}_spect.png")

            self.status.emit("Generating spectrogram...")
            spectrogram(
                data,
                duration=duration,
                prf=self.params["prf"],
                mti=True,
                is_save=True,      # your spectrogram() ignores this; savename controls saving
                savename=png_path,
            )

            self.status.emit(f"Saved spectrogram: {png_path}")
            self.finished.emit(png_path)

        except Exception as e:
            # Best-effort cleanup
            try:
                if getattr(self.radar, "device", None) is not None:
                    self.radar.close()
            except Exception:
                pass

            self.error.emit(str(e))

    def stop(self):
        """
        Call this from the UI thread to stop the acquisition loop.
        Worker will stop at the next loop iteration.
        """
        self._running = False