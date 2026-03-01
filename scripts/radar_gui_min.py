# ============================================================
# PhyGO Radar GUI (UI Match + Conda-safe + macOS-safe)
# Batch A: Imports, Path Setup, Backend, Helpers, Workers
# ============================================================

import sys
import time
import json
import shutil
from pathlib import Path

# Ensure scripts/ dir is on path (handles running from any working directory)
_here = Path(__file__).resolve().parent          # phygo/scripts/
_repo_root = _here.parent                        # phygo/
for _p in (_here, _repo_root):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Repo root (phygo/) — used for configs/ and data/
REPO_ROOT = _repo_root

import numpy as np

# CRITICAL: Set matplotlib backend BEFORE any other matplotlib imports
import matplotlib
matplotlib.use("Qt5Agg")

# Keras / TF for inference (graceful fallback if not installed)
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

# VEX AIM robot client (graceful fallback if not installed)
# Place the vex/ folder inside phygo/scripts/ (same folder as this script)
_aim_import_error_msg = ""
try:
    from vex import aim as vex_aim
    from vex.vex_types import TurnType
    AIM_AVAILABLE = True
except Exception as _e:
    AIM_AVAILABLE = False
    _aim_import_error_msg = str(_e)

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl

# Matplotlib canvas for embedding in PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import InfineonManager
from processing_utils import spectrogram, compute_microdoppler_spectrogram_db

# SDK exception for dropped frames (optional)
try:
    from ifxradarsdk.common.exceptions import ErrorFrameAcquisitionFailed
except Exception:
    ErrorFrameAcquisitionFailed = Exception


def get_resource_path(relative_path: str) -> Path:
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    except Exception:
        base_path = REPO_ROOT
    return base_path / relative_path


# -----------------------------
# Streaming Worker
# -----------------------------
class RadarStreamWorker(QtCore.QObject):
    frame = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self._running = False

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        self.status.emit("Streaming started.")
        try:
            while self._running:
                frame_contents = self.manager.device.get_next_frame()
                frame0 = frame_contents[0]
                self.frame.emit(frame0)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.status.emit("Streaming stopped.")
            self.finished.emit()

    def stop(self):
        self._running = False


# -----------------------------
# Matplotlib spectrogram canvas
# -----------------------------
class SpectrogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax.set_title("Live Micro-Doppler Spectrogram")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Velocity (m/s)")

        self.im = None
        self.cbar = None

    def update_image(self, spect_db: np.ndarray):
        if self.im is None:
            self.im = self.ax.imshow(
                spect_db,
                aspect="auto",
                origin="lower",
                vmin=-20,
                vmax=0,
                cmap="jet",
            )
            self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        else:
            self.im.set_data(spect_db)

        self.ax.set_xlim(0, spect_db.shape[1] - 1)
        self.ax.set_ylim(0, spect_db.shape[0] - 1)
        self.draw_idle()


# -----------------------------
# Capture helpers
# -----------------------------
def safe_get_next_frame(device, frame_retry: int = 10, retry_sleep: float = 0.01):
    failures = 0
    while True:
        try:
            frame_contents = device.get_next_frame()
            return frame_contents[0]
        except ErrorFrameAcquisitionFailed:
            failures += 1
            if failures > frame_retry:
                raise
            time.sleep(retry_sleep)


def estimate_seconds_per_frame(device, n_probe: int = 10, frame_retry: int = 10, retry_sleep: float = 0.01) -> float:
    t0 = time.time()
    for _ in range(n_probe):
        _ = safe_get_next_frame(device, frame_retry, retry_sleep)
    t1 = time.time()
    return (t1 - t0) / float(n_probe)


def fetch_n_frames(device, n_frames: int, frame_retry: int = 10, retry_sleep: float = 0.01) -> np.ndarray:
    frames = []
    while len(frames) < n_frames:
        frames.append(safe_get_next_frame(device, frame_retry, retry_sleep))
    return np.array(frames)


# -----------------------------
# Batch Capture Worker
# NOTE: Uses its own radar instance (matches your UI reference design)
# -----------------------------
class BatchCaptureWorker(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    sample_finished = QtCore.pyqtSignal(str, str, float, int, object)
    all_finished = QtCore.pyqtSignal()
    delay_countdown = QtCore.pyqtSignal(int)

    def __init__(
        self,
        cfg_seq: dict,
        cfg_chirp: dict,
        subject: str,
        gesture: str,
        batch_count: int = 1,
        target_seconds: float = 6.0,
        inter_sample_delay: float = 3.0,
        warmup_frames: int = 3,
        frame_retry: int = 10,
        retry_sleep: float = 0.01,
    ):
        super().__init__()
        self.cfg_seq = cfg_seq
        self.cfg_chirp = cfg_chirp
        self.subject = subject
        self.gesture = gesture
        self.batch_count = batch_count
        self.target_seconds = float(target_seconds)
        self.inter_sample_delay = float(inter_sample_delay)
        self.warmup_frames = int(warmup_frames)
        self.frame_retry = int(frame_retry)
        self.retry_sleep = float(retry_sleep)
        self._should_stop = False

    def stop(self):
        self._should_stop = True

    @QtCore.pyqtSlot()
    def run(self):
        radar = InfineonManager.InfineonManager()
        try:
            self.status.emit("Initializing radar for batch capture...")
            radar.init_device_fmcw(self.cfg_seq, self.cfg_chirp)

            params = radar.get_params({**self.cfg_chirp, **self.cfg_seq})

            self.status.emit("Initial warmup...")
            for _ in range(self.warmup_frames):
                _ = safe_get_next_frame(radar.device, self.frame_retry, self.retry_sleep)

            self.status.emit("Estimating frame rate...")
            sec_per_frame = estimate_seconds_per_frame(
                radar.device, n_probe=10, frame_retry=self.frame_retry, retry_sleep=self.retry_sleep
            )
            n_frames = max(8, int(round(self.target_seconds / sec_per_frame)))

            for sample_num in range(1, self.batch_count + 1):
                if self._should_stop:
                    self.status.emit("Batch capture stopped by user")
                    break

                self.status.emit(
                    f"Capturing sample {sample_num}/{self.batch_count} "
                    f"(~{self.target_seconds:.1f}s, frames={n_frames})..."
                )

                t0 = time.time()
                data = fetch_n_frames(radar.device, n_frames, self.frame_retry, self.retry_sleep)
                t1 = time.time()
                capture_seconds = t1 - t0

                base = REPO_ROOT / "data/train"
                raw_dir = base / "raw_data" / self.subject / self.gesture
                spect_dir = base / "spectrogram" / self.subject / self.gesture
                raw_dir.mkdir(parents=True, exist_ok=True)
                spect_dir.mkdir(parents=True, exist_ok=True)

                existing = sorted(spect_dir.glob("*.png"))
                next_idx = len(existing) + 1
                fname = f"{next_idx:04d}"

                raw_path = raw_dir / f"{fname}.npy"
                png_path = spect_dir / f"{fname}.png"

                np.save(raw_path, data)

                capture_info = {
                    "data": data,
                    "duration": data.shape[0] * float(self.cfg_seq["frame_repetition_time_s"]),
                    "prf": params["prf"],
                    "sample_num": sample_num,
                    "total_samples": self.batch_count,
                }
                self.sample_finished.emit(
                    str(png_path), str(raw_path), float(capture_seconds), int(data.shape[0]), capture_info
                )

                if sample_num < self.batch_count and not self._should_stop:
                    if self.inter_sample_delay > 0:
                        self.status.emit(f"Waiting {int(self.inter_sample_delay)}s before next sample...")
                        for remaining in range(int(self.inter_sample_delay), 0, -1):
                            self.delay_countdown.emit(remaining)
                            time.sleep(1)

                    for _ in range(3):
                        _ = safe_get_next_frame(radar.device, self.frame_retry, self.retry_sleep)

            self.status.emit("Batch capture complete.")
            self.all_finished.emit()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            try:
                radar.close()
            except Exception:
                pass
    
    # ============================================================
# Batch B: Train Model Tab + Control Robot Tab (UI Match)
# ============================================================

# -----------------------------
# Train Model Tab Widget
# -----------------------------
# Updated TrainModelTab for local Vision Transformer training
# Replace the existing TrainModelTab class with this

class TrainModelTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._training_process = None
        self._build_ui()

        # Refresh checklist periodically
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(3000)
        self.refresh_timer.timeout.connect(self.refresh_checklist)
        self.refresh_timer.start()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # ---------- Title ----------
        title = QtWidgets.QLabel("Train Your Gesture Model")
        title.setStyleSheet("font-size:18px; font-weight:700; color:#111827;")
        root.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Train a Vision Transformer model locally using your collected spectrograms."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size:12px; color:#6b7280;")
        root.addWidget(subtitle)

        # ---------- Main content: checklist left, training panel right ----------
        content = QtWidgets.QHBoxLayout()
        content.setSpacing(14)

        # ---- Left: Sample Checklist (KEEP THIS) ----
        checklist_frame = QtWidgets.QFrame()
        checklist_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        checklist_frame.setStyleSheet("""
            QFrame {
                background: #f9fafb;
                border-radius: 12px;
            }
        """)
        checklist_lay = QtWidgets.QVBoxLayout(checklist_frame)
        checklist_lay.setContentsMargins(14, 14, 14, 14)
        checklist_lay.setSpacing(8)

        checklist_title = QtWidgets.QLabel("📊 Dataset Checklist")
        checklist_title.setStyleSheet("font-size:13px; font-weight:700; color:#111827;")
        checklist_lay.addWidget(checklist_title)

        checklist_sub = QtWidgets.QLabel("Collect at least 20 samples per gesture before training. Counts include all subjects.")
        checklist_sub.setWordWrap(True)
        checklist_sub.setStyleSheet("font-size:11px; color:#6b7280;")
        checklist_lay.addWidget(checklist_sub)

        # Gesture rows
        self.gesture_labels = {}
        gestures = ["push", "swipe_left", "swipe_right", "swipe_up", "swipe_down", "idle"]
        for gesture in gestures:
            row = QtWidgets.QHBoxLayout()
            icon = QtWidgets.QLabel("❌")
            icon.setFixedWidth(20)
            lbl = QtWidgets.QLabel(gesture)
            lbl.setStyleSheet("font-size:12px; color:#374151;")
            count = QtWidgets.QLabel("0 samples")
            count.setStyleSheet("font-size:11px; color:#9ca3af;")
            count.setAlignment(QtCore.Qt.AlignRight)
            row.addWidget(icon)
            row.addWidget(lbl, 1)
            row.addWidget(count)
            checklist_lay.addLayout(row)
            self.gesture_labels[gesture] = (icon, count)

        checklist_lay.addSpacing(8)

        # Overall readiness
        self.lbl_readiness = QtWidgets.QLabel("Collect samples to get started")
        self.lbl_readiness.setWordWrap(True)
        self.lbl_readiness.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_readiness.setStyleSheet("""
            font-size: 11px; font-weight: 600;
            padding: 8px; border-radius: 8px;
            background: #f3f4f6; color: #374151;
        """)
        checklist_lay.addWidget(self.lbl_readiness)

        btn_refresh = QtWidgets.QPushButton("🔄 Refresh")
        btn_refresh.setStyleSheet("""
            QPushButton {
                background: #e5e7eb; color: #374151;
                border-radius: 8px; padding: 6px;
                font-size: 11px; font-weight: 600;
            }
            QPushButton:hover { background: #d1d5db; }
        """)
        btn_refresh.clicked.connect(self.refresh_checklist)
        checklist_lay.addWidget(btn_refresh)
        checklist_lay.addStretch(1)
        content.addWidget(checklist_frame, 2)

        # ---- Right: Training Panel ----
        training_frame = QtWidgets.QFrame()
        training_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        training_frame.setStyleSheet("""
            QFrame {
                background: #f9fafb;
                border-radius: 12px;
            }
        """)
        training_lay = QtWidgets.QVBoxLayout(training_frame)
        training_lay.setContentsMargins(14, 14, 14, 14)
        training_lay.setSpacing(10)

        training_title = QtWidgets.QLabel("🚀 Local Training")
        training_title.setStyleSheet("font-size:13px; font-weight:700; color:#111827;")
        training_lay.addWidget(training_title)

        # Training parameters
        params_grid = QtWidgets.QGridLayout()
        params_grid.setSpacing(8)

        # Epochs
        epochs_lbl = QtWidgets.QLabel("Epochs:")
        epochs_lbl.setStyleSheet("font-size:11px; color:#6b7280;")
        self.spin_epochs = QtWidgets.QSpinBox()
        self.spin_epochs.setRange(5, 50)
        self.spin_epochs.setValue(10)
        self.spin_epochs.setMinimumHeight(28)
        params_grid.addWidget(epochs_lbl, 0, 0)
        params_grid.addWidget(self.spin_epochs, 0, 1)

        # Batch size
        batch_lbl = QtWidgets.QLabel("Batch size:")
        batch_lbl.setStyleSheet("font-size:11px; color:#6b7280;")
        self.spin_batch = QtWidgets.QSpinBox()
        self.spin_batch.setRange(4, 32)
        self.spin_batch.setValue(8)
        self.spin_batch.setMinimumHeight(28)
        params_grid.addWidget(batch_lbl, 1, 0)
        params_grid.addWidget(self.spin_batch, 1, 1)

        # Learning rate
        lr_lbl = QtWidgets.QLabel("Learning rate:")
        lr_lbl.setStyleSheet("font-size:11px; color:#6b7280;")
        self.spin_lr = QtWidgets.QDoubleSpinBox()
        self.spin_lr.setDecimals(5)
        self.spin_lr.setRange(0.00001, 0.001)
        self.spin_lr.setSingleStep(0.00001)
        self.spin_lr.setValue(0.00002)
        self.spin_lr.setMinimumHeight(28)
        params_grid.addWidget(lr_lbl, 2, 0)
        params_grid.addWidget(self.spin_lr, 2, 1)

        training_lay.addLayout(params_grid)

        # Info box
        info_box = QtWidgets.QLabel(
            "💡 Training uses Vision Transformer (ViT) with transfer learning. "
            "The model trains locally on your computer - no internet required after initial download."
        )
        info_box.setWordWrap(True)
        info_box.setStyleSheet("""
            font-size:10px; color:#6b7280; 
            background:#e0e7ff; padding:8px; 
            border-radius:6px;
        """)
        training_lay.addWidget(info_box)

        # Training buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_train = QtWidgets.QPushButton("🧠  Train Model")
        self.btn_train.setMinimumHeight(40)
        self.btn_train.setStyleSheet("""
            QPushButton {
                background: #7c3aed; color: white;
                border-radius: 10px; font-size: 13px; font-weight: 700;
                padding: 6px 16px;
            }
            QPushButton:hover { background: #6d28d9; }
            QPushButton:disabled { background: #9ca3af; }
        """)
        self.btn_train.clicked.connect(self._on_train_model)
        btn_row.addWidget(self.btn_train)

        self.btn_open_folder = QtWidgets.QPushButton("📂  Open Data Folder")
        self.btn_open_folder.setMinimumHeight(40)
        self.btn_open_folder.setStyleSheet("""
            QPushButton {
                background: #6b7280; color: white;
                border-radius: 10px; font-size: 13px; font-weight: 700;
                padding: 6px 16px;
            }
            QPushButton:hover { background: #4b5563; }
        """)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        btn_row.addWidget(self.btn_open_folder)

        training_lay.addLayout(btn_row)

        # Training output console
        console_lbl = QtWidgets.QLabel("Training Output:")
        console_lbl.setStyleSheet("font-size:11px; color:#6b7280; font-weight:600;")
        training_lay.addWidget(console_lbl)

        self.training_console = QtWidgets.QPlainTextEdit()
        self.training_console.setReadOnly(True)
        self.training_console.setMinimumHeight(200)
        self.training_console.setStyleSheet("""
            QPlainTextEdit {
                background: #0b1220; color: #e5e7eb;
                border-radius: 8px; padding: 8px;
                font-family: 'Monaco', 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        self.training_console.setPlaceholderText("Training logs will appear here...")
        training_lay.addWidget(self.training_console)

        # Training status
        self.lbl_training_status = QtWidgets.QLabel("Ready to train")
        self.lbl_training_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_training_status.setWordWrap(True)
        self.lbl_training_status.setStyleSheet("""
            font-size: 11px; color: #6b7280;
            padding: 8px; border-radius: 8px;
            background: #f3f4f6;
        """)
        training_lay.addWidget(self.lbl_training_status)

        content.addWidget(training_frame, 3)
        root.addLayout(content)

        self.refresh_checklist()

    def refresh_checklist(self):
        """Scan spectrogram folders and update the checklist"""
        base = REPO_ROOT / "data/train/spectrogram"
        gestures = ["push", "swipe_left", "swipe_right", "swipe_up", "swipe_down", "idle"]

        ready_count = 0
        total_samples = 0
        MIN_SAMPLES = 20

        for gesture in gestures:
            icon_lbl, count_lbl = self.gesture_labels[gesture]
            count = 0
            if base.exists():
                count = len(list(base.glob(f"*/{gesture}/*.png")))
            total_samples += count

            if count >= MIN_SAMPLES:
                icon_lbl.setText("✅")
                count_lbl.setText(f"{count} samples")
                count_lbl.setStyleSheet("font-size:11px; color:#10b981; font-weight:600;")
                ready_count += 1
            elif count > 0:
                icon_lbl.setText("⚠️")
                count_lbl.setText(f"{count}/{MIN_SAMPLES} samples")
                count_lbl.setStyleSheet("font-size:11px; color:#f59e0b; font-weight:600;")
            else:
                icon_lbl.setText("❌")
                count_lbl.setText("0 samples")
                count_lbl.setStyleSheet("font-size:11px; color:#9ca3af;")

        # Overall readiness message
        if ready_count >= 2 and total_samples >= MIN_SAMPLES * 2:
            self.lbl_readiness.setText(f"🎉 Ready to train! {ready_count} gesture(s) have enough samples.")
            self.lbl_readiness.setStyleSheet("""
                font-size: 11px; font-weight: 600;
                padding: 8px; border-radius: 8px;
                background: #d1fae5; color: #065f46;
            """)
        elif total_samples > 0:
            needed = max(0, MIN_SAMPLES * 2 - total_samples)
            self.lbl_readiness.setText(f"⚠️ Collect {needed} more samples across at least 2 gestures.")
            self.lbl_readiness.setStyleSheet("""
                font-size: 11px; font-weight: 600;
                padding: 8px; border-radius: 8px;
                background: #fef3c7; color: #92400e;
            """)
        else:
            self.lbl_readiness.setText("Go to 'Collect Data' tab to capture samples first.")
            self.lbl_readiness.setStyleSheet("""
                font-size: 11px; font-weight: 600;
                padding: 8px; border-radius: 8px;
                background: #f3f4f6; color: #374151;
            """)

    def _on_open_folder(self):
        folder = REPO_ROOT / "data/train/spectrogram"
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_train_model(self):
        """Launch training in subprocess to avoid blocking GUI"""
        # Check if enough data
        base = REPO_ROOT / "data/train/spectrogram"
        if not base.exists():
            QtWidgets.QMessageBox.warning(
                self, "No Data", 
                "No training data found. Collect samples first in the 'Collect Data' tab."
            )
            return

        # Disable train button
        self.btn_train.setEnabled(False)
        self.btn_train.setText("Training...")
        self.training_console.clear()
        self.lbl_training_status.setText("🚀 Training in progress...")
        self.lbl_training_status.setStyleSheet("""
            font-size: 11px; font-weight: 600;
            padding: 8px; border-radius: 8px;
            background: #dbeafe; color: #1e40af;
        """)

        # Build command
        epochs = self.spin_epochs.value()
        batch_size = self.spin_batch.value()
        lr = self.spin_lr.value()

        train_script = REPO_ROOT / "scripts/train_vit.py"
        cmd = [
            sys.executable,
            str(train_script),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--data-dir", str(REPO_ROOT / "data/train/spectrogram"),
            "--output-dir", str(REPO_ROOT / "data/model"),
        ]

        # Start training process
        from subprocess import Popen, PIPE, STDOUT
        
        self._training_process = Popen(
            cmd,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Monitor output in a thread
        import threading
        def read_output():
            for line in iter(self._training_process.stdout.readline, ''):
                if line:
                    QtCore.QMetaObject.invokeMethod(
                        self,
                        "_append_training_log",
                        QtCore.Qt.QueuedConnection,
                        QtCore.Q_ARG(str, line.rstrip())
                    )
            
            # Process finished
            returncode = self._training_process.wait()
            QtCore.QMetaObject.invokeMethod(
                self,
                "_on_training_finished",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, returncode)
            )

        threading.Thread(target=read_output, daemon=True).start()

    @QtCore.pyqtSlot(str)
    def _append_training_log(self, line: str):
        """Append line to training console (called from worker thread)"""
        self.training_console.appendPlainText(line)
        # Auto-scroll to bottom
        self.training_console.verticalScrollBar().setValue(
            self.training_console.verticalScrollBar().maximum()
        )

    @QtCore.pyqtSlot(int)
    def _on_training_finished(self, returncode: int):
        """Called when training process completes"""
        self.btn_train.setEnabled(True)
        self.btn_train.setText("🧠  Train Model")
        
        if returncode == 0:
            self.lbl_training_status.setText("✅ Training complete! Model saved to data/model/")
            self.lbl_training_status.setStyleSheet("""
                font-size: 11px; font-weight: 600;
                padding: 8px; border-radius: 8px;
                background: #d1fae5; color: #065f46;
            """)
            self.training_console.appendPlainText("\n✅ SUCCESS - Model ready to use in Control Robot tab!")
        else:
            self.lbl_training_status.setText("❌ Training failed - check console output")
            self.lbl_training_status.setStyleSheet("""
                font-size: 11px; font-weight: 600;
                padding: 8px; border-radius: 8px;
                background: #fee2e2; color: #991b1b;
            """)
            self.training_console.appendPlainText(f"\n❌ FAILED - Exit code: {returncode}")


# -----------------------------
# Control Robot Tab
# -----------------------------
COMMAND_OPTIONS = ["kick", "move_forward", "move_backward", "turn_left", "turn_right", "stop"]

DEFAULT_MAPPING = {
    "push": "kick",
    "swipe_up": "move_forward",
    "swipe_down": "move_backward",
    "swipe_left": "turn_left",
    "swipe_right": "turn_right",
    "idle": "stop",
}


class RobotCommandWorker(QtCore.QObject):
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, robot, command: str, distance_mm: float = 200.0):
        super().__init__()
        self.robot = robot
        self.command = command
        self.distance_mm = distance_mm

    @QtCore.pyqtSlot()
    def run(self):
        try:
            cmd = self.command
            self.status.emit(f"🤖 Sending: {cmd}")

            from vex.vex_types import TurnType, KickType, LightType, Color
            ALL_LEDS = LightType.ALL_LEDS
            YELLOW = Color.YELLOW
            MEDIUM = KickType.MEDIUM

            if cmd == "kick":
                self.robot.led.on(ALL_LEDS, YELLOW)
                self.robot.kicker.kick(MEDIUM)

            elif cmd == "move_forward":
                self.robot.move_for(self.distance_mm, 0)

            elif cmd == "move_backward":
                self.robot.move_for(-self.distance_mm, 0)  # Negative distance = backward

            elif cmd == "turn_left":
                self.robot.led.on(LightType.LED1, YELLOW)
                self.robot.led.on(LightType.LED2, YELLOW)
                self.robot.turn_for(TurnType.LEFT, 90)

            elif cmd == "turn_right":
                self.robot.led.on(LightType.LED3, YELLOW)
                self.robot.led.on(LightType.LED4, YELLOW)
                self.robot.turn_for(TurnType.RIGHT, 90)

            elif cmd == "stop":
                self.robot.stop_all_movement()

            else:
                self.status.emit(f"⚠️ Unknown command: {cmd}")
                return

            self.robot.led.off(ALL_LEDS)
            self.status.emit(f"✅ Done: {cmd}")

        except Exception as e:
            self.status.emit(f"❌ Robot command failed: {e}")
        finally:
            self.finished.emit()


class PredictWorker(QtCore.QObject):
    result = QtCore.pyqtSignal(str, float)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, model, processor, labels: list, png_path: str, model_type: str = "keras"):
        super().__init__()
        self.model = model
        self.processor = processor
        self.labels = labels
        self.png_path = png_path
        self.model_type = model_type

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.status.emit("🔍 Running inference...")
            
            if self.model_type == "pytorch":
                self._run_pytorch_inference()
            else:
                self._run_keras_inference()
                
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
    
    def _run_pytorch_inference(self):
        """Run inference with PyTorch ViT model"""
        import torch
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(self.png_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get prediction
        idx = int(torch.argmax(probs))
        conf = float(probs[idx])
        label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
        
        self.status.emit(f"🎯 Predicted: {label} ({conf*100:.1f}%)")
        self.result.emit(label, conf)
    
    def _run_keras_inference(self):
        """Run inference with Keras/TM model"""
        if not KERAS_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras is not installed.")

        import tensorflow as tf

        class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

        with tf.keras.utils.custom_object_scope({"DepthwiseConv2D": FixedDepthwiseConv2D}):
            model = keras.models.load_model(self.model_path, compile=False)

        img = keras.utils.load_img(self.png_path, target_size=(224, 224))
        arr = keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, axis=0) / 255.0

        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"

        self.status.emit(f"🎯 Predicted: {label} ({conf*100:.1f}%)")
        self.result.emit(label, conf)

class ControlRobotTab(QtWidgets.QWidget):
    request_capture = QtCore.pyqtSignal(float)
    log_message = QtCore.pyqtSignal(str)

    def __init__(self, cfg_seq, cfg_chirp, parent=None):
        super().__init__(parent)
        self.cfg_seq = cfg_seq
        self.cfg_chirp = cfg_chirp

        self._robot = None
        self._model = None
        self._labels = []
        self._model_path = None

        self._cmd_thread = None
        self._cmd_worker = None
        self._pred_thread = None
        self._pred_worker = None

        self._pending_png = None

        self._build_ui()
        self._try_load_model()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        title = QtWidgets.QLabel("Control Robot")
        title.setStyleSheet("font-size:18px; font-weight:700; color:#111827;")
        root.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Connect to your VEX AIM robot, map gestures to commands, then record a gesture to control it."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size:12px; color:#6b7280;")
        root.addWidget(subtitle)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(12)

        # Connection panel
        conn_frame = QtWidgets.QFrame()
        conn_frame.setStyleSheet("QFrame { background:#f9fafb; border-radius:12px; }")
        conn_lay = QtWidgets.QVBoxLayout(conn_frame)
        conn_lay.setContentsMargins(14, 14, 14, 14)
        conn_lay.setSpacing(8)

        conn_title = QtWidgets.QLabel("🔌 Robot Connection")
        conn_title.setStyleSheet("font-size:13px; font-weight:700; color:#111827;")
        conn_lay.addWidget(conn_title)

        ip_row = QtWidgets.QHBoxLayout()
        ip_lbl = QtWidgets.QLabel("IP Address:")
        ip_lbl.setStyleSheet("font-size:12px;")
        self.in_robot_ip = QtWidgets.QLineEdit("192.168.4.1")
        self.in_robot_ip.setMinimumHeight(28)
        ip_row.addWidget(ip_lbl)
        ip_row.addWidget(self.in_robot_ip)
        conn_lay.addLayout(ip_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_robot_connect = QtWidgets.QPushButton("Connect")
        self.btn_robot_connect.setStyleSheet("""
            QPushButton { background:#2563eb; color:white; border-radius:8px;
                          font-weight:700; padding:6px 12px; }
            QPushButton:hover { background:#1d4ed8; }
            QPushButton:disabled { background:#9ca3af; }
        """)
        self.btn_robot_disconnect = QtWidgets.QPushButton("Disconnect")
        self.btn_robot_disconnect.setEnabled(False)
        self.btn_robot_disconnect.setStyleSheet("""
            QPushButton { background:#6b7280; color:white; border-radius:8px;
                          font-weight:700; padding:6px 12px; }
            QPushButton:hover { background:#4b5563; }
            QPushButton:disabled { background:#9ca3af; }
        """)
        btn_row.addWidget(self.btn_robot_connect)
        btn_row.addWidget(self.btn_robot_disconnect)
        conn_lay.addLayout(btn_row)

        self.lbl_conn_status = QtWidgets.QLabel("Not connected")
        self.lbl_conn_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_conn_status.setStyleSheet("""
            font-size:11px; font-weight:600; padding:6px; border-radius:8px;
            background:#f3f4f6; color:#374151;
        """)
        conn_lay.addWidget(self.lbl_conn_status)
        top_row.addWidget(conn_frame, 1)

        # Model status panel
        model_frame = QtWidgets.QFrame()
        model_frame.setStyleSheet("QFrame { background:#f9fafb; border-radius:12px; }")
        model_lay = QtWidgets.QVBoxLayout(model_frame)
        model_lay.setContentsMargins(14, 14, 14, 14)
        model_lay.setSpacing(8)

        model_title = QtWidgets.QLabel("🧠 Model Status")
        model_title.setStyleSheet("font-size:13px; font-weight:700; color:#111827;")
        model_lay.addWidget(model_title)

        self.lbl_model_loaded = QtWidgets.QLabel("No model loaded.")
        self.lbl_model_loaded.setWordWrap(True)
        self.lbl_model_loaded.setStyleSheet("font-size:11px; color:#6b7280;")
        model_lay.addWidget(self.lbl_model_loaded)

        self.lbl_labels_list = QtWidgets.QLabel("")
        self.lbl_labels_list.setWordWrap(True)
        self.lbl_labels_list.setStyleSheet("font-size:11px; color:#374151;")
        model_lay.addWidget(self.lbl_labels_list)

        btn_reload = QtWidgets.QPushButton("🔄 Reload Model")
        btn_reload.setStyleSheet("""
            QPushButton { background:#e5e7eb; color:#374151; border-radius:8px;
                          font-size:11px; font-weight:600; padding:6px; }
            QPushButton:hover { background:#d1d5db; }
        """)
        btn_reload.clicked.connect(self._try_load_model)
        model_lay.addWidget(btn_reload)
        
        btn_import = QtWidgets.QPushButton("📥 Import Model")
        btn_import.setStyleSheet("""
            QPushButton { background:#10b981; color:white; border-radius:8px;
                          font-size:11px; font-weight:600; padding:6px; }
            QPushButton:hover { background:#059669; }
        """)
        btn_import.clicked.connect(self._on_import_model)
        model_lay.addWidget(btn_import)
        
        model_lay.addStretch(1)

        top_row.addWidget(model_frame, 1)
        root.addLayout(top_row)

        # Mapping section (collapsible)
        mapping_frame = QtWidgets.QFrame()
        mapping_frame.setStyleSheet("QFrame { background:#f9fafb; border-radius:12px; }")
        mapping_outer = QtWidgets.QVBoxLayout(mapping_frame)
        mapping_outer.setContentsMargins(14, 10, 14, 10)
        mapping_outer.setSpacing(0)

        toggle_row = QtWidgets.QHBoxLayout()
        self._btn_mapping_toggle = QtWidgets.QPushButton("▶  Gesture → Command Mapping")
        self._btn_mapping_toggle.setStyleSheet("""
            QPushButton {
                font-size:12px; font-weight:700; color:#374151;
                background:transparent; border:none;
                text-align:left; padding:4px 0px;
            }
            QPushButton:hover { color:#111827; }
        """)
        self._btn_mapping_toggle.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        toggle_hint = QtWidgets.QLabel("customize gesture→command mapping")
        toggle_hint.setStyleSheet("font-size:10px; color:#9ca3af;")

        toggle_row.addWidget(self._btn_mapping_toggle)
        toggle_row.addStretch(1)
        toggle_row.addWidget(toggle_hint)
        mapping_outer.addLayout(toggle_row)

        self._mapping_content = QtWidgets.QWidget()
        mapping_lay = QtWidgets.QVBoxLayout(self._mapping_content)
        mapping_lay.setContentsMargins(0, 8, 0, 4)
        mapping_lay.setSpacing(8)

        map_sub = QtWidgets.QLabel("Assign a robot command to each gesture your model can predict.")
        map_sub.setStyleSheet("font-size:11px; color:#6b7280;")
        mapping_lay.addWidget(map_sub)

        header_row = QtWidgets.QHBoxLayout()
        for txt in ("Gesture Label", "Robot Command"):
            h = QtWidgets.QLabel(txt)
            h.setStyleSheet("font-size:11px; font-weight:700; color:#6b7280; text-transform:uppercase;")
            header_row.addWidget(h, 1)
        mapping_lay.addLayout(header_row)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        scroll_widget = QtWidgets.QWidget()
        self._mapping_grid = QtWidgets.QVBoxLayout(scroll_widget)
        self._mapping_grid.setSpacing(4)
        self._mapping_dropdowns = {}
        scroll.setWidget(scroll_widget)
        mapping_lay.addWidget(scroll)

        self._mapping_content.setVisible(False)
        mapping_outer.addWidget(self._mapping_content)
        self._btn_mapping_toggle.clicked.connect(self._toggle_mapping_panel)
        root.addWidget(mapping_frame)

        # Record & Predict panel
        predict_frame = QtWidgets.QFrame()
        predict_frame.setStyleSheet("QFrame { background:#f9fafb; border-radius:12px; }")
        predict_lay = QtWidgets.QVBoxLayout(predict_frame)
        predict_lay.setContentsMargins(14, 14, 14, 14)
        predict_lay.setSpacing(10)

        pred_title = QtWidgets.QLabel("🎯 Record Gesture & Send Command")
        pred_title.setStyleSheet("font-size:13px; font-weight:700; color:#111827;")
        predict_lay.addWidget(pred_title)

        dur_row = QtWidgets.QHBoxLayout()
        dur_lbl = QtWidgets.QLabel("Capture duration:")
        dur_lbl.setStyleSheet("font-size:12px;")
        self.spin_pred_seconds = QtWidgets.QDoubleSpinBox()
        self.spin_pred_seconds.setRange(1.0, 20.0)
        self.spin_pred_seconds.setSingleStep(0.5)
        self.spin_pred_seconds.setValue(6.0)
        self.spin_pred_seconds.setSuffix(" sec")
        self.spin_pred_seconds.setMaximumWidth(120)

        dist_lbl = QtWidgets.QLabel("Move distance:")
        dist_lbl.setStyleSheet("font-size:12px;")
        self.spin_distance = QtWidgets.QDoubleSpinBox()
        self.spin_distance.setRange(10.0, 1000.0)
        self.spin_distance.setSingleStep(10.0)
        self.spin_distance.setValue(50.0)
        self.spin_distance.setSuffix(" mm")
        self.spin_distance.setMaximumWidth(120)

        dur_row.addWidget(dur_lbl)
        dur_row.addWidget(self.spin_pred_seconds)
        dur_row.addSpacing(20)
        dur_row.addWidget(dist_lbl)
        dur_row.addWidget(self.spin_distance)
        dur_row.addStretch(1)
        predict_lay.addLayout(dur_row)

        self.btn_record_predict = QtWidgets.QPushButton("🎙  Record Gesture & Send to Robot")
        self.btn_record_predict.setMinimumHeight(44)
        self.btn_record_predict.setStyleSheet("""
            QPushButton {
                background: #7c3aed; color: white;
                border-radius: 10px; font-size: 14px; font-weight: 700;
            }
            QPushButton:hover { background: #6d28d9; }
            QPushButton:disabled { background: #9ca3af; }
        """)
        self.btn_record_predict.clicked.connect(self._on_record_predict)
        predict_lay.addWidget(self.btn_record_predict)

        result_row = QtWidgets.QHBoxLayout()
        result_row.setSpacing(10)

        pred_box = QtWidgets.QVBoxLayout()
        pred_box_lbl = QtWidgets.QLabel("Predicted Gesture")
        pred_box_lbl.setStyleSheet("font-size:11px; color:#6b7280; font-weight:600;")
        pred_box_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_predicted = QtWidgets.QLabel("—")
        self.lbl_predicted.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_predicted.setMinimumHeight(60)
        self.lbl_predicted.setStyleSheet("""
            font-size:22px; font-weight:800; color:#111827;
            background:#f3f4f6; border-radius:10px; padding:10px;
        """)
        pred_box.addWidget(pred_box_lbl)
        pred_box.addWidget(self.lbl_predicted)
        result_row.addLayout(pred_box, 1)

        conf_box = QtWidgets.QVBoxLayout()
        conf_box_lbl = QtWidgets.QLabel("Confidence")
        conf_box_lbl.setStyleSheet("font-size:11px; color:#6b7280; font-weight:600;")
        conf_box_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_confidence = QtWidgets.QLabel("—")
        self.lbl_confidence.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_confidence.setMinimumHeight(60)
        self.lbl_confidence.setStyleSheet("""
            font-size:22px; font-weight:800; color:#111827;
            background:#f3f4f6; border-radius:10px; padding:10px;
        """)
        conf_box.addWidget(conf_box_lbl)
        conf_box.addWidget(self.lbl_confidence)
        result_row.addLayout(conf_box, 1)

        cmd_box = QtWidgets.QVBoxLayout()
        cmd_box_lbl = QtWidgets.QLabel("Command Sent")
        cmd_box_lbl.setStyleSheet("font-size:11px; color:#6b7280; font-weight:600;")
        cmd_box_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_cmd_sent = QtWidgets.QLabel("—")
        self.lbl_cmd_sent.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_cmd_sent.setMinimumHeight(60)
        self.lbl_cmd_sent.setWordWrap(True)
        self.lbl_cmd_sent.setStyleSheet("""
            font-size:14px; font-weight:700; color:#111827;
            background:#f3f4f6; border-radius:10px; padding:10px;
        """)
        cmd_box.addWidget(cmd_box_lbl)
        cmd_box.addWidget(self.lbl_cmd_sent)
        result_row.addLayout(cmd_box, 1)

        predict_lay.addLayout(result_row)
        root.addWidget(predict_frame)
        root.addStretch(1)

        self.btn_robot_connect.clicked.connect(self._on_robot_connect)
        self.btn_robot_disconnect.clicked.connect(self._on_robot_disconnect)

    # -----------------------------
    # Model loading + mapping
    # -----------------------------
    def _try_load_model(self):
        model_path = REPO_ROOT / "data/model/keras_model.h5"
        labels_path = REPO_ROOT / "data/model/labels.txt"

        if not model_path.exists():
            self.lbl_model_loaded.setText("⚠️ No model found.\nImport a model from the Train Model tab first.")
            self.lbl_model_loaded.setStyleSheet("font-size:11px; color:#d97706;")
            self._model = None
            self._labels = []
            self._rebuild_mapping_rows([])
            return

        try:
            if not KERAS_AVAILABLE:
                raise RuntimeError("TensorFlow not installed")

            import tensorflow as tf

            class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
                def __init__(self, *args, **kwargs):
                    kwargs.pop("groups", None)
                    super().__init__(*args, **kwargs)

            with tf.keras.utils.custom_object_scope({"DepthwiseConv2D": FixedDepthwiseConv2D}):
                self._model = keras.models.load_model(str(model_path), compile=False)

            self._model_path = str(model_path)
            self.lbl_model_loaded.setText(f"✅ Model loaded:\n{model_path.name}")
            self.lbl_model_loaded.setStyleSheet("font-size:11px; color:#065f46; font-weight:600;")
        except Exception as e:
            self.lbl_model_loaded.setText(f"❌ Failed to load model:\n{e}")
            self.lbl_model_loaded.setStyleSheet("font-size:11px; color:#991b1b;")
            self._model = None
            self._model_path = None

        self._labels = []
        if labels_path.exists():
            raw = labels_path.read_text().strip().splitlines()
            for line in raw:
                parts = line.strip().split(" ", 1)
                label = parts[1].strip() if len(parts) == 2 else parts[0].strip()
                self._labels.append(label)

        if self._labels:
            self.lbl_labels_list.setText("Labels: " + ", ".join(self._labels))
        else:
            self.lbl_labels_list.setText("No labels.txt found.")

        self._rebuild_mapping_rows(self._labels)

    def _on_import_model(self):
        """Let user browse for trained PyTorch model directory"""
        model_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory (data/model/model/)",
            str(REPO_ROOT / "data"),
        )
        
        if not model_dir:
            return
        
        model_dir = Path(model_dir)
        
        # Check if valid HuggingFace model
        config_file = model_dir / "config.json"
        if not config_file.exists():
            QtWidgets.QMessageBox.warning(
                self, "Invalid Model", 
                "Selected directory must contain config.json.\n\nSelect data/model/model/ directory."
            )
            return
        
        # Copy to imported location  
        dest_dir = REPO_ROOT / "data/model_imported"
        try:
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(model_dir, dest_dir)
            self._load_pytorch_model(dest_dir)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import Failed", f"Failed to import:\n{e}")
    
    def _load_pytorch_model(self, model_dir: Path):
        """Load PyTorch ViT model"""
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            from PIL import Image
            
            self._processor = AutoImageProcessor.from_pretrained(str(model_dir))
            self._model = AutoModelForImageClassification.from_pretrained(str(model_dir))
            self._model_path = str(model_dir)
            
            # Get labels from model config
            if hasattr(self._model.config, 'id2label'):
                self._labels = [self._model.config.id2label[i] for i in range(len(self._model.config.id2label))]
            
            if self._labels:
                self.lbl_labels_list.setText("Labels: " + ", ".join(self._labels))
            self._rebuild_mapping_rows(self._labels)
            
            self.lbl_model_loaded.setText(f"✅ PyTorch model loaded:\n{model_dir.name}")
            self.lbl_model_loaded.setStyleSheet("font-size:11px; color:#065f46; font-weight:600;")
        except Exception as e:
            self.lbl_model_loaded.setText(f"❌ Failed to load:\n{e}")
            self.lbl_model_loaded.setStyleSheet("font-size:11px; color:#991b1b;")

    def _toggle_mapping_panel(self):
        visible = self._mapping_content.isVisible()
        self._mapping_content.setVisible(not visible)
        self._btn_mapping_toggle.setText(
            "▼  Gesture → Command Mapping" if not visible else "▶  Gesture → Command Mapping"
        )

    def _rebuild_mapping_rows(self, labels: list):
        while self._mapping_grid.count():
            item = self._mapping_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._mapping_dropdowns.clear()

        if not labels:
            placeholder = QtWidgets.QLabel("Import a model with labels to configure gesture mappings.")
            placeholder.setStyleSheet("font-size:11px; color:#9ca3af; padding:8px;")
            self._mapping_grid.addWidget(placeholder)
            return

        for label in labels:
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet("""
                font-size:12px; font-weight:600; color:#374151;
                background:#e0e7ff; border-radius:6px; padding:4px 10px;
            """)
            lbl.setFixedHeight(30)

            dd = QtWidgets.QComboBox()
            dd.addItems(COMMAND_OPTIONS)
            dd.setMinimumHeight(30)
            default = DEFAULT_MAPPING.get(label, "stop")
            idx = dd.findText(default)
            if idx >= 0:
                dd.setCurrentIndex(idx)

            row.addWidget(lbl, 1)
            row.addWidget(dd, 2)
            self._mapping_dropdowns[label] = dd

            row_widget = QtWidgets.QWidget()
            row_widget.setLayout(row)
            self._mapping_grid.addWidget(row_widget)

    # -----------------------------
    # Robot connect/disconnect
    # -----------------------------
    def _on_robot_connect(self):
        if not AIM_AVAILABLE:
            self._set_conn_status("❌ aim package not found", ok=False)
            self.log_message.emit(
                "❌ Cannot connect: vex/aim.py failed to import.\n"
                f"Reason: {_aim_import_error_msg}\n"
                "Check that phygo/scripts/vex/ folder exists with all files."
            )
            return

        ip = self.in_robot_ip.text().strip()
        self.btn_robot_connect.setEnabled(False)
        self.btn_robot_connect.setText("Connecting…")
        self._set_conn_status("Connecting…", ok=None)
        self.log_message.emit(f"🔌 Connecting to VEX AIM at {ip}...")
        self._connect_ip = ip

        # aim.py uses signal.signal(), must run on main thread
        QtCore.QTimer.singleShot(100, self._do_robot_connect)

    def _do_robot_connect(self):
        ip = self._connect_ip
        try:
            robot = vex_aim.Robot(host=ip)
            self._robot = robot
            self._on_connect_success()
        except Exception as e:
            self._on_connect_failed(str(e))

    @QtCore.pyqtSlot()
    def _on_connect_success(self):
        self._set_conn_status("✅ Connected", ok=True)
        self.btn_robot_connect.setText("Connect")
        self.btn_robot_connect.setEnabled(False)
        self.btn_robot_disconnect.setEnabled(True)
        self.log_message.emit("✅ VEX AIM robot connected!")

    @QtCore.pyqtSlot(str)
    def _on_connect_failed(self, error: str):
        self._set_conn_status("❌ Connection failed", ok=False)
        self.btn_robot_connect.setText("Connect")
        self.btn_robot_connect.setEnabled(True)
        self.log_message.emit(f"❌ Robot connection failed: {error}")

    def _on_robot_disconnect(self):
        self._robot = None
        self._set_conn_status("Disconnected", ok=False)
        self.btn_robot_connect.setEnabled(True)
        self.btn_robot_disconnect.setEnabled(False)
        self.log_message.emit("🔌 Robot disconnected.")

    def _set_conn_status(self, text: str, ok=None):
        self.lbl_conn_status.setText(text)
        if ok is True:
            self.lbl_conn_status.setStyleSheet("""
                font-size:11px; font-weight:600; padding:6px; border-radius:8px;
                background:#d1fae5; color:#065f46;
            """)
        elif ok is False:
            self.lbl_conn_status.setStyleSheet("""
                font-size:11px; font-weight:600; padding:6px; border-radius:8px;
                background:#fee2e2; color:#991b1b;
            """)
        else:
            self.lbl_conn_status.setStyleSheet("""
                font-size:11px; font-weight:600; padding:6px; border-radius:8px;
                background:#fef3c7; color:#92400e;
            """)

    # -----------------------------
    # Record -> predict -> send
    # -----------------------------
    def _on_record_predict(self):
        if self._model is None or self._model_path is None:
            QtWidgets.QMessageBox.warning(self, "No Model", "Please import a model first from the Train Model tab.")
            return
        if self._robot is None:
            QtWidgets.QMessageBox.warning(self, "Not Connected", "Please connect to the VEX AIM robot first.")
            return

        target_seconds = float(self.spin_pred_seconds.value())
        self.btn_record_predict.setEnabled(False)
        self.btn_record_predict.setText("Recording…")
        self._reset_result_display()
        self.log_message.emit("🎙 Starting gesture capture for prediction...")
        self.request_capture.emit(target_seconds)

    def on_capture_complete(self, png_path: str, capture_info: dict):
        self._pending_png = png_path
        self.log_message.emit(f"📸 Capture done. Running model on {png_path}")
        self._run_inference(png_path)

    def _run_inference(self, png_path: str):
        # Determine model type
        model_type = "pytorch" if hasattr(self, '_processor') else "keras"
        
        self._pred_thread = QtCore.QThread()
        self._pred_worker = PredictWorker(
            model=self._model,
            processor=getattr(self, '_processor', None),
            labels=self._labels,
            png_path=png_path,
            model_type=model_type,
        )
        self._pred_worker.moveToThread(self._pred_thread)
        self._pred_thread.started.connect(self._pred_worker.run)
        self._pred_worker.status.connect(self.log_message)
        self._pred_worker.result.connect(self._on_prediction_result)
        self._pred_worker.error.connect(self._on_inference_error)
        self._pred_worker.finished.connect(self._pred_thread.quit)
        self._pred_thread.finished.connect(self._pred_thread.deleteLater)
        self._pred_thread.start()

    @QtCore.pyqtSlot(str, float)
    def _on_prediction_result(self, label: str, confidence: float):
        self.lbl_predicted.setText(label)
        self.lbl_predicted.setStyleSheet("""
            font-size:22px; font-weight:800; color:#111827;
            background:#dbeafe; border-radius:10px; padding:10px;
        """)
        self.lbl_confidence.setText(f"{confidence*100:.1f}%")
        conf_color = "#d1fae5" if confidence >= 0.7 else "#fef3c7" if confidence >= 0.4 else "#fee2e2"
        self.lbl_confidence.setStyleSheet(f"""
            font-size:22px; font-weight:800; color:#111827;
            background:{conf_color}; border-radius:10px; padding:10px;
        """)

        command = "stop"
        if label in self._mapping_dropdowns:
            command = self._mapping_dropdowns[label].currentText()

        self.lbl_cmd_sent.setText(command)
        self.lbl_cmd_sent.setStyleSheet("""
            font-size:14px; font-weight:700; color:#111827;
            background:#ede9fe; border-radius:10px; padding:10px;
        """)

        self.log_message.emit(f"🎯 Gesture: {label} ({confidence*100:.1f}%) → Command: {command}")
        self._send_robot_command(command)

    def _on_inference_error(self, msg: str):
        self.log_message.emit(f"❌ Inference error: {msg}")
        self._restore_record_button()

    def _send_robot_command(self, command: str):
        if self._robot is None:
            self.log_message.emit("⚠️ Robot not connected, skipping command.")
            self._restore_record_button()
            return

        distance_mm = float(self.spin_distance.value())

        self._cmd_thread = QtCore.QThread()
        self._cmd_worker = RobotCommandWorker(
            robot=self._robot,
            command=command,
            distance_mm=distance_mm,
        )
        self._cmd_worker.moveToThread(self._cmd_thread)
        self._cmd_thread.started.connect(self._cmd_worker.run)
        self._cmd_worker.status.connect(self.log_message)
        self._cmd_worker.finished.connect(self._cmd_thread.quit)
        self._cmd_thread.finished.connect(self._cmd_thread.deleteLater)
        self._cmd_thread.finished.connect(self._restore_record_button)
        self._cmd_thread.start()

    @QtCore.pyqtSlot()
    def _restore_record_button(self):
        self.btn_record_predict.setEnabled(True)
        self.btn_record_predict.setText("🎙  Record Gesture & Send to Robot")

    def _reset_result_display(self):
        for lbl in (self.lbl_predicted, self.lbl_confidence, self.lbl_cmd_sent):
            lbl.setText("…")
            lbl.setStyleSheet("""
                font-size:22px; font-weight:800; color:#9ca3af;
                background:#f3f4f6; border-radius:10px; padding:10px;
            """)

# ============================================================
# Batch C: MainWindow (Collect Data tab + wiring) + Entry Point
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SensDS v1.1 GUI")

        # ---- Load configs ----
        configs_dir = get_resource_path("configs")
        chirp_cfg_path = configs_dir / "cfg_simo_chirp.json"
        seq_cfg_path = configs_dir / "cfg_simo_seq.json"

        if not chirp_cfg_path.exists() or not seq_cfg_path.exists():
            raise FileNotFoundError(
                f"Missing config files.\nExpected:\n  {chirp_cfg_path}\n  {seq_cfg_path}\n"
            )

        with open(chirp_cfg_path, "r") as f:
            self.cfg_chirp = json.load(f)
        with open(seq_cfg_path, "r") as f:
            self.cfg_seq = json.load(f)

        # PRF for live micro-doppler
        self.params = {"prf": 1.0 / float(self.cfg_seq["chirp_repetition_time_s"])}

        # Shared radar manager for streaming
        self.manager = InfineonManager.InfineonManager()

        # -----------------------------
        # Root layout
        # -----------------------------
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header
        header = QtWidgets.QHBoxLayout()
        self.lbl_title = QtWidgets.QLabel("SensDS v1.1")
        self.lbl_title.setStyleSheet("font-size:18px; font-weight:600;")
        header.addWidget(self.lbl_title)
        header.addStretch(1)
        self.lbl_status = QtWidgets.QLabel("Disconnected")
        self.lbl_status.setStyleSheet("padding:6px 10px; border-radius:10px; background:#eee;")
        header.addWidget(self.lbl_status)
        root.addLayout(header)

        # Controls row
        controls = QtWidgets.QHBoxLayout()
        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_start = QtWidgets.QPushButton("Start Stream")
        self.btn_stop = QtWidgets.QPushButton("Stop Stream")
        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")
        for b in (self.btn_connect, self.btn_start, self.btn_stop, self.btn_disconnect):
            b.setMinimumHeight(36)
        controls.addWidget(self.btn_connect)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_disconnect)
        controls.addStretch(1)
        root.addLayout(controls)

        # -----------------------------
        # Tabs
        # -----------------------------
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                background: white;
            }
            QTabBar::tab {
                padding: 8px 20px;
                font-size: 13px;
                font-weight: 600;
                border-radius: 8px 8px 0 0;
                margin-right: 4px;
                background: #f3f4f6;
                color: #6b7280;
            }
            QTabBar::tab:selected {
                background: #2563eb;
                color: white;
            }
        """)

        # --- Tab 1: Collect Data ---
        collect_tab = QtWidgets.QWidget()
        collect_lay = QtWidgets.QVBoxLayout(collect_tab)
        collect_lay.setContentsMargins(8, 8, 8, 8)
        collect_lay.setSpacing(10)

        content = QtWidgets.QHBoxLayout()
        content.setSpacing(10)

        left = QtWidgets.QVBoxLayout()
        self.canvas = SpectrogramCanvas(self)
        self.canvas.setMinimumSize(640, 420)
        left.addWidget(self.canvas)
        content.addLayout(left, 3)

        right = QtWidgets.QFrame()
        right.setFrameShape(QtWidgets.QFrame.StyledPanel)
        right.setStyleSheet("""
            QFrame { background: #f7f7f9; border-radius: 14px; padding: 10px; }
            QLabel { font-size: 12px; }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { min-height: 28px; }
            QPushButton { min-height: 34px; }
        """)
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setSpacing(10)

        lbl = QtWidgets.QLabel("Dataset Capture")
        lbl.setStyleSheet("font-size:14px; font-weight:600;")
        rlay.addWidget(lbl)

        form = QtWidgets.QFormLayout()
        self.in_subject = QtWidgets.QLineEdit("michael")
        self.dd_gesture = QtWidgets.QComboBox()
        self.dd_gesture.addItems(["push", "swipe_left", "swipe_right", "swipe_up", "swipe_down", "idle"])

        self.spin_seconds = QtWidgets.QDoubleSpinBox()
        self.spin_seconds.setRange(1.0, 20.0)
        self.spin_seconds.setSingleStep(0.5)
        self.spin_seconds.setValue(6.0)

        self.spin_batch_count = QtWidgets.QSpinBox()
        self.spin_batch_count.setRange(1, 100)
        self.spin_batch_count.setValue(25)
        self.spin_batch_count.setSuffix(" samples")

        self.spin_batch_delay = QtWidgets.QDoubleSpinBox()
        self.spin_batch_delay.setRange(0.0, 30.0)
        self.spin_batch_delay.setSingleStep(0.5)
        self.spin_batch_delay.setValue(3.0)
        self.spin_batch_delay.setSuffix(" sec")

        form.addRow("Subject:", self.in_subject)
        form.addRow("Gesture:", self.dd_gesture)
        form.addRow("Capture (sec):", self.spin_seconds)
        form.addRow("Batch count:", self.spin_batch_count)
        form.addRow("Delay between:", self.spin_batch_delay)
        rlay.addLayout(form)

        self.lbl_countdown = QtWidgets.QLabel("")
        self.lbl_countdown.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_countdown.setStyleSheet("""
            font-size: 72px; font-weight:900;
            padding: 30px; border-radius: 16px;
            background: #1f2937; color: #fbbf24;
            border: 4px solid #fbbf24;
        """)
        self.lbl_countdown.setMinimumHeight(180)
        self.lbl_countdown.hide()
        rlay.addWidget(self.lbl_countdown)

        self.lbl_batch_progress = QtWidgets.QLabel("")
        self.lbl_batch_progress.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_batch_progress.setStyleSheet("""
            font-size: 14px; font-weight:600;
            padding: 8px; border-radius: 8px;
            background: #3b82f6; color: white;
        """)
        self.lbl_batch_progress.hide()
        rlay.addWidget(self.lbl_batch_progress)

        self.btn_record = QtWidgets.QPushButton("Record Sample")
        self.btn_open_folder = QtWidgets.QPushButton("Open Dataset Folder")
        rlay.addWidget(self.btn_record)
        rlay.addWidget(self.btn_open_folder)

        self.lbl_last_saved = QtWidgets.QLabel("Last saved: —")
        self.lbl_last_saved.setWordWrap(True)
        rlay.addWidget(self.lbl_last_saved)

        rlay.addStretch(1)
        content.addWidget(right, 2)
        collect_lay.addLayout(content)

        # Log panel
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(100)
        collect_lay.addWidget(self.log)

        # --- Tab 2: Train Model ---
        self.train_tab = TrainModelTab(self)

        # --- Tab 3: Control Robot ---
        self.control_tab = ControlRobotTab(self.cfg_seq, self.cfg_chirp, self)
        self.control_tab.request_capture.connect(self._on_control_tab_capture_request)
        self.control_tab.log_message.connect(self.append_log)

        self.tabs.addTab(collect_tab, "📡  Collect Data")
        self.tabs.addTab(self.train_tab, "🧠  Train Model")
        self.tabs.addTab(self.control_tab, "🤖  Control Robot")
        root.addWidget(self.tabs)

        # -----------------------------
        # Button states
        # -----------------------------
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_disconnect.setEnabled(False)

        # Signals
        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_disconnect.clicked.connect(self.on_disconnect)
        self.btn_record.clicked.connect(self.on_record_clicked)
        self.btn_open_folder.clicked.connect(self.on_open_dataset_folder)

        # Streaming internals
        self.thread = None
        self.worker = None
        self.frame_count = 0
        self._last_log_t = time.time()

        self.live_window_frames = 64
        from collections import deque
        self.frame_buffer = deque(maxlen=self.live_window_frames)

        self.spect_timer = QtCore.QTimer(self)
        self.spect_timer.setInterval(200)
        self.spect_timer.timeout.connect(self.update_live_spectrogram)

        # Capture internals
        self.capture_thread = None
        self.capture_worker = None

        self.countdown_timer = QtCore.QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._countdown_tick)
        self._countdown_left = 0
        self._manager_was_connected = False

        # Predict mode: if True, after capture completes send result to control_tab instead of saving only
        self._predict_mode = False
        self._pred_target_seconds = 6.0

        # App styling
        self.setStyleSheet("""
            QMainWindow { background: white; }
            QPushButton {
                border-radius: 10px;
                padding: 6px 12px;
                background: #2563eb;
                color: white;
                font-weight: 600;
            }
            QPushButton:disabled { background: #9ca3af; }
            QPlainTextEdit { border-radius: 10px; background: #0b1220; color: #e5e7eb; }
        """)

    # -----------------------------
    # Logging + status
    # -----------------------------
    def append_log(self, msg: str):
        self.log.appendPlainText(msg)

    def set_status_pill(self, text: str, ok: bool = False):
        self.lbl_status.setText(text)
        if ok:
            self.lbl_status.setStyleSheet("padding:6px 10px; border-radius:10px; background:#d1fae5; color:#065f46;")
        else:
            self.lbl_status.setStyleSheet("padding:6px 10px; border-radius:10px; background:#fee2e2; color:#991b1b;")

    # -----------------------------
    # Streaming Controls
    # -----------------------------
    def on_connect(self):
        try:
            self.manager.init_device_fmcw(self.cfg_seq, self.cfg_chirp)
            self.append_log("✅ Connected & configured radar.")
            self.set_status_pill("Connected", ok=True)

            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

            # Auto-start streaming
            self.on_start()

        except Exception as e:
            self.append_log(f"❌ Connect failed: {e}")
            self.set_status_pill("Connect failed", ok=False)

    def on_start(self):
        if self.manager.device is None:
            self.append_log("❌ Not connected. Click Connect first.")
            return
        if self.thread is not None:
            self.append_log("⚠️ Already streaming.")
            return

        self.btn_record.setEnabled(False)
        self.frame_count = 0
        self._last_log_t = time.time()
        self.frame_buffer.clear()

        self.thread = QtCore.QThread()
        self.worker = RadarStreamWorker(self.manager)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.append_log)
        self.worker.error.connect(lambda s: self.append_log(f"❌ {s}"))
        self.worker.frame.connect(self.on_frame)
        self.worker.finished.connect(self.on_stream_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.spect_timer.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_disconnect.setEnabled(False)

    def on_stop(self):
        if self.worker is not None:
            self.worker.stop()

    def on_stream_finished(self):
        self.append_log("✅ Stream worker finished.")
        self.spect_timer.stop()
        self.thread = None
        self.worker = None

        connected = self.manager.device is not None
        self.btn_start.setEnabled(connected)
        self.btn_stop.setEnabled(False)
        self.btn_disconnect.setEnabled(connected)
        self.btn_record.setEnabled(True)

    def on_disconnect(self):
        if self.worker is not None:
            self.on_stop()

        try:
            self.manager.close()
            self.append_log("🔌 Disconnected radar.")
            self.set_status_pill("Disconnected", ok=False)
        except Exception as e:
            self.append_log(f"❌ Disconnect failed: {e}")

        self.btn_connect.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_disconnect.setEnabled(False)
        self.btn_record.setEnabled(True)

    @QtCore.pyqtSlot(object)
    def on_frame(self, frame0):
        self.frame_buffer.append(frame0)
        self.frame_count += 1

        now = time.time()
        if now - self._last_log_t >= 1.0:
            self.append_log(f"📡 frames received: {self.frame_count} (buffer={len(self.frame_buffer)})")
            self._last_log_t = now

    def update_live_spectrogram(self):
        if len(self.frame_buffer) < self.live_window_frames:
            return
        try:
            data = np.asarray(self.frame_buffer)
            spect_db = compute_microdoppler_spectrogram_db(
                data, prf=self.params["prf"], mti=True,
            )
            self.canvas.update_image(spect_db)
        except Exception as e:
            self.spect_timer.stop()
            self.append_log(f"❌ Spectrogram update error: {e}")

    # -----------------------------
    # Dataset Capture
    # -----------------------------
    def on_record_clicked(self):
        if self.thread is not None:
            self.append_log("⚠️ Stop streaming before recording samples.")
            return

        subject = self.in_subject.text().strip()
        gesture = self.dd_gesture.currentText().strip()

        if not subject:
            self.append_log("❌ Please enter a subject name.")
            return

        self.btn_record.setEnabled(False)
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)

        batch_count = self.spin_batch_count.value()
        if batch_count > 1:
            self.lbl_batch_progress.setText(f"Sample 0/{batch_count}")
            self.lbl_batch_progress.show()
            self.append_log(f"🎯 Starting batch capture: {batch_count} samples | subject={subject} gesture={gesture}")
        else:
            self.append_log(f"🎯 Recording sample | subject={subject} gesture={gesture}")

        self._countdown_left = 3
        self.lbl_countdown.setText("Get ready… 3")
        self.lbl_countdown.show()
        self.countdown_timer.start()

    def _countdown_tick(self):
        self._countdown_left -= 1
        if self._countdown_left > 0:
            self.lbl_countdown.setText(f"Get ready… {self._countdown_left}")
            return

        self.countdown_timer.stop()
        self.lbl_countdown.setText("GO!")

        if self._predict_mode:
            QtCore.QTimer.singleShot(250, self._start_predict_capture_worker)
        else:
            QtCore.QTimer.singleShot(250, self._start_batch_capture_worker)

    def _start_batch_capture_worker(self):
        subject = self.in_subject.text().strip()
        gesture = self.dd_gesture.currentText().strip()
        target_seconds = float(self.spin_seconds.value())
        batch_count = self.spin_batch_count.value()
        batch_delay = float(self.spin_batch_delay.value())

        self.lbl_countdown.setText("Capturing…")

        # Important: batch capture worker opens its own radar instance
        self._manager_was_connected = self.manager.device is not None
        if self._manager_was_connected:
            try:
                self.manager.close()
                self.append_log("📴 Temporarily disconnected streaming manager")
            except Exception as e:
                self.append_log(f"⚠️ Error closing manager: {e}")

        self.capture_thread = QtCore.QThread()
        self.capture_worker = BatchCaptureWorker(
            cfg_seq=self.cfg_seq,
            cfg_chirp=self.cfg_chirp,
            subject=subject,
            gesture=gesture,
            batch_count=batch_count,
            target_seconds=target_seconds,
            inter_sample_delay=batch_delay,
            warmup_frames=3,
            frame_retry=10,
            retry_sleep=0.01,
        )
        self.capture_worker.moveToThread(self.capture_thread)

        self.capture_thread.started.connect(self.capture_worker.run)
        self.capture_worker.status.connect(self.append_log)
        self.capture_worker.error.connect(self._capture_error)
        self.capture_worker.sample_finished.connect(self._sample_finished)
        self.capture_worker.delay_countdown.connect(self._show_delay_countdown)
        self.capture_worker.all_finished.connect(self._all_captures_finished)
        self.capture_worker.all_finished.connect(self.capture_thread.quit)
        self.capture_thread.finished.connect(self._on_capture_thread_finished)

        self.capture_thread.start()

    def _capture_error(self, msg: str):
        self.append_log(f"❌ Capture error: {msg}")
        self.lbl_countdown.hide()
        self.lbl_batch_progress.hide()

    def _on_control_tab_capture_request(self, target_seconds: float):
        if self.thread is not None:
            self.append_log("⚠️ Stop streaming before recording a gesture for prediction.")
            self.control_tab._restore_record_button()
            return

        self._predict_mode = True
        self._pred_target_seconds = float(target_seconds)

        self.btn_record.setEnabled(False)
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(False)
        self.btn_start.setEnabled(False)

        self._countdown_left = 3
        self.lbl_countdown.setText("Get ready… 3")
        self.lbl_countdown.show()
        self.countdown_timer.start()

    def _start_predict_capture_worker(self):
        subject = "predict_temp"
        gesture = "unknown"
        target_seconds = self._pred_target_seconds

        self.lbl_countdown.setText("Capturing…")

        self._manager_was_connected = self.manager.device is not None
        if self._manager_was_connected:
            try:
                self.manager.close()
            except Exception:
                pass

        self.capture_thread = QtCore.QThread()
        self.capture_worker = BatchCaptureWorker(
            cfg_seq=self.cfg_seq,
            cfg_chirp=self.cfg_chirp,
            subject=subject,
            gesture=gesture,
            batch_count=1,
            target_seconds=target_seconds,
            inter_sample_delay=0.0,
            warmup_frames=3,
            frame_retry=10,
            retry_sleep=0.01,
        )
        self.capture_worker.moveToThread(self.capture_thread)

        self.capture_thread.started.connect(self.capture_worker.run)
        self.capture_worker.status.connect(self.append_log)
        self.capture_worker.error.connect(self._capture_error)
        self.capture_worker.sample_finished.connect(self._sample_finished)
        self.capture_worker.all_finished.connect(self.capture_thread.quit)
        self.capture_thread.finished.connect(self._on_capture_thread_finished)

        self.capture_thread.start()

    def _show_delay_countdown(self, seconds_left: int):
        self.lbl_countdown.setText(f"Next sample in… {seconds_left}")
        self.lbl_countdown.show()

    def _sample_finished(self, png_path: str, raw_path: str, capture_seconds: float, frames: int, capture_info: dict):
        sample_num = capture_info["sample_num"]
        total_samples = capture_info["total_samples"]

        if total_samples > 1:
            self.lbl_batch_progress.setText(f"Sample {sample_num}/{total_samples}")

        self.append_log(f"📊 Generating spectrogram for sample {sample_num}...")

        try:
            spectrogram(
                capture_info["data"],
                duration=capture_info["duration"],
                prf=capture_info["prf"],
                mti=True,
                is_save=True,
                savename=png_path,
            )
            self.append_log(f"✅ [{sample_num}/{total_samples}] Saved: {png_path}")
        except Exception as e:
            self.append_log(f"⚠️ Spectrogram generation failed: {e}")

        self.append_log(f"⏱ Capture time: {capture_seconds:.2f}s | frames={frames}")
        self.lbl_last_saved.setText(f"Last saved: {png_path}")

        if self._predict_mode:
            self.control_tab.on_capture_complete(png_path, capture_info)
        else:
            self.train_tab.refresh_checklist()

    def _all_captures_finished(self):
        batch_count = self.spin_batch_count.value()
        if batch_count > 1:
            self.append_log(f"✅ Batch capture complete! {batch_count} samples saved.")
        else:
            self.append_log("✅ Capture complete!")

    def _on_capture_thread_finished(self):
        self.append_log("🧹 Capture thread cleaned up")
        self.lbl_countdown.hide()
        self.lbl_batch_progress.hide()

        if self._manager_was_connected:
            try:
                self.manager.init_device_fmcw(self.cfg_seq, self.cfg_chirp)
                self.append_log("🔄 Reconnected streaming manager")
                self.set_status_pill("Connected", ok=True)
            except Exception as e:
                self.append_log(f"⚠️ Failed to reconnect: {e}")
                self.set_status_pill("Reconnect failed", ok=False)
                self._manager_was_connected = False

        if not self._predict_mode:
            self.train_tab.refresh_checklist()

        self._predict_mode = False
        self._restore_buttons_after_capture()

    def _restore_buttons_after_capture(self):
        connected = self.manager.device is not None
        self.btn_connect.setEnabled(not connected)
        self.btn_disconnect.setEnabled(connected)
        self.btn_start.setEnabled(connected)
        self.btn_stop.setEnabled(False)
        self.btn_record.setEnabled(True)
        self.capture_thread = None
        self.capture_worker = None

    def on_open_dataset_folder(self):
        folder = REPO_ROOT / "data/train/spectrogram"
        folder.mkdir(parents=True, exist_ok=True)
        self.append_log(f"📂 Opening folder: {folder}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


# -----------------------------
# Entry Point
# -----------------------------
def _main():
    print("[SensDS v1.1] Starting Qt application...")
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(1150, 800)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(_main())