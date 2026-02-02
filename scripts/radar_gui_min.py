import sys
import time
from collections import deque

import numpy as np
from PyQt5 import QtCore, QtWidgets

# Matplotlib canvas for embedding in PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# import your manager
import InfineonManager

# import the live spectrogram compute helper we added
from processing_utils import compute_microdoppler_spectrogram_db


APP_QSS = """
/* ===== Base ===== */
QMainWindow {
    background-color: #0b1220;
}
QWidget {
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Arial;
    font-size: 13px;
}

/* ===== “Card” containers ===== */
QFrame#Card {
    background-color: #0f1b2d;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
}

/* ===== Buttons ===== */
QPushButton {
    background-color: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 10px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.16);
}
QPushButton:pressed {
    background-color: rgba(255,255,255,0.14);
}
QPushButton:disabled {
    color: rgba(230,237,243,0.35);
    background-color: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
}

/* Accent buttons by objectName */
QPushButton#Primary {
    background-color: #2eaadc;
    border: 1px solid rgba(46,170,220,0.5);
    color: #061018;
}
QPushButton#Primary:hover {
    background-color: #3bb7e6;
}
QPushButton#Danger {
    background-color: rgba(255, 86, 86, 0.16);
    border: 1px solid rgba(255, 86, 86, 0.35);
}
QPushButton#Danger:hover {
    background-color: rgba(255, 86, 86, 0.24);
    border: 1px solid rgba(255, 86, 86, 0.45);
}

/* ===== Text area ===== */
QPlainTextEdit {
    background-color: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    padding: 10px;
}

/* ===== Labels ===== */
QLabel#Title {
    font-size: 18px;
    font-weight: 800;
}
QLabel#Subtitle {
    color: rgba(230,237,243,0.70);
}
"""


# Matplotlib-friendly colors (tuples, not CSS rgba strings)
TEXT = (230 / 255, 237 / 255, 243 / 255)                 # #e6edf3
TEXT_FAINT = (230 / 255, 237 / 255, 243 / 255, 0.75)
TEXT_TICKS = (230 / 255, 237 / 255, 243 / 255, 0.65)


def set_badge(label: QtWidgets.QLabel, text: str, kind: str):
    """
    kind: 'disconnected' | 'connected' | 'streaming' | 'error'
    """
    styles = {
        "disconnected": ("Disconnected", "rgba(255,255,255,0.10)", "rgba(255,255,255,0.18)"),
        "connected": ("Connected", "rgba(86, 255, 168, 0.14)", "rgba(86, 255, 168, 0.35)"),
        "streaming": ("Streaming", "rgba(46,170,220,0.18)", "rgba(46,170,220,0.45)"),
        "error": ("Error", "rgba(255, 86, 86, 0.18)", "rgba(255, 86, 86, 0.45)"),
    }
    _, bg, br = styles.get(kind, (text, "rgba(255,255,255,0.10)", "rgba(255,255,255,0.18)"))
    label.setText(text)
    label.setStyleSheet(
        f"""
        QLabel {{
            padding: 6px 10px;
            border-radius: 999px;
            background-color: {bg};
            border: 1px solid {br};
            font-weight: 700;
        }}
        """
    )


class RadarStreamWorker(QtCore.QObject):
    frame = QtCore.pyqtSignal(object)      # emits one frame (numpy array)
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


class SpectrogramCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), dpi=110)
        self.ax = self.fig.add_subplot(111)

        # Dark figure background to match UI
        self.fig.patch.set_facecolor("#0f1b2d")
        self.ax.set_facecolor("#0f1b2d")

        super().__init__(self.fig)
        self.setParent(parent)

        self.ax.set_title("Live Micro-Doppler Spectrogram", color=TEXT, pad=10, fontsize=13, fontweight="bold")
        self.ax.set_xlabel("Time bins", color=TEXT_FAINT)
        self.ax.set_ylabel("Frequency bins", color=TEXT_FAINT)
        self.ax.tick_params(colors=TEXT_TICKS)

        self.im = None
        self._cbar = None

    def update_image(self, spect_db: np.ndarray):
        # Recreate image if shape changes (prevents blank right side)
        if self.im is None or self.im.get_array().shape != spect_db.shape:
            self.ax.clear()
            self.ax.set_facecolor("#0f1b2d")
            self.ax.set_title("Live Micro-Doppler Spectrogram", color=TEXT, pad=10, fontsize=13, fontweight="bold")
            self.ax.set_xlabel("Time bins", color=TEXT_FAINT)
            self.ax.set_ylabel("Frequency bins", color=TEXT_FAINT)
            self.ax.tick_params(colors=TEXT_TICKS)

            self.im = self.ax.imshow(
                spect_db,
                aspect="auto",
                origin="lower",
                vmin=-20,
                vmax=0,
                cmap="jet",
            )

            # Remove existing colorbar (if any) then add new
            if self._cbar is not None:
                self._cbar.remove()
                self._cbar = None
            self._cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
            self._cbar.ax.tick_params(colors=TEXT_TICKS)
        else:
            self.im.set_data(spect_db)

        self.draw_idle()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Infineon Radar Stream")

        # Apply modern theme
        self.setStyleSheet(APP_QSS)

        self.manager = InfineonManager.InfineonManager()

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(14)

        # ===== Header Card =====
        header_card = QtWidgets.QFrame()
        header_card.setObjectName("Card")
        header_layout = QtWidgets.QHBoxLayout(header_card)
        header_layout.setContentsMargins(16, 14, 16, 14)

        title_col = QtWidgets.QVBoxLayout()
        lbl_title = QtWidgets.QLabel("SensDS Radar Live Spectrogram v0.1")
        lbl_title.setObjectName("Title")
        lbl_subtitle = QtWidgets.QLabel("Connect → Start Streaming → Observe motion in real time")
        lbl_subtitle.setObjectName("Subtitle")
        title_col.addWidget(lbl_title)
        title_col.addWidget(lbl_subtitle)

        header_layout.addLayout(title_col)
        header_layout.addStretch(1)

        self.badge = QtWidgets.QLabel()
        set_badge(self.badge, "Disconnected", "disconnected")
        header_layout.addWidget(self.badge)

        root.addWidget(header_card)

        # ===== Controls Card =====
        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("Card")
        controls_layout = QtWidgets.QHBoxLayout(controls_card)
        controls_layout.setContentsMargins(16, 14, 16, 14)
        controls_layout.setSpacing(10)

        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.setObjectName("Primary")

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.setObjectName("Primary")

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setObjectName("Danger")

        self.btn_disconnect = QtWidgets.QPushButton("Disconnect")

        controls_layout.addWidget(self.btn_connect)
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.btn_disconnect)

        root.addWidget(controls_card)

        # ===== Plot Card =====
        plot_card = QtWidgets.QFrame()
        plot_card.setObjectName("Card")
        plot_layout = QtWidgets.QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(12, 12, 12, 12)

        self.canvas = SpectrogramCanvas(self)
        plot_layout.addWidget(self.canvas)

        root.addWidget(plot_card, stretch=1)

        # ===== Log Card =====
        log_card = QtWidgets.QFrame()
        log_card.setObjectName("Card")
        log_layout = QtWidgets.QVBoxLayout(log_card)
        log_layout.setContentsMargins(12, 12, 12, 12)
        log_layout.setSpacing(8)

        log_title = QtWidgets.QLabel("Logs")
        log_title.setStyleSheet("font-weight: 800;")
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)

        log_layout.addWidget(log_title)
        log_layout.addWidget(self.log)

        root.addWidget(log_card)

        # Initial states
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_disconnect.setEnabled(False)

        # signals
        self.btn_connect.clicked.connect(self.on_connect)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_disconnect.clicked.connect(self.on_disconnect)

        # thread/worker
        self.thread = None
        self.worker = None

        # counters
        self.frame_count = 0
        self._last_log_t = time.time()

        # Config
        self.cfg_seq = {
            "frame_repetition_time_s": 30.303e-3,
            "chirp_repetition_time_s": 300e-6,
            "num_chirps": 32,
            "tdm_mimo": False,
        }
        self.cfg_chirp = {
            "start_frequency_Hz": 58.5e9,
            "end_frequency_Hz": 62.5e9,
            "sample_rate_Hz": 2e6,
            "num_samples": 64,
            "rx_mask": 7,
            "tx_mask": 1,
            "tx_power_level": 31,
            "lp_cutoff_Hz": 500000,
            "hp_cutoff_Hz": 80000,
            "if_gain_dB": 30,
        }

        self.params = {"prf": 1.0 / float(self.cfg_seq["chirp_repetition_time_s"])}

        # Live buffer + update timer
        self.live_window_frames = 64
        self.frame_buffer = deque(maxlen=self.live_window_frames)

        self.spect_timer = QtCore.QTimer(self)
        self.spect_timer.setInterval(200)
        self.spect_timer.timeout.connect(self.update_live_spectrogram)

    def append_log(self, msg: str):
        self.log.appendPlainText(msg)

    def on_connect(self):
        try:
            self.manager.init_device_fmcw(self.cfg_seq, self.cfg_chirp)
            self.append_log("✅ Connected & configured radar.")
            set_badge(self.badge, "Connected", "connected")

            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
        except Exception as e:
            set_badge(self.badge, "Error", "error")
            self.append_log(f"❌ Connect failed: {e}")

    def on_start(self):
        if self.manager.device is None:
            self.append_log("❌ Not connected. Click Connect first.")
            return
        if self.thread is not None:
            self.append_log("⚠️ Already streaming.")
            return

        set_badge(self.badge, "Streaming", "streaming")

        self.frame_count = 0
        self._last_log_t = time.time()
        self.frame_buffer.clear()

        self.thread = QtCore.QThread()
        self.worker = RadarStreamWorker(self.manager)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.status.connect(self.append_log)
        self.worker.error.connect(self.on_worker_error)
        self.worker.frame.connect(self.on_frame)
        self.worker.finished.connect(self.on_stream_finished)

        self.worker.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.spect_timer.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_connect.setEnabled(False)
        self.btn_disconnect.setEnabled(False)

    def on_stop(self):
        if self.worker is not None:
            self.worker.stop()

    def on_worker_error(self, s: str):
        set_badge(self.badge, "Error", "error")
        self.append_log(f"❌ {s}")

    def on_stream_finished(self):
        self.append_log("✅ Stream worker finished.")
        self.spect_timer.stop()

        self.thread = None
        self.worker = None

        connected = self.manager.device is not None
        set_badge(self.badge, "Connected" if connected else "Disconnected", "connected" if connected else "disconnected")

        self.btn_start.setEnabled(connected)
        self.btn_stop.setEnabled(False)
        self.btn_disconnect.setEnabled(connected)

    def on_disconnect(self):
        if self.worker is not None:
            self.on_stop()

        try:
            self.manager.close()
            self.append_log("🔌 Disconnected radar.")
            set_badge(self.badge, "Disconnected", "disconnected")
        except Exception as e:
            set_badge(self.badge, "Error", "error")
            self.append_log(f"❌ Disconnect failed: {e}")

        self.btn_connect.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_disconnect.setEnabled(False)

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
                data,
                prf=self.params["prf"],
                mti=True,
            )
            self.canvas.update_image(spect_db)
        except Exception as e:
            self.spect_timer.stop()
            set_badge(self.badge, "Error", "error")
            self.append_log(f"❌ Spectrogram update error: {e}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.resize(980, 760)
    w.show()
    sys.exit(app.exec())