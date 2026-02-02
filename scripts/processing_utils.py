# processing_utils.py
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import butter, lfilter


def stft(data, window, nfft, shift):
    """
    Compute the Short-Time Fourier Transform (STFT) of the input data.

    Args:
        data (np.ndarray): Input 1D signal data.
        window (int): Length of each segment/window.
        nfft (int): Number of FFT points.
        shift (int): Number of samples to shift for the next segment.

    Returns:
        np.ndarray: STFT of the input data with shape (nfft, n_segments).
    """
    n = (len(data) - window - 1) // shift
    out1 = np.zeros((nfft, n), dtype=complex)

    for i in range(n):
        tmp1 = data[i * shift : i * shift + window].T
        tmp2 = np.hanning(window)
        tmp3 = tmp1 * tmp2
        tmp = np.fft.fft(tmp3, n=nfft)
        out1[:, i] = tmp

    return out1


def plot_spectrogram(spect, duration, prf, savename=None):
    """
    Plot and optionally save the spectrogram.

    IMPORTANT: This function is kept aligned with your current output styling:
      - dB scaling normalized by max
      - 'jet' colormap
      - vmin=-20 (via Normalize)
      - when saving, axes/ticks are removed and saved tightly with transparency

    Args:
        spect (np.ndarray): Spectrogram magnitude (linear) or already-shifted magnitude.
        duration (float): Duration of the signal in seconds.
        prf (float): Pulse repetition frequency in Hz.
        savename (str, optional): If provided, the path to save the figure.
    """
    fig = plt.figure(frameon=True)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    # Note: original code did not add ax to fig; we keep behavior but use plt.* calls.
    # This preserves the "exact look" you already had.

    maxval = np.max(spect) if np.max(spect) != 0 else 1.0
    norm = colors.Normalize(vmin=-20, vmax=None, clip=True)

    im = plt.imshow(
        20 * np.log10((abs(spect) / maxval)),
        cmap="jet",
        norm=norm,
        aspect="auto",
        extent=[0, duration, -prf / 2, prf / 2],
    )
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Micro-Doppler Spectrogram")

    if savename is not None:
        print(f"Saving spectrogram to {savename}")
        plt.axis("off")
        plt.tick_params(
            axis="both",
            left=False,
            top=False,
            right=False,
            bottom=False,
            labelleft=False,
            labeltop=False,
            labelright=False,
            labelbottom=False,
        )
        im.get_figure().gca().set_title("")
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())  # kept for parity
        plt.savefig(savename, bbox_inches="tight", transparent=True, pad_inches=0)
        plt.close()


def spectrogram(data, duration, prf, mti=False, is_save=None, savename=None):
    """
    Generate and plot the spectrogram from radar data (batch/offline behavior).

    Args:
        data (np.ndarray): Raw radar data with shape (n_frame, n_rx, n_chirp, n_sample).
        duration (float): Duration of the signal in seconds.
        prf (float): Pulse repetition frequency in Hz.
        mti (bool, optional): Whether to apply MTI filtering. Defaults to False.
        is_save (bool, optional): Kept for API compatibility. Saving is controlled by savename.
        savename (str, optional): If provided, the path to save the figure.

    Returns:
        None
    """
    start_time = time.time()

    # Match your original pipeline exactly
    data = data[:, 0, :, :]
    data = np.transpose(data, (2, 1, 0))
    num_samples = data.shape[0]
    num_chirps = data.shape[1] * data.shape[2]
    data = data.reshape((num_samples, num_chirps), order="F")

    range_fft = np.fft.fft(data, 2 * num_samples, axis=0)[num_samples:] / num_samples
    range_fft -= np.expand_dims(np.mean(range_fft, 1), 1)

    if mti:
        b, a = butter(1, 0.01, "high", output="ba")
        rngpro = np.zeros_like(range_fft)
        for r in range(rngpro.shape[0]):
            rngpro[r, :] = lfilter(b, a, range_fft[r, :])
    else:
        rngpro = range_fft

    rBin = np.arange(num_samples // 2, num_samples - 1)
    nfft = 2**10
    window = 256
    noverlap = 200
    shift = window - noverlap

    vec = np.sum(rngpro[rBin, :], 0)

    spect = stft(vec, window, nfft, shift)
    spect = np.abs(np.fft.fftshift(spect, 0))

    print(f"Generated spectrogram in {time.time() - start_time:.2f} seconds")

    plot_spectrogram(spect, duration, prf, savename=savename)


def compute_microdoppler_spectrogram_db(data, prf, mti=True):
    """
    NEW: Compute the micro-Doppler spectrogram matrix (in dB) without plotting/saving.
    This is what we’ll use for LIVE display in the GUI.

    It uses the same math + parameters as spectrogram() so the live display matches
    your saved images as closely as possible.

    Args:
        data (np.ndarray): Raw radar data with shape (n_frame, n_rx, n_chirp, n_sample).
        prf (float): Pulse repetition frequency in Hz.
        mti (bool): Apply MTI filtering.

    Returns:
        np.ndarray: spectrogram in dB, shape (nfft, n_time_bins)
    """
    # Same preprocessing
    data = data[:, 0, :, :]
    data = np.transpose(data, (2, 1, 0))
    num_samples = data.shape[0]
    num_chirps = data.shape[1] * data.shape[2]
    data = data.reshape((num_samples, num_chirps), order="F")

    range_fft = np.fft.fft(data, 2 * num_samples, axis=0)[num_samples:] / num_samples
    range_fft -= np.expand_dims(np.mean(range_fft, 1), 1)

    if mti:
        b, a = butter(1, 0.01, "high", output="ba")
        rngpro = np.zeros_like(range_fft)
        for r in range(rngpro.shape[0]):
            rngpro[r, :] = lfilter(b, a, range_fft[r, :])
    else:
        rngpro = range_fft

    rBin = np.arange(num_samples // 2, num_samples - 1)
    nfft = 2**10
    window = 256
    noverlap = 200
    shift = window - noverlap

    vec = np.sum(rngpro[rBin, :], 0)

    spect = stft(vec, window, nfft, shift)
    spect = np.abs(np.fft.fftshift(spect, 0))

    maxval = np.max(spect) if np.max(spect) != 0 else 1.0
    spect_db = 20 * np.log10(np.abs(spect) / maxval)

    return spect_db