# Getting Started

## Clone PhyGO Project

`git clone https://github.com/htil/phygo.git`

---

## Navigate to Repository

`cd phygo`

---

## Create Virtual Environment

`python -m venv venv`

Helpful tutorial (Windows):  
Python Tutorial: VENV (Windows) – How to Use Virtual Environments  
https://www.youtube.com/watch?v=APOPm01BVrk

---

## Activate Environment

### Windows (PowerShell)

`.\venv\Scripts\activate`

### macOS / Linux

`source venv/bin/activate`

---

## Upgrade pip

`python -m pip install --upgrade pip`

---

## Install Libraries

If a `requirements.txt` file is provided:

`python -m pip install -r requirements.txt`

---

## Install Infineon Radar SDK (ifxradarsdk)

Install the wheel provided with the project or by your lab:

`python -m pip install path/to/ifxradarsdk-*.whl`

Example:

`python -m pip install inf_wheel/ifxradarsdk-3.6.4+4b4a6245-py3-none-macosx_10_14_universal2.whl`

Verify installation:

`python -c "import ifxradarsdk; print('ifxradarsdk OK')"`

---

## (Optional) Update Requirements File

After confirming everything runs correctly:

`python -m pip freeze > requirements.txt`

---

## Navigate to Scripts Directory

`cd scripts`

---

## Run Live Radar GUI

`python radar_gui_min.py`

---

## Basic Usage

1. Connect the Infineon radar via USB  
2. Click **Connect**  
3. Click **Start** to begin streaming  
4. Move a hand or object in front of the radar  
5. Observe the live micro-Doppler spectrogram  
6. Click **Stop** to stop streaming  
7. Click **Disconnect** when finished  

---

## Other Helpful Commands

### Find Python Installation Path (Windows)

`Get-Command python | Select-Object Source`

---

### Addressing PowerShell Unauthorized Script Error (Windows)

If activation fails due to execution policy:

`Set-ExecutionPolicy Unrestricted -Scope Process`

---

### Install Git on Windows

`winget install --id Git.Git -e --source winget`

---

## Notes

- The GUI uses PyQt5 + Matplotlib (no Streamlit)
- Live spectrogram updates use a rolling frame buffer
- Performance can be tuned via `live_window_frames` and the spectrogram update interval