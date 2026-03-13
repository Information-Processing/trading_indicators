# Chatbot folder – what you need

All paths in the notebook point to **this folder only**. Run the notebook from this directory on the PYNQ (so `os.getcwd()` is this folder).

## Required files in this folder

| File | Purpose |
|------|--------|
| **base.bit** | FPGA overlay bitstream (you have this) |
| **base.hwh** | Hardware metadata; PYNQ finds it next to `base.bit` (you have this) |
| **new_audio.py** | Python driver for the audio IP (included here) |
| **libaudio.so** | C audio library; must sit **next to** `new_audio.py` (same folder). Get it from your PYNQ image, e.g. copy from `/usr/local/lib/python3.x/dist-packages/pynq/lib/_pynq/_audio/libaudio.so` or from the PYNQ repo build. |

## Optional / reference (no path references)

- `base.tcl` – Vivado block design script
- `sigma_delta.v`, `audio_direct_v1_1.v`, `audio_direct_v1_1_S00_AXI.v` – RTL for the audio IP

## If something is missing

- **libaudio.so**: Copy it from the PYNQ board’s existing PYNQ install (path above) into this folder. Do not use the C++/Makefile from `drivers/pcm_driver`; only the Python wrapper (`new_audio.py`) is needed here.
- **base.bit / base.hwh**: Must be the overlay that includes the modified `audio_direct` IP and PDM→PCM path. If you built from `base.tcl`, the generated bitstream and `.hwh` go here.

## Running on PYNQ

1. Put this whole folder on the PYNQ (e.g. under `jupyter_notebooks/`).
2. Open Jupyter from that directory so the working directory is this folder.
3. Run the notebook; the overlay loads `base.bit` from this folder.
