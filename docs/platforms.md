# Platforms

What runs where, how FlipTuned picks a backend, and how to fit training into the memory you have.
Fixes for common problems are in `docs/troubleshooting.md`.

**What has actually been run:** Windows 11 on CPU (`hf_peft`). The CUDA and MLX paths are written
and unit-tested with fakes, but have **never been run** on real hardware. Commands below marked
*(macOS)* or *(NVIDIA)* have not been run by the project.

## Backend selection

`fliptuned doctor` shows what was detected, which backend will be used and why, and
`fliptuned doctor --suggest-install` prints the `uv sync` command for this machine:

```text
uv run fliptuned doctor --home ./.dev-home
uv run fliptuned doctor --suggest-install
```

| Machine | Backend | Install |
|---|---|---|
| macOS 14+ on Apple Silicon | `mlx` (CPU `hf_peft` for tests) | `uv sync --extra mlx --extra cpu --extra web` |
| Windows/Linux + NVIDIA (driver ≥ 580, Turing or newer) | `hf_peft` on `cuda`, 4-bit QLoRA | `uv sync --extra cuda --extra qlora --extra web` |
| Anything else | `hf_peft` on `cpu` (slow; ≤ 1B params by default) | `uv sync --extra cpu --extra web` |

Add `--extra dev` for development and tests. With `backend: auto` (the default) the order is:

1. A backend or device set in the config (or in the UI's Advanced settings) is checked against this
   machine; if it isn't available, the error says why and which install command would enable it.
2. macOS on Apple Silicon with `mlx` installed → `mlx`.
3. PyTorch sees a CUDA GPU → `hf_peft` on `cuda`: 4-bit QLoRA if bitsandbytes is installed,
   otherwise bf16 if the GPU supports it, otherwise fp16.
4. Otherwise `hf_peft` on `cpu` in fp32, with a warning.

Adapters belong to the backend that trained them: an `mlx` adapter can't be loaded by `hf_peft` and
the other way round. Evaluation uses the run's backend for both models, so the comparison is fair.

## NVIDIA (CUDA)

- The `cuda` extra uses PyTorch 2.14.1 built for CUDA 13.0 (`cu130`), which needs an NVIDIA driver
  **580 or newer** and a GPU with compute capability **7.5+** (RTX 20-series / T4 and newer).
  `nvidia-smi` *(NVIDIA)* shows the driver version in its header.
- Check which PyTorch build is installed (`+cpu` means the CPU build; `True` means a usable GPU):

  ```text
  uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```

  If `doctor` says "NVIDIA GPU found but PyTorch is CPU-only", run the `cuda` install command. The
  `cuda` and `cpu` extras exclude each other; `uv sync` swaps the PyTorch build.
- Older GPUs (GTX 10-series, V100) aren't supported by CUDA 13. They could use the `cu126` build
  (driver ≥ 560) by changing the `pytorch-cu130` index URL in `pyproject.toml` to
  `https://download.pytorch.org/whl/cu126`. This path is **not run** or tested by FlipTuned.

### bitsandbytes (4-bit QLoRA)

| OS | bitsandbytes | What FlipTuned does |
|---|---|---|
| Linux + NVIDIA | official wheels (CUDA 13 wheels need sm75+) | QLoRA when the `qlora` extra is installed |
| Windows + NVIDIA | official wheels | QLoRA when the `qlora` extra is installed |
| macOS | not used | `mlx` quantizes on its own (4-bit `mlx-community` models) |
| CPU only | not used | fp32 LoRA |

`doctor` lists the installed bitsandbytes version (or "not installed"). Without it, CUDA training
uses bf16/fp16 LoRA, which needs considerably more VRAM for the same model; `models list` shows
the catalog's estimate for each precision.

## macOS (Apple Silicon)

- Apple Silicon only; Intel Macs have no current PyTorch build.
- **Rosetta check.** `doctor`'s Python line says `native` or `emulated!`. Emulated means Python is
  running under Rosetta (x86_64), and MLX can't be used. Install a native arm64 Python from a
  native terminal and recreate the environment *(macOS)*:

  ```text
  uv python install 3.12
  uv sync --extra mlx --extra cpu --extra web
  ```

  `uv run python -c "import platform; print(platform.machine())"` prints `arm64` for a native Python
  on a Mac (`AMD64` or `x86_64` elsewhere).

## Windows

- **Long paths:** if `doctor` warns that long paths are disabled, enable them (admin PowerShell),
  then sign out and back in *(not run by the project: it changes a system setting)*:
  `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force`
- **Model cache disk:** the Hugging Face cache defaults to `%USERPROFILE%\.cache\huggingface`. To keep
  it on a bigger drive, set `HF_HOME`, e.g. `setx HF_HOME K:\hf-cache`, and open a new terminal.
  Without Developer Mode, the cache stores copies instead of symlinks and uses more space.
- Background workers are started detached and break away from the terminal's job object, so
  closing the terminal doesn't stop a training job. Keep-awake uses `SetThreadExecutionState`.
- Antivirus scanning can make the first model load and `uv sync` slow; see
  `docs/troubleshooting.md`.

## Linux

- NVIDIA works as on Windows (same `cuda` extra and driver requirement).
- Keep-awake uses `systemd-inhibit` when it's available; otherwise the run monitor shows a warning
  and you should stop the machine from sleeping yourself.

## Memory tips per device

The Models screen and `models list` label each model `fits`, `tight` or `won't fit` from the
catalog's **estimate** of peak training memory against this machine's memory pool. The run monitor
shows the measured peak for a run. When training runs out of memory, change these in the training
config (Advanced settings in the UI), roughly in this order:

| Device | Memory pool | What to try |
|---|---|---|
| NVIDIA | free VRAM | install the `qlora` extra (`precision: qlora_4bit`); `batch_size: 1` with more `grad_accumulation_steps`; keep `gradient_checkpointing: true`; lower `max_seq_length` (and rebuild the dataset with the same value); a smaller model; close other programs using the GPU |
| Apple Silicon | unified memory minus at least 4 GB for macOS | a 4-bit `mlx-community` model; `batch_size: 1`; `gradient_checkpointing: true`; `max_seq_length: 1024`; fewer LoRA layers (`training.mlx.num_layers`, default 8); close memory-heavy apps |
| CPU | system RAM minus room for the OS | stay at or under 1B parameters (`training.hf_peft.cpu_max_params_b`); `batch_size: 1`; lower `max_seq_length`; the `quick` preset |

A lower LoRA `rank` saves a little memory, much less than the steps above. Lowering
`max_seq_length` quarantines longer examples at build time, so check the dataset statistics
afterwards.
