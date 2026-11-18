# Troubleshooting

Every error FlipTuned shows says what went wrong, then what to do; the UI keeps technical details
behind "Details", and the CLI shows them with `--verbose`. This page covers the problems that need
more than that. Commands marked *(macOS/Linux)* or *(admin)* have not been run by the project.

Start with the environment check; it never opens your data:

```text
uv run fliptuned doctor --home ./.dev-home
```

(Leave out `--home ./.dev-home` for your real app home, `~/.fliptuned`.)

## The web UI

**Port in use.** `fliptuned serve` tries the port from Settings (default 8765) and, if something
else is listening there, the next free one of the following 19; the printed link always has the
port actually used. To choose another starting port:

```text
uv run fliptuned serve --home ./.dev-home --port 8780 --no-browser
```

To see what holds a port: `Get-NetTCPConnection -LocalPort 8765` (Windows PowerShell),
`lsof -i :8765` *(macOS/Linux)*.

**"This page needs the access token" or "Missing or wrong access token".** Every launch of `serve`
makes a new token, carried in the link it prints (`…/#token=…`). A bookmark, an old tab, or a tab
from before a restart has the old token. Open the link the command printed most recently. The
token lives only in that browser tab, so a new tab needs the link too.

**"The web app needs the 'web' extra".** `fliptuned serve` was started without the web
packages. Run `uv run fliptuned doctor --suggest-install` and use the command it prints.

**A placeholder page instead of the app.** The UI hasn't been built. Build it once (Node 22):

```text
npm --prefix web ci
npm --prefix web run build
```

**"This server only answers requests for 127.0.0.1."** The server answers only `127.0.0.1:<port>` and `localhost:<port>`. Use
the printed link, not the machine's name or network address; other computers can't reach it.

## Jobs

**Interrupted jobs.** A job whose process ended unexpectedly (the computer slept or restarted, the
process was killed, the system ran out of memory) is marked `interrupted` the next time FlipTuned
looks: when `serve` starts, or on `jobs list`. Training keeps a checkpoint at every evaluation, so
an interrupted, failed or cancelled run can continue from its last checkpoint:

```text
uv run fliptuned jobs list --home ./.dev-home
uv run fliptuned jobs resume <run_id> --home ./.dev-home
```

In the UI: the run monitor's **Resume** button. Resume is refused when the run has no checkpoint
yet (start it again instead) or another job holds the model.

**"Another job is using the model: …".** Only one job that loads a model (training, evaluation, chat,
merge) runs at a time. Wait for it, cancel it (`jobs cancel <job_id>`, or Cancel in the run
monitor), or close the chat that holds it (Chat → "Close and release the model"; an idle chat
releases the model by itself after the idle timeout).

**"… is not downloaded".** Model commands work offline. Allow the download once: `--allow-download`
on the command, or the download option in the UI (Models, Dataset, Preflight).

**Review decisions after changing a source file.** Decisions belong to a record's content.
After `ingest --overwrite`, rows that only moved (because rows were inserted or deleted above
them) keep their decision; a row whose question or answer changed goes back to review.

## Out of memory

What it looks like and what to do, per device. Settings are in the training config (Advanced
settings in the UI); `docs/platforms.md` has the full list of memory tips.

A job that runs out of memory fails with "The job ran out of memory." and a hint; chat shows
"The model ran out of memory while answering." and keeps the session open.

| Device | What you see | What to do |
|---|---|---|
| NVIDIA | the job fails; Details mention `OutOfMemoryError` / "CUDA out of memory" | install the `qlora` extra, `batch_size: 1` with more `grad_accumulation_steps`, lower `max_seq_length`, a smaller model; close other programs using the GPU |
| Apple Silicon | the job fails with a Metal / memory error, or macOS slows to a crawl while swapping | a 4-bit `mlx-community` model, fewer `training.mlx.num_layers`, `max_seq_length: 1024`, `batch_size: 1`; close memory-heavy apps |
| CPU | the job fails with a memory error, or the system kills it and it shows as `interrupted` | a model of 1B parameters or less, `batch_size: 1`, lower `max_seq_length`; close other programs |

The run monitor shows the run's peak memory against the machine's total, which helps to judge how
much to cut. Preflight's memory check is an estimate, never a guarantee.

## Windows

**Long paths.** Deep Hugging Face cache paths can pass Windows' 260-character limit and fail with
"file not found" errors. `doctor` warns when long paths are disabled. Enable them *(admin)*, then
sign out and back in:

```text
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

Keeping the project and the cache in short folders (e.g. `C:\FlipTuned`, `HF_HOME=C:\hf`) avoids
the problem without the setting.

**Antivirus slowness.** Real-time scanning inspects every file that's read or written, which can
make `uv sync`, the first model load and checkpoint saves much slower. If this is your own machine
and your security policy allows it, you can exclude the project's `.venv`, the Hugging Face cache
and the app home from scanning *(admin; not done by the project)*:

```text
Add-MpPreference -ExclusionPath "C:\path\to\FlipTuned\.venv"
```

On a managed (work) computer, ask your IT team instead. Excluding folders lowers protection for
them.

**"Terminate batch job (Y/N)?"** after Ctrl+C on the `.cmd` launcher: answer `Y`. The server has
already stopped; running jobs keep going.

**Files in use.** Windows can't delete or move a file another program has open. If deleting a run
fails, close any editor or Explorer preview showing its files and try again.
