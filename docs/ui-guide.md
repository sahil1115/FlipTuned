# Web UI guide

Start the app with `uv run fliptuned serve` (add `--home ./.dev-home` for the synthetic fixtures).
It opens `http://127.0.0.1:8765/#token=…` in your browser. Only this computer can reach it, and
the token in the link changes every launch: if the page says it needs the token, open the link the
command printed again.

The UI needs a build first: `npm --prefix web ci && npm --prefix web run build`. Without one, `serve`
shows a placeholder page.

The screens (screenshots from the end-to-end tests are in `docs/screenshots/`). Every
step of the workflow can be done here without the CLI:

## Dashboard

- **This machine:** system, memory, accelerator, and which backend Auto picks, with the reason and
  any install suggestion.
- **Active job:** the job running now, its progress and whether it holds the model. A training job
  links to its run monitor.
- **Training runs:** the latest runs with their status, best validation loss, latest verdict and
  number of exports, each linking to its run monitor, plus a link to all runs.
- **Data:** data sources and datasets.
- **Models:** how many are in the catalog, downloaded, and estimated to fit here.
- **Training configs:** the latest saved configs, each with a link to its preflight.
- **Jobs:** recent jobs with their status, and which job holds the model.
- **Disk use:** per folder of the app home.

## Data sources

1. **Allowed folders.** Add the folder that holds your Q&A files (its full path). FlipTuned reads
   data only from allowed folders and from files you upload. The app home itself can't be added.
2. **Browse** to a folder and press **Scan this folder**. Or **upload** CSV, XLSX, Markdown or TXT
   files: they're copied into the app home and scanned.
3. The **scan** shows:
   - file counts by type;
   - files FlipTuned won't read;
   - encoding problems;
   - for every CSV file or XLSX sheet, a column mapping filled in with FlipTuned's suggestion.

   Question and answer are required. Every other field is optional, with an explanation under it.
4. **Ingest.** The progress bar follows the job. You can leave the page; the latest ingest is shown
   when you come back. When it finishes you see records by status, quality flags, privacy findings
   and exclusions.

## Review

- **Filter** by status, data source, flag (for example `exact_duplicate`), category, or duplicate
  clusters.
- **Read a record:** select it to see its question and answer. Code blocks are shown exactly as
  stored, with their language.
- **Privacy findings** show only the type and length of what was found, never the value.
- **Single decisions:** Approve, Reject or Back to pending.
- **Bulk decisions:** tick records, or "Select all on this page", then Approve or Reject.
- Quarantined records and text blocks can't be approved.

## Dataset

**Build a dataset:**
- **Name.**
- **Model:** the model you'll train. Its tokenizer measures sequence lengths, and only models
  whose template test passed are listed.
- **Sources:** none selected means all of them.
- **Split shares, seed and maximum sequence length:** each field has a hint and is checked before
  anything is sent.
- **Download the tokenizer:** tick this if the model's tokenizer isn't on this computer yet.

After the build:
- **Warnings:** for example a small test set, one group dominating a split, the same question
  with different approved answers, or a category with no test examples.
- **Statistics:** records considered and included, duplicates removed, the largest group, and per
  split the target vs actual share, examples, groups and token-length percentiles.
- **Exclusions,** with their reasons.
- **Check files** verifies the dataset on disk.

## Models

One card per catalog model:
- parameters, license, model card address;
- **backends here:** the backends this machine has that the model has a version for;
- **memory on this machine:** `fits`, `tight`, `won't fit` or `unknown`, from the catalog's
  estimate of peak training memory against this machine's memory (free VRAM, unified memory or RAM,
  minus room for the OS). It's always an estimate, never a measurement;
- download size and whether it's downloaded. **Download** runs as a job with progress; you can
  leave the page, and the latest download is shown when you come back;
- whether its chat template passed the prefix test (only those can be trained).

Tiny test-only models appear only with developer mode on (`dev_mode` in the settings).

## Training config

- **Saved configs:** every config with its latest version. **Edit** loads it into the form (saving
  a change makes the next version; old versions stay, and every run records the one it used).
  **Duplicate** copies it under a new name. **Preflight** checks it.
- **Beginner view** (what you see first): name, model, dataset (the most recent one if you don't
  pick), preset (quick / standard / thorough) and max epochs. Under them, the backend Auto picks on
  this machine, and why.
- **Advanced settings** (collapsed): backend and device, and precision, listing only what this
  machine has (and why the others aren't available), then every training, LoRA and prompt setting.
  Each field shows its explanation and the value it gets from the preset or default. Leave a field
  empty to keep that value, or type your own: it's checked as you type, marked "changed", and
  **Reset** puts it back.
- Only the fields you change are saved, so the preset keeps deciding everything else.
- **Save and preflight** saves, then opens Preflight for that version.

## Preflight

- Pick a saved config and a version (latest by default), and whether model files may be
  downloaded.
- Every check shows **pass**, **warn** or **fail**, its message and what to do: dataset, system
  prompt, model, backend, memory (estimate), CPU size limit, model files, disk, the chat-template
  test on real training examples, whether every example fits `max_seq_length`, the update count
  and the CPU time estimate.
- **Start training** stays disabled while any check fails; the server refuses too. Training runs as
  a background job, and the run monitor opens for it.
- If another job is using the model (training, evaluation, chat or merge), a notice at the top
  names it, links to it when it's a training run, and **Start training** stays disabled until it
  finishes. One model job runs at a time.

## Run monitor

- Pick a run (newest first; the running one is chosen when you open the screen). The run is in the
  address, so reloading the page shows the same run.
- **Status** in plain words for every state: queued, running, succeeded, failed (with the error and
  what to do), cancelled and interrupted (the process ended unexpectedly, e.g. the computer slept).
- **Step**, epoch, **time left** (counting down from the last report), the **best checkpoint**, and
  **peak memory** for the run's device: VRAM on NVIDIA, unified memory on Apple Silicon, RAM on the
  CPU, out of the machine's total.
- **Loss chart:** train and validation loss by optimizer update; a dashed line marks the best
  checkpoint (the lowest validation loss, "best so far" while training). The numbers are under
  "Loss values". Falling loss alone doesn't show the model got better; the evaluation does.
- **Warnings** from preflight and from the job (for example no keep-awake on this system).
- **Log:** status changes, losses and warnings as they happen ("● Live"). If the connection drops
  it says "Reconnecting…" and continues where it stopped. The log holds metrics and messages only,
  never record text.
- **Cancel** stops at the next step and saves a checkpoint (it is force-stopped if it doesn't stop
  within a minute). **Resume** continues an interrupted, failed or cancelled run from its last
  checkpoint; it is disabled, with the reason, when there is no checkpoint or another job is using
  the model. **Attempts** lists the first start and every resume.
- Training keeps going if you close the page or the browser, or restart `fliptuned serve`; open
  the new link and the run monitor shows it live again.

## Evaluation

- Pick a finished run (or follow "evaluate the run" from the run monitor). Its evaluations are
  listed with their status, verdict and how many pairs you have reviewed.
- **Start an evaluation:** both the base model and the fine-tuned model answer every held-out test
  question and about 20 general coding prompts, through the run's backend, with the same system
  prompt and greedy decoding. It runs as a background job under the model lock. "Only the first N
  test items" makes a quick check (below 50 items the verdict is "inconclusive").
- **Blind review:** one pair at a time: the question, the approved answer (folded away), Answer A
  and Answer B, in random order. Press **1** (A is better), **2** (B is better) or **3** (tie), or use
  the buttons; "Previous pair" and "Next pair" let you revisit one. Judgements are saved as you go,
  so you can stop and come back. The page is never told which model wrote A or B.
- **Or review in a spreadsheet:** fill in the evaluation's `review_sheet.csv` and upload it; it's
  scored the same way.
- **Results:** the verdict (accepted, rejected or inconclusive) and why, every criterion with its
  value, the threshold and whether it was met, wins/losses/ties, the sign test, the automatic-check
  and general-coding pass rates for both models, regressions, results by category, the test-set
  size, any note that test results influenced earlier data changes, and the limitations.

## Chat

- A permanent banner: **model weights only — no document retrieval**. Answers come from the model
  alone, and nothing you type is logged or saved.
- Pick a finished run and the fine-tuned model (base + adapter) or the base model alone, then
  **Load the model and chat**. Replies stream in; Enter sends, Shift+Enter adds a line. **Stop**
  ends a reply early and keeps what arrived. If a reply fails, your message goes back into the
  box to send again; if it hit the length limit, the page says it was cut off.
- The model is released after the idle timeout (Settings) without a message; the page says so, and
  you load it again to continue (the conversation starts over). **Close and release the model**
  frees it at once; closing the tab does too. If another job holds the model, the screen names it,
  and offers to release it when it's an earlier chat.

## Runs

- **History** of training runs with filters (status, model, run ID or config name) and an option to
  show archived runs, with the best validation loss and latest verdict.
- **Compare:** tick two runs to see their metrics and verdicts side by side and every setting that
  differs in their effective configs (with a note if they used different datasets).
- Open a run for its **artifacts** (the run folder's files with sizes; text files such as
  `metadata.json`, `report.md` and `review_sheet.csv` can be downloaded), links to its monitor,
  evaluation and chat, **export** (the adapter, or a merged standalone model as a job with a 5-prompt
  check), **archive** (hides the run and keeps everything; **Restore** brings it back) and
  **Delete permanently**, which asks you to type the run ID and removes the run's adapter,
  checkpoints, evaluations with their judgements, exports in the run folder and job logs.

## Settings

- **Allowed folders** (add or remove) and the **app home** (read-only; set with `--home` or
  `FLIPTUNED_HOME`).
- **Options:** chat idle timeout, largest upload, the **port** `fliptuned serve` tries first (from the
  next launch; `--port` overrides it) and developer mode (shows tiny test-only models).
- **This machine:** the doctor report (system, Python, CPU, memory, accelerator, the backend in use
  and why, every backend with whether it's available here, warnings, packages, disks and caches)
  and **install suggestions**: the exact `uv sync` command for this machine and what installing
  more would unlock.

## Everywhere

- **Errors** say what went wrong, then what to do; technical details are behind "Details".
- **Keyboard:** everything works with the keyboard. "Skip to content" is the first Tab stop.
- **Themes:** light and dark (the button in the header; the first visit follows your OS setting).
- **Paths** are shown the way your OS writes them.
