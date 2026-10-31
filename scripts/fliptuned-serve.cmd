@echo off
rem Start the FlipTuned web app from anywhere (double-click, a shortcut, or a terminal).
rem Extra arguments go to `fliptuned serve`, e.g. --port 8780 or --home <folder>.
uv run --project "%~dp0.." fliptuned serve %*
