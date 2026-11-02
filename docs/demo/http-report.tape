# Records docs/demo/http-report.gif from a real run: `vhs docs/demo/http-report.tape` from the repo root.
Output docs/demo/http-report.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 520
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `python -m preference_optimization serve --port 18321 >/dev/null 2>&1 &` Enter
Sleep 2s
Sleep 40s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18321/report | jq` Enter
Sleep 4s
Type `curl -s localhost:18321/metrics | head -n 6` Enter
Sleep 3s
Hide
Type `kill %1` Enter
Sleep 1s
