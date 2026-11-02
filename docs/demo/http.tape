# Records docs/demo/http.gif from a real run: `vhs docs/demo/http.tape` from the repo root.
Output docs/demo/http.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `python -m preference_optimization serve --port 18320 >/dev/null 2>&1 &` Enter
Sleep 2s
Sleep 40s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18320/score -d '{"prompt": 3}' | jq -c '{gold_best_response, policy_greedy_response}, (.responses[] | {response, true_reward, policy_prob})'` Enter
Sleep 4s
Hide
Type `kill %1` Enter
Sleep 1s
