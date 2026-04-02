# Tool Button Program

- Goal: maximize detector composite `score`.
- Default data: `configs/config.tool_button_detection.yaml`.
- Primary metric: `score`.
- Secondary metrics: `mAP50`, `click_success`, `latency_ms_mean`.
- One experiment run executes the repo's detector optimization flow and evaluates the promoted checkpoint.
- Allowed mutation surface for experiments: repo training code, config, objective tuning, and this workspace.
- Do not edit `third_party/autoresearch/` during normal runs.
