# Router Civ6 Program

- Goal: maximize `primitive_id` macro F1 on the held-out router validation split.
- Default data: `configs/config.router_classification.yaml`.
- Primary metric: `primitive_macro_f1`.
- Secondary metrics: `primitive_accuracy`, `screen_type_accuracy`, `situation_id_accuracy`.
- Allowed mutation surface for experiments: repo training code, config, dataset adapters, and this workspace.
- Do not edit `third_party/autoresearch/` during normal runs.
