# Civ6 Harvest Taxonomy Templates

These split files are reference templates for page-level harvest labeling.

Use them to keep routing labels, grounding semantics, and review decisions
consistent before large-scale collection.

Runtime source of truth:

- `configs/harvest_taxonomy/civ6.yaml`

The current code path loads the single-file taxonomy above.
This folder is kept as a human-editable breakdown/reference copy.

Files:

- `primitives.yaml`: stable primitive registry used for routing labels
- `primitive_backlog.yaml`: candidate or future primitives not ready for stable export
- `situations.yaml`: page/situation labels used for routing context
- `elements.yaml`: actionable UI element ontology used for grounding review
- `rois.yaml`: named ROI priors for routing and element lookup
- `capture_policy.yaml`: collection constraints to keep the dataset consistent

Important:

- The primitive list does not need to be complete on day 1.
- Keep only stable router classes in `primitives.yaml`.
- Put new or uncertain primitives in `primitive_backlog.yaml` first.
- Promote a backlog primitive to `primitives.yaml` only after its trigger criteria,
  completion rule, and review examples are stable.
- Keep `primitive_id` coarse when needed, but let `situation_id` be finer-grained.
- In practice, router classes stay stable while `situations.yaml` captures prompt-
  level screen states such as confirmation popup, policy screen, governor map, or
  research chooser open.

Minimum fields you should decide before collecting:

1. `primitive_id`
2. `situation_id`
3. `semantic_id`
4. ROI names and normalized boxes

Minimum fields you should review per sample:

1. `page.situation_id`
2. `route_label.primitive_id`
3. `elements[].semantic_id`
4. `elements[].bbox_xyxy`

Recommended workflow:

1. Fill `primitives.yaml` with only stable router-included primitives.
2. Put uncertain or future primitives in `primitive_backlog.yaml`.
3. Fill `situations.yaml`, `elements.yaml`, and `rois.yaml`.
4. Collect only route-critical moments.
5. Run `data-harvest label-auto`.
6. Review with the taxonomy as the source of truth.
7. Export `router`, `grounding`, and `unified`.

Situation design rule:

- `primitive_id`: what expert module should handle this page
- `situation_id`: what exact page state is visible right now

Example:

- `popup_primitive` can cover:
  - `confirm_accept_popup`
  - `informational_popup`
  - `next_turn_ready`
  - `research_prompt_visible`
  - `production_prompt_visible`
  - `civic_prompt_visible`
- `governor_primitive` can cover:
  - `governor_list_open`
  - `governor_promotion_open`
  - `governor_city_assignment_map`
  - `governor_assignment_confirm`
