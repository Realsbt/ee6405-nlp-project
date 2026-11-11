# Yingqi presentation deliverables

`yingqi_same_window_ablation.pptx` is a ten-slide, editable PowerPoint section
covering the fair-window RACE mechanism ablation. It contains six native data
charts, two native tables and speaker notes that point to the repository
evidence. The deck clearly separates completed CPU/data-integrity evidence from
the pending 12-run GPU matrix.

The six presentation-ready PNG figures are stored in
`experiments/reports/same_window/figures/`. Their source values are in
`experiments/reports/same_window/figure_data.csv`.

## Rebuild

The deck uses the bundled Codex presentation runtime. From the repository root,
set these paths to the values returned by the workspace dependency loader:

```bash
export RUNTIME_NODE=/path/to/bundled/node
export RUNTIME_NODE_MODULES=/path/to/bundled/node_modules
export RUNTIME_PYTHON=/path/to/bundled/python
export CODEX_PRESENTATION_SKILL_DIR=/path/to/presentations/skills/presentations

mkdir -p .build/yingqi-presentation
ln -sfn "$RUNTIME_NODE_MODULES" .build/yingqi-presentation/node_modules
cp scripts/build_yingqi_deck.mjs .build/yingqi-presentation/build.mjs
"$RUNTIME_NODE" .build/yingqi-presentation/build.mjs
```

The finalizer checks package integrity, slide geometry, approved fonts, native
chart ownership and import through the first-party Artifact Tool. It does not
replace a manual review in PowerPoint.
