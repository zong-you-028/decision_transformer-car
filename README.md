# decision_transformer-car

This repository implements a Decision Transformer for the highway-env environment.

The original single `CAR.py` script has been split into smaller modules located in the `decision_transformer` package:

- `model.py` – the `DecisionTransformer` network definition
- `data.py` – dataset utilities and trajectory collection
- `trainer.py` – model training logic
- `evaluator.py` – evaluation helpers
- `main.py` – entry point that ties everything together

The data collector now supports three driving styles:

- **aggressive** – accelerates and changes lanes frequently
- **cautious** – keeps lower speeds and avoids risky moves
- **normal** – a balanced heuristic

After training, a plot comparing the three driving styles is saved as
`style_analysis.png`. An animation of the evaluation episodes is also
generated as `decision_transformer_highway.mp4`.

Collected trajectories are stored in `highway_trajectories.pkl`. If this
file already exists, the main script will load it instead of collecting
new data, allowing you to reuse previously gathered datasets.

Run the training pipeline with:

```bash
python main.py
```
