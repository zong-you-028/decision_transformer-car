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

After training, an animation of the training environment is saved as
`training_animation.mp4`.

Collected trajectories are stored in `highway_trajectories.pkl`. The
`load_or_collect_trajectories` helper in `data.py` automatically loads
this file if present or collects new data otherwise, so running the
training script twice will reuse the saved dataset.
python main.py
```
