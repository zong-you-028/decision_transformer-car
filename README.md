# decision_transformer-car

This repository implements a Decision Transformer for the highway-env environment.

The original single `CAR.py` script has been split into smaller modules located in the `decision_transformer` package:

- `model.py` – the `DecisionTransformer` network definition
- `data.py` – dataset utilities and trajectory collection
- `trainer.py` – model training logic
- `evaluator.py` – evaluation helpers
- `main.py` – entry point that ties everything together

Run the training pipeline with:

```bash
python main.py
```
