<!-- English draft of docs/presentation-guide.md on main (state d87f370). Drop-in replacement: same links, same structure. -->
# Presentation Walkthrough

The complete current state is on `main`; no branch switch is needed for the demo.
Japanese is explicitly documented in [the language overview](languages/japanese.md).

1. Open the README: media-monitoring problem, full text or one URL,
   three classes, native DE/EN and separately reported JA.
2. Show the [structure](repository-structure.md): lecture template, separation of
   `src/`, HTTP contract in `app/schemas.py`, local data/models and evidence.
3. Open the local application and enter a short news text of your own.
   Show the model version and scores; scores are not calibrated probabilities.
4. Compare: the same 691-article test for four candidates, DE273/EN118/JA300,
   240 train. Qwen LoRA 0.8218, Zero 0.7796, MiniLM 0.5928, Frozen 0.5150
   in equally weighted DE/EN macro-F1. Selection was made before the test.
5. Name the trade-off: MiniLM about 3.66 ms/article and 241 MiB;
   Qwen LoRA about 184.26 ms and 7699 MiB including the base model.
6. Development: 27 evaluations inside train, no saturation of the small
   learning curve. 240 is not a data optimum; no artificial enlargement of the test.
7. Show [original sources and labels](data/README.md) and the
   [project-side review](data/project-review.md). The original
   AI-assisted creation is documented as provenance. Restricted
   archive and possible pre-training overlap remain limitations.
   95% intervals do not mean 95% accuracy.
8. Reproduce and Data & licences: fixed hashes, new MLflow runs, exact
   source attribution; raw texts and weights are deliberately not in the Git repository.
9. [Gradient clipping ablation](../reports/clipping_control/interpretation-v1.md):
   same seed, same split, same recipe, only clipping 1.0 or off.
   30 distinct pairs; within the CV three folds and three seeds per recipe.
   With minus without clipping in the DE/EN main score: existing recipe +0.32 points
   [−2.13; +2.61], new recipe −0.61 [−3.43; +1.75]. Both conditional intervals
   include zero. No robust quality gain shown, no model change.
   The values are exploratory train-internal CV, not a repeat of the 691-article test.

The earlier studies stay in the collapsible archive. Do not compare their different
test sets with the full-article test as a progress curve.
The service runs locally on `127.0.0.1:8000`; there is no public deployment.

## Next weekly block: Qwen code in English

To explain the best model and the Qwen variants, follow the
[English code walkthrough](qwen-code-walkthrough.md). It separates the
training path, candidate selection, inference and FastAPI integration and names
the functions that should actually be shown.
