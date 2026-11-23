# Bridge slide draft: From pitch to full-article study

Draft text for one slide in the final presentation. Not in the Canva deck yet; the deck stays unchanged until Teo decides how the final deck is built.

## Slide title

From pitch prototype to full-article study

## Slide text (left: pitch, right: now)

**Executive Pitch (Nov 2026)**
- Input: single financial sentences (PhraseBank + Kenpache)
- Run 1 LinearSVC macro-F1 0.7202, run 2 MiniLM 0.7187
- Lesson: sentences are not what newsrooms, PR & communications teams and analysts read

**Full-article study (main, 14 Nov 2026)**
- Input: complete news articles or one URL, native DE, EN and JA
- 240 train articles, 691 fresh test articles (DE 273, EN 118, JA 300)
- Qwen3-4B LoRA chosen before the test: DE/EN macro-F1 0.8218
- Cost: about 184 ms and 7.7 GB per article vs. 3.7 ms and 241 MB for MiniLM

## Speaker note

- The pitch numbers and the full-article numbers come from different test sets. Show them side by side as two stages, not as one progress curve.
- What carried over: the three classes, MLflow tracking, the FastAPI service and the test suite.
- What changed: the unit of analysis (article instead of sentence), the data (own reviewed articles) and the model family (LoRA on a small LLM).
- Open limitation: no public deployment yet, the service runs locally.

## Sources

- Pitch values: `reports/` on the pitch state and the Canva deck (design DAHZaEaLHcQ)
- Full-article values: `README.md` and `reports/next_stage/comparison-v1.md` on `main` at `2c612d8`
