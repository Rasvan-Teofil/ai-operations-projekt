# Modellvergleich

Gemeinsamer Testsplit der aufbereiteten Finanznachrichten-Sätze (Seed 42, stratifiziert nach `label|language`): **n_train = 5923**, **n_val = 1270**, **n_test = 1270**. Val wurde nicht zum Abstimmen benutzt. Labelraum: `negative`, `neutral`, `positive`. Metriken aus einem MLflow-Experiment `article-sentiment`.

Quellen: [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) `sentences_75agree` (Englisch, CC BY-NC-SA 3.0) und [Kenpache/multilingual-financial-sentiment](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment) nur Deutsch (akademisch / nicht-kommerziell). Details in `data/README.md`.

## Kennzahlen

Sortierung: macro-F1 absteigend, dann Inferenzzeit. Gemessen auf CPU, ohne GPU.

| model | model_version | role | accuracy | f1_macro | f1_negative | f1_neutral | f1_positive | inference_ms_per_text | size_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tfidf_linearsvc | tfidf-linearsvc-v1 | classic | 0.7575 | 0.7202 | 0.6758 | 0.8260 | 0.6587 | 0.0221 | 4.0826 |
| tfidf_logreg | tfidf-logreg-v1 | baseline | 0.7189 | 0.6562 | 0.6017 | 0.8062 | 0.5608 | 0.0264 | 4.0828 |
| finetuned | finetuned:distilbert-base-multilingual-cased | finetuned | 0.6819 | 0.6142 | 0.5946 | 0.7962 | 0.4517 | 8.3083 | 516.2315 |
| cardiffnlp | cardiffnlp/twitter-xlm-roberta-base-sentiment | pretrained_zero_shot | 0.6039 | 0.4859 | 0.4175 | 0.7165 | 0.3237 | 19.0046 | 1060.6611 |
| tabularisai | tabularisai/multilingual-sentiment-analysis | pretrained_zero_shot | 0.5409 | 0.3636 | 0.2864 | 0.6851 | 0.1193 | 8.8217 | 516.2373 |

## Was wo gelaufen ist

- **sklearn (beide):** voller Train-Split, 5 923 Sätze, CPU.
- **Zero-Shot (beide):** keine Anpassung, voller Testsplit, CPU. `tabularisai` faltet fünf Klassen auf drei (Very Negative + Negative, Neutral, Positive + Very Positive). `cardiffnlp` ist schon dreiklassig. Ein Testsatz ist länger als 126 Inhaltstokens (Maximum 137), alle anderen passen in ein Fenster. Bei 512 Tokens ist keiner zu lang.
- **Fine-Tuning, dieser Lauf:** `distilbert-base-multilingual-cased`, CPU, **1 Epoche**, Lernrate 2e-5, Batch 8, Kontext **128**, **2 000 stratifizierte Trainingssätze** von 5 923 (`--max-samples 2000`). Bewertung auf dem vollen Testsplit. Ein Testsatz (137 Tokens) wurde bei der Inferenz gechunkt, weil das Trainingsfenster 128 ist. Trainingszeit etwa 201 Sekunden.
- **Nicht gelaufen:** drei Epochen auf allen 5 923 Sätzen mit Kontext 256. Das ist der GPU-Lauf in `notebooks/02_finetune_colab.ipynb`. Dessen Kennzahl steht hier nicht, weil wir sie nicht gemessen haben.

Die Scores von LinearSVC sind ein Softmax der Entscheidungsfunktion, keine kalibrierten Wahrscheinlichkeiten. Der Argmax entspricht `LinearSVC.predict`. macro-F1 hängt nur am Label.

## Trade-off

Auf diesem Testsatz gewinnt **TF-IDF + LinearSVC** klar: macro-F1 0.72 gegen 0.66 (LogReg), 0.61 (Fine-Tuning-Subset), 0.49 (Twitter-XLM-RoBERTa) und 0.36 (tabularisai). Es ist zugleich das schnellste und das kleinste Modell, etwa 0,02 ms pro Satz und 4 MB.

Der Abstand zur Baseline ist kein Gleichstand, bei dem die Latenz entscheiden müsste. LinearSVC liegt bei jeder Klasse vorn, auch bei `positive`, wo die Transformer am schwächsten sind (tabularisai 0.12, cardiffnlp 0.32, Fine-Tuning-Subset 0.45). Neutral ist überall am leichtesten, das folgt aus der Klassenverteilung.

Die Zero-Shot-Modelle sehen die Trainingslabels nicht. CardiffNLP ist auf Tweets trainiert, nicht auf Finanzsätze aus Investorsicht. tabularisai faltet fünf allgemeine Stimmungsstufen; „positiv für den Kurs“ ist nicht dasselbe. Beides erklärt die Lücke besser als die reine Modellgröße.

Das eigene Fine-Tuning liegt über beiden Zero-Shot-Modellen und unter beiden TF-IDF-Modellen. Es hat nur ein Drittel der Trainingssätze, eine Epoche und den kürzeren Kontext gesehen. Ein voller GPU-Lauf kann das verbessern. Er kann den Abstand von 0,11 macro-F1 zu LinearSVC nicht versprechen. Solange diese Zahl nicht gemessen ist, bleibt LinearSVC die belegte Wahl.

Für die API zählt zusätzlich: LinearSVC braucht kein PyTorch, startet mit `requirements.txt` und antwortet auf einem ganzen Artikel weiterhin schnell, weil TF-IDF den vollen Text in einem Vektor sieht. Die Transformer stückeln bei 512 Tokens.

## Empfehlung

**Modell: `tfidf_linearsvc`.** Das ist der API-Standard (`SENTIMENT_MODEL` schaltet um). Die CI bleibt bei `tfidf_logreg`, damit der Workflow keine zusätzliche Modelldatei und kein Torch braucht; beide TF-IDF-Modelle entstehen mit `python -m src.train`.

LogReg ist die Alternative, wenn kalibriertere Scores wichtiger sind als das Label. Das Fine-Tuning erst dann erneut prüfen, wenn der Colab-Lauf auf demselben Testsplit gemessen wurde.
