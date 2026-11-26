# Öffentlicher MLflow-Zugang

**[MLflow öffnen](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments)**

Dieser Link öffnet die bestehende MLflow-Instanz auf dem Tower über einen
öffentlichen, lesenden Zugang. Es gibt keine zusätzliche Ergebnisdatenbank,
keine umsortierten Experimente und keine neu importierten Läufe. Originale
Laufkennungen, Startzeiten, Parameter, Metriken, Verläufe und Artefakte bleiben
identisch. Neue lokale Ergebnisse werden über denselben Zugang sichtbar.

| Originalexperiment | Öffentlicher Link |
|---|---|
| article-sentiment-second-path | [Experiment 1](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/1) |
| article-optimizer-ablation-v1 | [Experiment 2](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/2) |
| article-optimizer-candidate-v1 | [Experiment 3](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/3) |
| article-optimizer-validation-v1 | [Experiment 4](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/4) |
| Neuer Volltrain-Kandidat | [Originalmodelllauf](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/3/runs/5230309ca351479980869ef61fc239f0) |

Beim abschließenden Abgleich am 26.11.2026 enthält die Originalinstanz fünf
Experimente einschließlich des leeren Default-Experiments und 187 gespeicherte
Läufe: 131 in Experiment 1, 33 in Experiment 2, einer in Experiment 3 und 22 in
Experiment 4. Die neue Validierung umfasst 21 neue Fits und sechs exakt
übernommene ältere Fits; die sechs behalten ihre alten Kennungen aus Experiment 2.
Ein zusätzlicher Setup-Fehlschlag in Experiment 4 hat null Trainingsschritte
und keine Vorhersagen. Er bleibt als Ausführungshistorie erhalten.
Die abgeschlossene Faktorstudie umfasst weiterhin 32 Fits.
[Abschluss und Modellentscheidung](../reports/optimizer_validation/README.md).

Die Oberfläche ist dieselbe MLflow-Version. Browserabhängige Einstellungen wie
Tabellenspalten, Filter und Sprache können auf anderen Geräten abweichen.
Öffentliches Anlegen, Ändern, Hochladen und Löschen sind gesperrt.

## Verfügbarkeit

Tower, lokales MLflow auf Port 5000, lesender Gateway auf Port 5002 und Tunnel
müssen laufen. Der kostenlose [Cloudflare Quick Tunnel](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
ist vorübergehend; sein Link kann sich nach einem Tunnelneustart ändern.
GitHub hostet die Datenbank nicht. Beim Betrachter ist keine Installation nötig.

[Startanleitung für den Zugang](../src/studies/mlflow_presentation/README.md).
Ohne laufenden Server bleiben die [Optimizer-Berichte](../reports/optimizer_ablation/README.md)
und die [Modellkarte](../reports/optimizer_candidate/README.md) auf GitHub lesbar.
