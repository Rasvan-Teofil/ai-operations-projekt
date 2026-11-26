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
| Neuer Volltrain-Kandidat | [Originalmodelllauf](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/3/runs/5230309ca351479980869ef61fc239f0) |

Beim Abgleich am 26.11.2026 waren alle vier Experimente einschließlich des
leeren Default-Experiments und alle 165 gespeicherten Läufe öffentlich und lokal
identisch: 131 in Experiment 1, 33 in Experiment 2, einer in Experiment 3.
Die 33 Ablationseinträge sind die gespeicherte Historie; die abgeschlossene
Studie umfasst weiterhin 32 Fits. Es wurden keine zusätzlichen Versuche erzeugt.
Auch der Ergebnisdownload des neuen Modelllaufs wurde bytegenau verglichen.

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
