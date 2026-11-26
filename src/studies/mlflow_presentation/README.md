# Lesender Zugang zum lokalen MLflow

Der Gateway reicht lesende Anfragen direkt an das bestehende MLflow auf
`127.0.0.1:5000` weiter. Keine zusätzliche Datenbank, kein Import, kein Training.
[Öffentlicher Link und Verfügbarkeit](../../../docs/mlflow-presentation.md).

Bestehenden MLflow-Prozess weiterverwenden. In getrennten Prozessen starten:

```bash
.venv/bin/python -m uvicorn src.studies.mlflow_presentation.proxy:app --host 127.0.0.1 --port 5002 --workers 1
runtime/mlflow-presentation/cloudflared tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:5002
```

Vorher bestehende Listener prüfen. Nur den Gateway-Port veröffentlichen. Der
Tunnel verwendet die lokal geprüfte offizielle cloudflared-Binärdatei. Einen
geänderten Tunnel-Link nach externer Prüfung in den Dokumentationseinstiegen
aktualisieren. Tower, Originalserver, Gateway und Tunnel müssen weiterlaufen.

Nur definierte lesende REST-Endpunkte und geprüfte GraphQL-Wurzelfelder sind
zulässig. Manche MLflow-Suchen verwenden GraphQL-Mutationssyntax; erlaubt sind
nur die namentlich freigegebenen Leseoperationen. Schreib-, Upload-, Lösch- und
Verwaltungsoperationen sind gesperrt. Der Upstream ist fest vorgegeben.
