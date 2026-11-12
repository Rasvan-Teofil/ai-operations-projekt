# Öffentliches Hosting ohne Tower

**Stand: 12. November 2026. Budget: 0 €. Noch keine öffentliche Deployment-URL.**

Ziel ist eine öffentliche Webseite für Lehrende und andere Besucher. Ein Browser
sendet Eingaben an einen extern betriebenen Dienst; der private Tower soll dafür
nicht eingeschaltet sein müssen. GitHub ist der Einstieg und Quellcodebestand,
führt das GPU-Modell jedoch nicht selbst als dauerhaft öffentliche Anwendung aus.

Die vorhandene lokale FastAPI-Anwendung funktioniert. Ihr aktuell gewähltes
Qwen-LoRA-Modell erfordert im eingefrorenen Runtime-Code eine CUDA-GPU mit BF16;
CPU-Fallback ist ausdrücklich deaktiviert. Nur ein Modell zu betreiben reduziert
den Umfang, beseitigt diesen Hardwarebedarf aber nicht. Es gibt noch keinen
gemessenen CPU- oder quantisierten Deployment-Ersatz dieses Artefakts.

## Geprüfte kostenfreie Möglichkeit und aktuelles Hindernis

[Hugging Face ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu) kann
gemeinsam genutzte GPU-Zeit für eine Demo bereitstellen. Laut aktueller Dokumentation
können persönliche Konten mit bestätigter E-Mail und einem Alter von mehr als
30 Tagen bis zu zwei solcher Spaces kostenlos betreiben. Die SDK-Unterstützung
beschränkt sich auf Gradio; die bestehende FastAPI-Anwendung lässt sich deshalb
nicht unverändert als Docker-Space in dieses kostenlose Angebot schieben.
Warteschlangen und tägliche Kontingente gelten, kein zugesagter Dauerbetrieb.

Es existiert nach Rückmeldung noch kein geeignetes Konto, und hier ist kein
Hugging-Face-Konto angemeldet. Dieser Weg ist daher gegenwärtig nicht deploybar.
Die normale [Docker-Space-Variante](https://huggingface.co/docs/hub/en/spaces-overview)
erfordert nach aktueller Dokumentation einen bezahlten Tarif und erfüllt das
0-€-Budget nicht. Keine kostenpflichtigen Ressourcen wurden angelegt.

## Weitere Optionen, noch nicht als verfügbar bestätigt

- Ein kostenlos bereitgestellter Hochschul-/Projektserver mit geeigneter GPU
  könnte dieselbe FastAPI-Anwendung und das unveränderte Defaultmodell betreiben.
  Es ist noch kein solcher Server zugesagt oder eingerichtet.
- Ein kleineres CPU-Modell könnte auf knapperen Ressourcen betreibbar sein.
  Das wäre eine ausdrücklich gekennzeichnete andere Modellwahl. MiniLM erzielt
  im bestehenden Test einen niedrigeren DE/EN-Hauptwert; keine stille Ersetzung
  des ausgewählten Qwen-LoRA-Modells.
- Browserinferenz oder Quantisierung wären neue Deployment-Varianten. Sie sind
  weder implementiert noch mit der bisherigen Modellqualität gleichzusetzen.

Die bestehenden Messungen und der einmalige Abschlusstest werden wegen einer
Hostingentscheidung nicht geändert oder erneut zur Auswahl verwendet. Für einen
öffentlichen Dienst genügen das Defaultmodell, seine notwendigen Basisgewichte
und der Adapter. Historische Gewichte, Rohdatensätze und MLflow müssen dafür
nicht öffentlich verteilt werden. Herkunft und geltende Hinweise bleiben erhalten.
