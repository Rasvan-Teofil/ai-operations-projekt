> Abschlussstand 11. November 2026: 27 Entwicklungsauswertungen und vier vollständige Train-Fits abgeschlossen. Frischer Test: 691 Artikel (DE273/EN118/JA300). Vor Test gewählte Qwen-LoRA-Variante jetzt lokaler Standard. [Ergebnisbericht](../reports/next_stage/comparison-v1.md), [Modellkarten](../reports/next_stage/model-cards-v1.md), [Reproduktion](reproduce-full-article.md). Die folgenden Planungs-/Checkpointabschnitte bleiben als Phasenhistorie erhalten; offene Formulierungen dort beschreiben ihren damaligen Stand.

# Nächste Phase: Vollartikel und belastbare Vergleichsgruppen

Stand: 11. November 2026. Gemeinsames Gruppenprojekt; `second-path` und
`article-sentiment-second-path` bleiben technische Kennungen. Beide Quellenpiloten
sind zweimal geprüft. 240 geeignete Volltexte bilden den eingefrorenen Train-Bestand.
Die 24 festgelegten Entwicklungsauswertungen laufen; der frische Abschlusstest
wird erst nach Festlegung der Kandidaten erhoben.

## Bewertungsregel · outcome-rubric-v1

Ein Text erhält ein Label für die **hauptsächlich berichtete Entwicklung beim
zentralen Bezugsgegenstand**. Das ist keine allgemeine Bewertung aller Akteure
und keine kundenspezifische PR-Bewertung eines beliebig gewählten Unternehmens.
Der Bezugsgegenstand wird vor dem Label schriftlich festgehalten.

| Label | Inhaltliche Regel |
|---|---|
| negative | Belegter Schaden, Verlust, Verschlechterung, Scheitern oder eine konkret belegte erhebliche Gefährdung beziehungsweise verschlechterte Aussicht. Eine bloß erwähnte abstrakte Möglichkeit reicht nicht. |
| neutral | Keine belegte gerichtete Entwicklung: Termine, Beschreibungen und Routineankündigungen; auch tatsächlich ausgewogene gegenläufige Entwicklungen ohne dominanten Bezugsgegenstand. |
| positive | Belegter Gewinn, Erfolg, Wiederherstellung, Rettung oder Verbesserung beim zentralen Bezugsgegenstand. Eine erhoffte, noch nicht erreichte Verbesserung allein reicht nicht. |

Negationen, Zuschreibungen und Erwartungen gehören zum Inhalt: „kein Verlust“ ist
nicht automatisch ein Gewinn; eine zitierte Erfolgserwartung ist kein belegter
Erfolg. Ein erreichtes Wiederöffnen kann die Hauptentwicklung sein, obwohl der
Artikel einen früheren Schaden erläutert. Ein Wahlsieg ist für den zentralen
Gewinner positiv; ein Text über die Folgen für den Verlierer kann negativ sein.
Nüchterner Stil und Dringlichkeit bestimmen das Sentiment nicht.

Gemischte oder schwer verständliche Fälle bleiben sichtbar. Neutral ist kein
Unsicherheitslabel. Bei verbleibender Mehrdeutigkeit: dominanten Bezug begründen,
Mehrdeutigkeit markieren und Sensitivität getrennt berichten. Wo kein belastbares
Drei-Klassen-Referenzlabel möglich ist, keinen scheinbar sicheren Wert erfinden;
Fall, Grund und ausgeschlossene Abdeckung dokumentieren. Keine nachträgliche
Entfernung wegen schlechter Modellvorhersagen.

Der Assistent liest den exakten Volltext und annotiert selbst. Die zweite
Eigenprüfung erfolgt ohne Kandidatenvorhersagen; Änderungen an Label oder Bezug
brauchen einen Auflösungsvermerk. Herkunft bleibt **AI-labelled · Codex-reviewed**.
Beide Durchgänge stammen von einem AI-Annotator und sind keine unabhängige
Verifikation. Eine unbekannte genaue Modellrevision wird als `unavailable`
angegeben. Satzlabels sind eine eigene Aufgabe; Satzfarben erklären keine Wörter
kausal und sind kein Nachweis der Artikelqualität.

## Pilot und Rechte

Zunächst 30, danach insgesamt etwa 100 native Volltexte je Sprache DE/EN/JA.
Diese Fälle sind Entwicklungsbestand. Unauflösbare Annotationen werden als `unresolved` aufbewahrt und blockieren die reguläre Aufnahme; sie werden weder entfernt noch zu neutral umgedeutet. Japanisch bleibt die bevorzugte Erweiterung;
fehlende Rechte, Menge oder Klassenabdeckung blockieren DE/EN nicht. Pilotzeit,
Rechteverluste, Themen, Klassen, Publikationszeiträume, Zeichen- und Tokenlängen
werden gemessen, bevor eine größere Menge festgelegt wird.

Die JSONL-Schemas liefert `python -m src.next_stage schema`. `ArticleRecord`
enthält Text und exakten SHA-256, stabile Dokumentkennung, Revision, Quellen- und
History-URL, Attribution, Zeitangaben, Textlizenz und Prüfbeleg, Drittpassagenprüfung,
Transformation sowie überprüfte Ereignis- und verwandte Dokumentkennungen.
`AnnotationRecord` enthält keine Rohtexte, aber denselben Texthash, Rubrik- und
Annotationsversion, Bezugsgegenstand, Begründung und Mehrdeutigkeit je Durchgang.
Neue Annotationsmetadaten bleiben CC0, soweit Projektrechte bestehen; das ändert
die Lizenz des Ursprungstextes nicht.

Lokale Arbeitsdateien unter `data/next_stage/pilot/` ablegen (ignoriert). Kein
echter Pilotdatensatz wird als Beispiel mit erfundenen Ergebnissen eingecheckt.
Die synthetischen Strukturen in `tests/test_article_pilot.py` zeigen das Format.

```bash
source .venv/bin/activate
python -m src.next_stage schema
# Erst nach tatsächlicher Sammlung und Annotation:
python -m src.next_stage audit \
  --articles data/next_stage/pilot/articles.jsonl \
  --annotations data/next_stage/pilot/annotations.jsonl
# Nur vorhandene lokale Tokenizer; CPU, keine Downloads und keine Modellgewichte:
python -m src.next_stage profile \
  --articles data/next_stage/pilot/articles.jsonl \
  --tokenizer models/expanded_news_v4/final/minilm_bf16_news_v4/tokenizer
python -m src.next_stage profile \
  --articles data/next_stage/pilot/articles.jsonl \
  --tokenizer models/local_llm/weights
```

`audit` gibt JSON auf stdout aus und liefert bei offenen Punkten Exitcode 2.
Ohne Annotationen werden die Quellen geprüft und `annotations_pending` gemeldet.
Keine Dateien werden dabei verändert. Fehlende bekannte historische Val-/Test-
Dateien blockieren die Aufnahme, statt eine vollständige Überlappungsprüfung
vorzutäuschen. Die geprüften Dateien und ihre Hashes stehen im Ergebnis.

Die Prüfung verbindet stabile Seitenkennungen, exakte Texte/Titel, gemeinsame
Ursprungsquellen und explizite Ereignisverbindungen transitiv, auch über historische
Testzeilen. Ihre Grenzen sind wesentlich: semantische Near-Duplicates und nicht
erkannte Übersetzungen/Ereignisse werden dadurch **nicht vollständig erkannt**.
Diese Prüfung ist vor einem echten Split zu ergänzen; `ready_for_development_intake`
ist keine Freigabe für einen Abschlusstest. Das Werkzeug schreibt keine Splits. Geänderte Schutzdateien werden gegen ihre eingefrorenen Manifest-Hashes erkannt.

Rechteprüfung vom 10. November 2026:

- DE-Wikinews nennt CC BY 2.5; EN unterscheidet Publikationszeiträume.
  [DE-Regeln](https://de.wikinews.org/wiki/Wikinews:Lizenzbestimmungen),
  [EN-Regeln](https://en.wikinews.org/wiki/Wikinews:Copyright).
- Bei JA stehen ältere Angaben zu CC BY 2.5 oder neuer neben aktuellen
  CC-BY-4.0-Hinweisen. Vor Verwendung die genaue Revision und deren Anteile prüfen;
  die EN-Datumsregel nicht übertragen.
  [JA-Projektbeschreibung](https://ja.wikinews.org/wiki/ウィキニュース:ウィキニュースについて),
  [JA-Lizenzhinweis](https://ja.wikinews.org/wiki/MediaWiki:Copyrightwarning).
- Wikinews ist ein historisches Archiv, kein zugesicherter neuer Nachrichtenfeed.
  Die frühere lokale API-Beobachtung `readonly=false` bleibt erhalten und wird
  nicht zu einer aktuellen Verfügbarkeitsgarantie umgedeutet.
  [Wikimedia-Schließungsticket](https://phabricator.wikimedia.org/T421796).
- Kenpache/PhraseBank bleiben außerhalb des veröffentlichungsorientierten Pools.
  SENTiVENTs Annotationserlaubnis deckt Verlagsartikel ausdrücklich nicht pauschal
  ab. GerNTSA ist weiterhin nur Kandidat mit ungeklärtem Ursprungstextumfang.
  Die historischen Belege in `licenses/` bleiben unverändert erhalten.

Die Prüfung ist eine dokumentierte Projektauswahl, keine pauschale Rechtegarantie.
Volltexte erfordern auch bei bereits bekannten Seiten neue Inhalts-/Rechteprüfung
und Annotation; alte Lead-Labels werden nicht automatisch übernommen.

## Vollartikel und Anwendung

`full-article-v5` extrahiert geordnet die sichtbaren Absätze eines Artikelkörpers
einschließlich Zwischenüberschriften. Unterstützte Strukturen sind `articleBody`,
`article`, `.mw-parser-output`, `main` und einfache Body-Layouts. Versteckte Inhalte,
Navigation und Formulare werden entfernt. Bekannte Zugriffs-/Loginseiten werden
zurückgewiesen; dies ist keine Garantie gegen jede denkbare Fehlerseite.
Absätze werden nicht auf einen Lead verkürzt. Inline-Zitate bleiben Inhalt und
brauchen bei Datensammlung eine gesonderte Rechteprüfung.

Alle Modelle erhalten bei URL-Eingaben dieselbe Extraktionsversion. Historische
Lead-/Extraktionsfunktionen bleiben für Reproduktionen erhalten. Die API liefert
zusätzlich `input_policy`; Modellversionen bleiben unverändert. Neue Live-URL-
Ausgaben sind wegen der geänderten Aufbereitung keine Wiederholung alter
Benchmarks. Ein neuer Vergleich muss den Extraktionshash mit einfrieren.

Die Grenze bleibt 20.000 Zeichen: längere Artikel werden abgelehnt, nicht gekürzt.
Die bisherige Qwen-Artikelpolicy lässt oberhalb von 3.072 Tokens den Mittelteil
weg; MiniLM mittelt Chunks. Beides bleibt als Modellgrenze sichtbar und ist noch
keine angemessene Vollartikelmethode. Der CPU-Profiler misst Tokenlängen, Chunkzahl
und den rechnerisch behaltenen Tokenanteil. Prompt-/Few-Shot-Overhead und eine
GPU-Kapazitätsprüfung folgen separat, bevor die konkreten Rezepte eingefroren werden.

HTTP-Verträge bleiben 422 (ungültig/nicht extrahierbar), 403 (robots-Verbot),
502 (Upstream-/Transportfehler) und 503 (Modell/Überlastung). Ein Upstream-403
bleibt API-502, erhält aber `X-Article-Error: upstream_forbidden`. DNS, Timeout,
Redirect und Extraktion haben unterscheidbare Kennungen. Die UI bietet Textpaste
und erhält URL, Text und Modellauswahl. DNS-Pinning, Privatadresssperre,
robots-Prüfung und Redirectverbot bleiben bestehen.

## Begrenzte Versuche und Einfrieren

Maschinenlesbarer Stand: `reports/next_stage/protocol.json`. Festgelegt sind
Rubrik, primäres DE/EN-Mittel, Schwelle 0,005 und der begrenzte Ablauf. Noch nicht
eingefroren sind Daten, konkrete Trainings-/Kontextrezepte und finale Kandidaten.
Keine Trainingsfunktion wird mit diesem Paket ausgeführt oder als fertig behauptet.

1. Vier Kernbedingungen auf drei gemeinsamen Train-Gruppenfolds: 12 Auswertungen.
2. Eine Modellfamilie, zwei zusätzliche geschachtelte Mengen: weitere 6.
3. Höchstens eine zusätzliche Forschungsfrage, zwei Bedingungen: höchstens 6.
4. Bei Bedarf zwei Finalisten mit einem weiteren Seed: höchstens 6.
5. Vollständige Train-Fits, dann ein frischer gemeinsamer Test für vier, höchstens
   fünf eingefrorene Kandidaten. Keine Optimierung auf diesem Test.

Das sind 24–30 Entwicklungs-Auswertungen; Fits und technische Prüfungen werden
getrennt gezählt. Kleinere Mengen werden nicht wegen eines guten Testwerts
übersprungen. 3.000 Train ist eine Obergrenze, 6.000 bleibt zurückgestellt.
Die angedachten 300 Testtexte je Sprache garantieren weder seltene Klassen noch
eine Auflösung von 0,005. Natürliche Verteilung, Klassenbesetzung, Gruppenbootstrap
und Annotationseinschränkungen werden mitberichtet. Schwache Ergebnisse sind
gültige Ergebnisse; 95 % und ein LLM-Sieg sind keine Pflichtannahmen.

`src.next_stage_protocol` stellt reine Hilfsfunktionen für Primärmaß, Auswahl und
vollständige Laufidentität bereit. Vergleichsgruppen hängen von Phase, Protokoll,
Eingaben, Annotationen und Split ab, nicht vom jeweiligen Kandidatenartefakt.
Neue MLflow-Läufe bleiben im bestehenden Experiment; das Paket schreibt keine
Läufe. Historische Ergebnisse werden nicht neu erzeugt oder überschrieben. Die Berichtsgeneratoren verweigern die ursprünglichen Berichtsordner und schreiben keine Metriken, Tags oder Artefakte in bestehende MLflow-Läufe; Reproduktionen benötigen eigene Ausgabepfade.

Transfer, Kaskade/Distillation und stärkere Encoder brauchen erst einen Befund.
Bei Transfer zunächst eine Modellfamilie mit kontrollierter Gesamtmenge und
Klassenverteilung; Zielsprachenlabels nicht für Beispiele, Auswahl oder Stopping
nutzen. Bereits gelesene Pilotfälle bleiben als bekannte Entwicklungsevidenz
gekennzeichnet. Quelle, Thema und Zeit sind mögliche Sprachkonfounder.

Vor GPU-Arbeit erneut `nvidia-smi`; ohne GPU kein Training. Belegte GPU-Ressourcen
werden vor einem isolierten Kapazitäts-/Messlauf koordiniert. Python 3.12 und die
bestehende CUDA-Umgebung verwenden, niemals `requirements-transformer.txt`.

## Prüfung und Auslieferung

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -p no:cacheprovider
```

Die Tests verwenden synthetische Texte und gemockte Modelle/Transporte. Live-URL-
Smokes und GPU-Arbeit sind separate Schritte; ein grüner Offline-Test behauptet
keine aktuelle Verlagszugänglichkeit oder Artikel-F1. Ein neues Standardartefakt
wird erst nach eingefrorenem Vergleich und Technikprüfung gewählt. Website,
README, API-Metadaten und Lizenzbestand werden dabei gemeinsam aktualisiert.

Historische Pläne mit menschlicher Pflichtannotation oder personenbezogenem
Vergleich sind überholt. Ihre Berichte und Quellenattribution bleiben im
[Projektarchiv](archive/project-history-through-v5.md) erhalten. Für die aktuelle
Phase gelten die Nutzerentscheidungen und dieses Protokoll; keine Präsentation.

## Erster Quellen-Checkpoint und Folgeschritt

Die Metadaten unter `reports/next_stage/pilot/` dokumentieren 90 native Texte
(30 DE, 30 EN, 30 JA), ihre exakten v3-Texthashes, zwei Eigenprüfungen,
Bezugsgegenstände, Begründungen, Mehrdeutigkeiten und Rechteentscheidungen.
Die Volltexte und HTML-Antworten bleiben im ignorierten lokalen Datenordner.
88/90 Labels stimmen zwischen den Durchgängen überein. Das ist **Eigenkonsistenz**,
keine unabhängige Labelgenauigkeit. Zwei anfangs unaufgelöste Fälle wurden mit
expliziter Perspektive aufgelöst; Hurrikan Nate bleibt mehrdeutig.

Nur 35 Texte wurden hinsichtlich Drittpassagen als kompatibel eingeschätzt:
17 DE, 5 EN, 13 JA. Die englischen fünf enthalten keinen positiven Fall.
Eine pauschale Freigabe wegen der Wikinews-Seitenlizenz wäre unzulässig; die
übrigen 55 bleiben mit Begründung sichtbar. Die gespeicherte Zeitspanne zwischen
Prüfungen enthält andere Arbeiten und ist keine aktive Arbeitszeitmessung.

Der Pilot zeigte außerdem störende japanische Themenkästen, englische
verwandte Meldungen und versteckte Originalzitat-Tooltips sowie deutsche
Kommentarlinks. `full-article-v4` behebt diese beobachteten Fälle. V3-Texte und
Annotationen bleiben unverändert. Es erfolgt keine automatische Labelübernahme
auf neu extrahierte Texte.

Für den separaten zweiten Pilot gelten vor Annotation und Modellvorhersagen:
100 weitere Texte je Sprache aus demselben eingefrorenen Kategorienpool,
ausgeschlossen sind die bereits geprüften 90 Seiten und direkte Zitate ab
80 Zeichen in DE/EN beziehungsweise 40 in JA. Dies ist eine konservative,
klassenblinde Quellenregel zur Machbarkeit, keine automatische Rechtefreigabe.
Die Einzelprüfung bleibt erforderlich. Die eingeschränkte Quellenpopulation
kann Themen und Klassen verschieben; spätere Ergebnisse dürfen deshalb nicht
als Genauigkeit auf beliebigen Nachrichtenartikeln dargestellt werden.


## Zweiter Pilot und eingefrorener Entwicklungsbestand

Der getrennte V4-Pilot umfasst 300 Artikel (100 je Sprache), jeweils zweimal
von Codex gelesen. 298 Klassenlabels blieben im zweiten Durchgang gleich. Das
ist Selbstkonsistenz desselben KI-Annotators, keine gemessene Labelgenauigkeit.
25 Artikel haben ungeklärte Dritttextrechte, ein weiteres englisches Label
bleibt ungelöst. Die vollständigen Prüfmetadaten stehen in
`reports/next_stage/pilot/checkpoint-300-v4.json`.

`full-article-v5` ergänzt zuvor fehlende Interview-Antworten, Tabellenbeschriftungen
und deklarierte Bild-Alternativtexte in Tabellen; Weblinks-Fußteile werden
beendet. Neun Texte ändern ihren Hash. Fünf grundsätzlich nutzbare Texte wurden
auf ihrem neuen Hash erneut zweimal gelesen. Die vier übrigen geänderten Texte
bleiben wegen offener Rechte gesperrt. Unveränderte Hashes übernehmen explizit
die beiden bisherigen Prüfungen. V4-Dateien bleiben erhalten.

Die ursprüngliche Metadatenbezeichnung `published_at` enthielt fälschlich den
Zeitpunkt der Seitenerstellung. Im neuen Entwicklungsbestand wird das sichtbare
Datum der fixierten Artikelfassung verwendet; die alte Angabe wird separat als
`page_created_at` dokumentiert. Das bestätigt weder Ereignisdatum noch Wahrheit
der Quelldatierung. Es gibt keinen zeitlichen Split auf diesen Angaben.

Nach zusätzlichen Sprachfassungs- und Ereignisprüfungen verbleiben 240
Train-Artikel: 78 DE, 71 EN und 91 JA. Alle Klassen sind je Sprache vertreten.
Der konservative Schutz schließt zusätzlich alle alten Wikinews-Sammlungen ein,
auch frühere Train-Seiten. 385 lexikalisch ähnliche Paare wurden anhand ihrer
identifizierten Ereignisse geprüft; 59 Verknüpfungen verbinden gleiche oder
eng zusammenhängende Ereignisreihen. Das Verfahren beweist keine vollständige
semantische Überschneidungsfreiheit. Ausschlüsse sind nachvollziehbar protokolliert.

Die geplante Obergrenze von 3000 ist kein belegtes Optimum. Das erste begrenzte
Volltext-Experiment nutzt den gesamten jetzt geeigneten Bestand; geschachtelte
Lernkurven prüfen die Mengenfrage. Kleine Klassen und eine einzige, vorab
eingeschränkte Quellenpopulation begrenzen die Aussagekraft.

Vor Kandidatenprognosen fixiert `development-protocol-v1.json` vier Kernbedingungen
über drei Falten (12 Auswertungen), zwei kleinere MiniLM-Trainmengen auf denselben
Dev-Falten (6) sowie MiniLM-Fenster mit 256 und 512 statt 128 Tokens (6). Alle
MiniLM-Bedingungen verwenden denselben Mittelwert der Chunk-Merkmale und einen
Artikelkopf; die Verlustfunktion greift auf Artikelebene. Qwen verarbeitet den
gesamten Text mit Überlängenfehler statt stiller Mittelkürzung. Ein frischer
Abschlusstest wurde noch nicht erhoben oder prognostiziert.
