# GPU-Ausfälle: Systemanalyse vom 26. November 2026

Die Grafikkarte ist wiederholt vollständig aus dem Zugriff des Treibers
verschwunden. **Vier getrennte Ausfallereignisse derselben physischen GPU sind
in den erhaltenen Kerneljournalen belegt, bereits vor dieser Studie und über
zwei Kernel-/Treibergenerationen hinweg.** Welches Bauteil oder welcher
Firmware-/Treiberpfad die Ausfälle auslöst, ist damit noch nicht bewiesen.
Die Karte wird deshalb nicht allein aufgrund von Xid 79 als defekt bezeichnet.

## Lokale Belege und Zeitleiste

Es wurden 78.800 Kernelmeldungen aus 35 erhaltenen Systemstarts ausgewertet.
Physische Zuordnung über die NVIDIA-GPU-UUID, nicht nur über die wiederverwendbare
PCI-Adresse. Die UUID und vollständige Rohprotokolle bleiben lokal.

| Zeitpunkt (Europe/Berlin) | Kernel | NVIDIA-Treiber | Befund |
|---|---|---|---|
| 20.04.2026, 22:46 | Banner nicht erhalten | Banner nicht erhalten | Xid 79; dieselbe GPU-UUID |
| 21.04.2026, 09:54 | 6.17.0-20 | 580.126.09 | Xid 79/154; dieselbe GPU-UUID |
| 17.11.2026, 20:48 | 7.0.0-38 | 595.99.02 | Drei Xid-79-Meldungen innerhalb eines Ereignisses, danach Xid 154 |
| 26.11.2026, 10:21 | 7.0.0-38 | 595.99.02 | Xid 79/154 während des ersten Optimizer-Fits; CUDA/cuBLAS scheitert, Neustart erforderlich |

Insgesamt sechs Xid-79-, vier Xid-154- und eine Xid-31-Meldung. Die drei Meldungen
vom 17. November werden nicht als drei unabhängige Ausfälle gezählt. Der frühere
Xid 31 stammt vom 20. April um 22:57 mit Xorg als gemeldetem Prozess; daraus
wird keine Aussage über den damaligen gesamten GPU-Workload abgeleitet.

Der Tower beantwortete nach dem heutigen GPU-Ausfall weiterhin Befehle. Die
Bildausgabe war verloren, der gesamte Rechner aber nicht unmittelbar stromlos.
Nach dem durch den Benutzer veranlassten Aus-/Einschalten startete der geprüfte
Einmal-Retry. Mehrere vollständige Fits sind seitdem gelungen. Das beweist
Betriebsfähigkeit über diesen Zeitraum, keine dauerhafte Fehlerfreiheit.

[NVIDIA beschreibt Xid 79](https://docs.nvidia.com/deploy/xid-errors/archive/index.html#xid-79-gpu-has-fallen-off-the-bus)
als nicht mehr erreichbare GPU über PCIe; sowohl Verbindungs-/Hardwareprobleme
als auch Treiberprobleme kommen infrage. Ein cuBLAS-Fehler in derselben Situation
ist daher keine eigenständige Diagnose eines Optimizer- oder CUDA-Bibliotheksfehlers.

## Ausgelesener Systemstand

| Bereich | Tatsächlicher Befund | Aussagegrenze |
|---|---|---|
| GPU | GeForce RTX 5060 Ti, 16 GiB; Gigabyte-Subsystem 1458:418f | Kommerzielle GV-Artikelnummer nicht gemeldet; Board Part Number und FRU sind N/A |
| GPU-Firmware | VBIOS 98.06.39.80.63; GSP 595.99.02 | Keine Aussage, ob ein exakt passendes OEM-Firmwareupdate existiert |
| Mainboard/CPU | ASUS ProArt B650-CREATOR, Ryzen 5 7600 | Keine physische Kontaktprüfung aus dem Betriebssystem möglich |
| Mainboard-BIOS | 2904 vom 04.03.2025 | Neuer ist nicht automatisch ein nachgewiesener Fix |
| Grafiktreiber | Geladenes Modul, GSP und Benutzerbibliotheken 595.99.02 | Kein Versionsmischstand gefunden |
| Kernel | Aktiv 7.0.0-38; 7.0.0-43 installiert; 7.0.0-44 mit passendem Modul verfügbar | Vorhandene GRUB_TOP_LEVEL-Festlegung hält bewusst den früher geprüften Kernel aktiv |
| Paketverwaltung | dpkg-Audit sauber; letzte protokollierte Grafiktreiberaktualisierung 19.10.2026 | Kein belegter Grafiktreiberwechsel unmittelbar vor dem Ausfall |
| Laufende CUDA-Bibliotheken | cuBLAS/CUDA-Runtime aus der Projekt-venv, Host-libcuda 595.99.02 | Das separat installierte System-Toolkit 13.1 ist nicht die gemappte Trainings-Runtime |
| GPU-Tuning | Ausgelesene Grafik- und Speichertakt-Offsets jeweils 0; Power-Limit 180 W | Factory-Boost, GPU-Spannungsstabilität und UEFI-OC damit nicht vollständig geprüft |
| Hauptspeicher | Ausgelesene EDAC-Zähler und gefundene MCE-/EDAC-Fehlermarker unauffällig | Kein vollständiger RAM-Test, keine Garantie für Fehlerfreiheit |

Die Root-Diagnose über die normale Systemauthentifizierung war erfolgreich.
`dmidecode --type 39` liefert **keinen Netzteil-Datensatz**. Auch USB-, hwmon-
und power_supply-Inventar zeigen keine passende Netzteil-Telemetrie. Hersteller,
Modell, Wattzahl und Qualität des angeschlossenen Netzteils sind somit weiterhin
nicht ausgelesen. Das gilt auch mit Administratorzugriff. Ein passiver Riser
kann sich gegenüber dem Betriebssystem wie eine direkte Verbindung verhalten.
NVIDIAs Feld „Direct Connected“ ist kein Nachweis, dass kein passives Kabel existiert.

## PCIe: konkreter Befund und Beobachtungslücke

Firmware-Slotinventar und PCI-Baum ordnen die Karte dem primären **PCIEX16_1**
am CPU-Port `00:01.1` zu. Sie steckt laut elektronischem Inventar nicht im
unteren Chipsatz-x4-Slot. Beide erkannten NVMe-Laufwerke haben eigene CPU-Ports.

GPU und Root-Port melden unter Last **16 GT/s, vier ausgehandelte Lanes**.
Gen4 ist auf diesem [Mainboard normal](https://www.asus.com/motherboards-components/motherboards/proart/proart-b650-creator/techspec/).
Die ungeklärte Breite x4 ist davon getrennt. Die angezeigte maximale Fähigkeit
x16 beweist keine x16-Verdrahtung des konkreten GPU-PCBs. Eine Soll-Lanezahl für
die noch nicht eindeutig identifizierte Verkaufsversion wird nicht erfunden.

Vorhandene lokale Wartungsnachweise dokumentieren bereits: erzwungenes primäres
X16-Routing, ein Gen3-Vergleich sowie spätere Rückkehr auf Auto und ein Kaltstart
haben x4 nicht beseitigt. Diese früheren kurzen Bandbreitenversuche waren kein
Langzeit-Stabilitätsvergleich und schließen die damaligen Einstellungen als
Ausfallursache nicht aus. Sie werden hier nicht blind wiederholt.

Die privilegiert ausgelesenen Register zeigen **ASPM L1 aktiv** auf beiden Enden.
Die GPU bietet AER mit aktuell leeren Fehlerstatusregistern und Nullzählern.
Der tatsächliche CPU-Root-Port bietet in seiner ausgelesenen Extended-Capability-
Liste dagegen **keine AER-Capability**; Root-AER-Zähler fehlen. Global hat ACPI
zwar AER-Kontrolle an Linux übergeben. Das ersetzt aber keine Capability dieses
konkreten Ports. Laut [Linux-AER-Dokumentation](https://www.kernel.org/doc/html/latest/PCI/pcieaer-howto.html)
bindet der AER-Treiber nur an unterstützende Ports. Fehlende AER-Meldungen sind
hier somit kein belastbarer Freispruch für den gesamten PCIe-Signalweg.

## Recherche eingeordnet

Im [NVIDIA-Fehlertracker](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1151)
finden sich ähnliche Verluste von Blackwell-GPUs, einschließlich RTX 5060 Ti und
Berichten unter neueren Treibern. Die Berichte betreffen verschiedene Systeme,
Symptome und teils mehrere zugleich geänderte Bedingungen. Sie begründen keinen
bestätigten gemeinsamen Fehler und keinen sicheren Fix für diesen Rechner.
Ein [separater GB206-Bericht](https://forums.developer.nvidia.com/t/three-rtx-5060-ti-gb206-randomly-fallen-off-the-bus-xid-79-154-at-any-load-level-no-pcie-aer-precursor-open-kernel-module-595-99-02/384551)
meldet Fortbestehen nach einem kleinen Treiberupdate. Das ist ergänzende Evidenz,
kein Ausschluss eines lokalen Hardwareproblems.

NVIDIAs Mitarbeiter stellt in einer häufig referenzierten Diskussion ausdrücklich
klar, dass die dort anfangs zusammengefassten Fälle unterschiedliche Ursachen
haben und [separat verfolgt werden](https://forums.developer.nvidia.com/t/deterministic-xid-120-gsp-kernel-panic-on-rtx-pro-6000-blackwell-under-sustained-cublas-workload-595-84/376750/17).
Die Bugnummer 6426268 wird deshalb nicht als bestätigte Erklärung dieses Falls verwendet.

Für Blackwell unterstützt NVIDIA ausschließlich
[offene Kernelmodule](https://download.nvidia.com/XFree86/Linux-x86_64/595.99.02/README/kernel_open.html).
Ein pauschaler Wechsel zum geschlossenen Kernelmodul ist daher kein geeigneter
Reparaturvorschlag. Ebenso wird ein Abschalten von GSP nicht als Blackwell-Fix empfohlen.

ASUS bietet [BIOS 4003 vom 22.09.2026](https://www.asus.com/de/motherboards-components/motherboards/proart/proart-b650-creator/helpdesk_bios?model2Name=ProArt-B650-CREATOR)
mit neuerem AGESA und allgemeinen Stabilitätsänderungen an; eine passende
Xid-79-/x4-Reparatur ist daraus nicht belegt. Einige Zwischenversionen schließen
Rollback aus. Es wurde keine Firmware geflasht.
NVIDIAs [5060-UEFI-Updater](https://nvidia.custhelp.com/app/answers/detail/a_id/5665/)
adressiert Bildausfälle beim Neustart auf bestimmten Systemen. Der hier belegte
Laufzeit-Ausfall mit weiterlaufendem Linux ist eine andere Beobachtung.

## Arbeitshypothesen und nächster unterscheidender Test

| Hypothese | Dafür / dagegen | Geeigneter nächster Test |
|---|---|---|
| Kontakt, passiver Riser, Karten-/Mainboard-Signalweg | Ungeklärte x4-Breite und Gerätverlust; fehlende Root-AER-Sichtbarkeit. Kein konkretes Bauteil bewiesen. | Nach Beendigung der Fits, stromlos: Anschlussweg und Verriegelung prüfen; bestehenden Riser gegebenenfalls umgehen; Karte kontrolliert neu einsetzen; danach Breite und Stabilität erneut messen. |
| Karten-/Versorgungsproblem | Wiederholungen derselben GPU über Versionen hinweg; Netzteil und Steckverbindungen unbekannt. | Bekannt gutes Netzteil mit passenden Originalkabeln oder Karte in bekannt gutem Rechner; alternativ andere GPU im gleichen System. Ein Faktor je Vergleich. |
| Treiber-/GSP-/UEFI-Fehler | Ähnliche öffentliche Berichte, vorhandene alte Firmware; verschiedene Treiber teilen weiterhin Code und Firmwarepfade. | Nach Abschluss des unveränderten Experiments zuerst passenden neueren Kernel separat mit Rückfall-Boot prüfen; Treiberwechsel als eigenen Vergleich behandeln. BIOS erst nach gesicherter Identität und dokumentiertem Wiederherstellungsweg. |
| Temperatur / Speichererschöpfung | Bisherige Nach-Neustart-Samples unauffällig; rund 9–10 GiB von 16 GiB belegt. Temperatur beim ursprünglichen Ausfall nicht lückenlos gespeichert. | Laufenden Mitschnitt beim nächsten Ereignis auswerten; VRAM-/Speichertest erst exklusiv bei freier GPU. |
| Aktuelle Lernrate/Clipping allein | Derselbe Gerätverlust trat Monate vor der Studie auf. | Keine Anpassung der eingefrorenen Studie als vermeintliche Hardware-Reparatur. Trainingscode und finaler Test bleiben unverändert. |

Der aussagekräftigste Bauteilvergleich ist: Folgt der Fehler der GPU in einen
anderen stabilen Rechner, oder bleibt er bei einer anderen GPU im ursprünglichen
System? Damit lässt sich mehr unterscheiden als durch gleichzeitige BIOS-,
Kernel-, Treiber- und Spannungsänderungen. Keine Netzteil-Modularkabel zwischen
verschiedenen Netzteilen mischen. Interne Kontakte nur bei ausgeschaltetem,
vom Netz getrenntem Tower prüfen.

## Bereits umgesetzt und reproduzierbar

- Private Kernelbelege, PCIe-Register, Firmwareinventar und offizieller
  NVIDIA-Bugreport im sicheren Modus gesichert; Collector-Exitcode 0.
- Identische GPU-UUID über die historischen Ereignisse geprüft. Öffentlicher
  [strukturierter Befund](gpu-investigation-v1.json) enthält keine UUID, Seriennummern
  oder vollständigen Systemlogs. Rohbelege liegen ignoriert unter
  `runtime/gpu-investigation-2026-11-26/`.
- Begrenzter Monitor `article-gpu-health-capture.service`: alle fünf Sekunden
  Temperatur, Leistung, Belegung, Takt, P-State, Linkbreite/-geschwindigkeit,
  CPU-Temperatur und verfügbare AER-Zähler; paralleler Kernel-Mitschnitt.
  Maximal drei Stunden, Ende bei abgeschlossenem/fehlgeschlagenem Worker oder
  GPU-Abfragefehler; kein automatischer Neustart und keine GPU-Steuerung.
- Keine Kernel-, Treiber-, Firmware-, BIOS-, ASPM-, Takt- oder Power-Limit-Änderung
  während dieser Diagnose. Der Mitschnitt begann während Lauf 4; Unterschiede
  der Trainingslaufzeiten nicht als sauberen Hardware-Benchmark interpretieren.

```bash
# Nur lesen, private neue Ausgabedatei; erfasst alle erhaltenen Boots:
.venv/bin/python scripts/analyze_gpu_journal.py --output runtime/gpu-journal-new.json
# Privilegierte, einmalige Lesediagnose über Systemauthentifizierung:
pkexec /bin/sh "$PWD/scripts/capture_gpu_privileged.sh"
# Aktuelle zeitlich begrenzte Aufzeichnung:
systemctl --user status article-gpu-health-capture.service
```

Vollständiger Herstellerbericht ist lokal verfügbar und kann nach Durchsicht
gezielt an den Support gegeben werden. Er wurde nicht hochgeladen oder versendet.
Die vollständige Bauteilursache bleibt bis zu den unterscheidenden Tests offen.
