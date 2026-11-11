"""Reports from saved full-article predictions only; never runs a model."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.config import ROOT
from src.next_stage_experiment import OUT, ART
from src.next_stage_collect import write_json
from src.next_stage_prepare import frozen_write


def development():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((OUT/'development-v1.json').read_text());rows=result['out_of_fold'];runs=result['runs']
    fields=['model','research_question','primary_score','macro_f1_de','macro_f1_en','macro_f1_ja','fold_primary_sd','inference_ms_per_text','full_model_size_bytes']
    with (OUT/'development-summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in rows)
    lines=['# Vollartikel: Entwicklung innerhalb Train','',
        '240 Artikel (78 DE, 71 EN, 91 JA), drei gruppierte Falten. Primärwert: Mittel von DE- und EN-macro-F1 aus den zusammengesetzten Out-of-fold-Vorhersagen; jede Train-Zeile wird je Bedingung genau einmal außerhalb ihres Fit-Teils bewertet. Kein frischer Abschlusstest in diesem Bericht.',
        '', '**AI-labelled · Codex-reviewed**: ein KI-Annotator mit Eigenprüfung; keine unabhängig geprüften Humanlabels.', '',
        '| Bedingung | DE/EN-macro-F1 | DE | EN | JA | Falten-SD des Primärwerts | warme ms/Artikel |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+r['model']+' | '+' | '.join(f'{r[k]:.4f}' for k in ['primary_score','macro_f1_de','macro_f1_en','macro_f1_ja','fold_primary_sd'])+f" | {r['inference_ms_per_text']:.3f} |")
    lines += ['', '## Was diese Messung zeigt','',
        '- Die größere MiniLM-Trainmenge verbessert den OOF-Primärwert in dieser festen Rezeptur. Die Kurve zeigt noch keine Sättigung. 240 ist der vollständige geeignete Pilotbestand und kein belegtes Optimum oder Nachweis ausreichender Produktionsdaten.',
        '- Die Mengenbedingungen verwenden im Mittel etwa 57, 110 und 160 Fit-Artikel je Falte auf identischen Dev-Teilen. Die finale vollständige Train-Menge ist 240. Tokenzahlen und Optimiererschritte sind in den zugehörigen Läufen gespeichert.',
        '- Ein größeres MiniLM-Fenster verbessert den Primärwert hier nicht. Andere Daten, Rezepte oder ausreichend lange Texte könnten anders ausfallen; eine allgemeine Aussage über Kontextlänge folgt daraus nicht.',
        '- Qwen LoRA und Qwen ohne Anpassung liegen nahe beieinander. Der kleine Abstand beweist keine statistisch gesicherte Überlegenheit. Die vorab festgelegte optionale Seed-Prüfung wird getrennt protokolliert.',
        '- Alle Aussagen gelten für eine einzelne, vorab eingeschränkte Wikinews-Population. Insbesondere umfangreiche Drittzitate und Interviews fehlen häufig nach der Rechteprüfung. Unbekannte Vortrainingsüberschneidungen bleiben möglich.',
        '', '## Schutz der Vergleichsgruppen','',
        'Frühere Val-/Test-Dateien und konservativ alle alten Wikinews-Sammlungen sind geschützt. Sprachfassungen, Ursprungsquellen und geprüfte Ereignisreihen werden transitiv verbunden. Zusätzliche FIFA- und Noto-Ereignisaliase lagen bereits vollständig in jeweils derselben bestehenden Falte; Text, Label und Mitgliedschaft wurden nicht geändert. Das Verfahren garantiert keine vollständige semantische Duplikaterkennung.',
        '', 'Falten-SD ist kein Konfidenzintervall. Latenzen umfassen Tokenisierung, Transfer, vollständigen Kontext und Aggregation; Laden und URL-Abruf sind ausgeschlossen. Die Referenzlabels und eine mögliche KI-Fehlerkorrelation begrenzen jede Qualitätsinterpretation.',
        '', '24 Entwicklungsauswertungen: 12 Kernbedingungen, 6 Mengenbedingungen und 6 Kontextbedingungen. Technische Kapazitätsprüfungen, optionale Seed-Prüfung und spätere vollständige Train-Fits werden separat gezählt.',
        '', '![Entwicklungsvergleich](development-summary.png)', '']
    (OUT/'development-summary.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    core=[r for r in rows if r['research_question']=='core'];axes[0].bar([r['model'] for r in core],[r['primary_score'] for r in core],color=['#8193a8','#477fa6','#6c9a70','#285d39']);axes[0].set_title('Kernvergleich · OOF');axes[0].set_ylim(0,1);axes[0].tick_params(axis='x',rotation=25)
    order=['minilm-third','minilm-two-thirds','minilm128'];x=[np.mean([r['fit_rows'] for r in runs if r['model']==name]) for name in order];y=[next(r['primary_score'] for r in rows if r['model']==name) for name in order]
    axes[1].plot(x,y,'o-',color='#477fa6');axes[1].set_title('Menge · gleiche Dev-Falten');axes[1].set_xlabel('Mittlere Fit-Artikel je Falte');axes[1].set_ylim(0,1)
    context=[next(r for r in rows if r['model']==name) for name in ['minilm128','minilm256','minilm512']];axes[2].bar(['128','256','512'],[r['primary_score'] for r in context],color='#477fa6');axes[2].set_title('MiniLM · Kontextfenster');axes[2].set_xlabel('Tokens je Chunk');axes[2].set_ylim(0,1)
    for ax in axes:ax.set_ylabel('Mittlerer DE/EN-macro-F1');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Vollartikel · 240 Train-Artikel · KI-Referenzlabels · kein Abschlusstest',fontsize=12)
    fig.savefig(OUT/'development-summary.png',dpi=180);fig.savefig(OUT/'development-summary.svg');plt.close(fig)
    print('Wrote development summary from fixed saved results; no inference.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['development']);args=parser.parse_args();development()
