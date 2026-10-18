"""Render measured ablations; no fabricated upward trend or test/CV mixing."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import REPORTS


def main():
    folder=Path(os.getenv('SENTIMENT_DEV_REPORTS',str(REPORTS/'development'))).resolve()
    decision=json.loads((folder/'decision.json').read_text())
    rows=decision['results']
    df=pd.DataFrame(rows)
    runs=pd.DataFrame(json.loads((folder/'runs.json').read_text()))
    labels=['B0\nFrozen','R0\nReference FT','R1\nBalanced loss','R2\nArticle','R3\nBF16 deploy','R4\nCombined']
    x=np.arange(len(df))
    colors=['#8894a2']+['#2166ac']*4+['#00876c']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(17,5.5))
    ax=axes[0]
    ax.errorbar(x,df.mean_macro_f1,yerr=df.sd_macro_f1,fmt='o-',color='#2166ac',capsize=4,label='mean ± fold SD')
    for i,row in df.iterrows():
        values=runs[runs.stage==row.stage].macro_f1
        ax.scatter([i]*len(values),values,color='#2166ac',alpha=.35,s=18)
        ax.annotate(f'{row.mean_macro_f1:.3f}',(i,row.mean_macro_f1),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    ax.set(title='Classification: source-group CV',ylabel='macro-F1',ylim=(max(0,float(df.mean_macro_f1.min()-df.sd_macro_f1.max()-.06)),min(1,float(df.mean_macro_f1.max()+df.sd_macro_f1.max()+.06))))
    ax.legend(fontsize=8,loc='lower right')
    ax=axes[1]
    ax.plot(x,df.synthetic_article_macro_f1,'o-',color='#d6604d',label='synthetic text macro-F1')
    ax.plot(x,df.html_fixture_exact_match,'s--',color='#00876c',label='HTML exact match (6 fixtures)')
    ax.set(title='Article checks: controlled fixtures',ylabel='score — not real-world accuracy',ylim=(0,1.1))
    ax.legend(fontsize=8,loc='lower right')
    ax=axes[2]
    ax.bar(x,df.median_latency_ms,color=colors,alpha=.85)
    ax.set(title='Deployment cost: warm CUDA',ylabel='ms/text (median of fold means)',ylim=(0,float(df.median_latency_ms.max()*1.45)))
    for i,row in df.iterrows():
        ax.text(i,row.median_latency_ms+.05,f'{row.median_latency_ms:.2f} ms\n{row.model_size_bytes/2**20:.0f} MiB',ha='center',va='bottom',fontsize=8)
    for ax in axes:
        ax.set_xticks(x,labels,fontsize=8)
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Development v2 — separate ablations, then combined',fontsize=15,fontweight='bold')
    fig.text(.02,.025,'Same 3 source-held-out folds inside original training data. Old test/Val untouched. Error bars: fold SD, not confidence intervals. Stages are ablations, not a guaranteed upward trajectory.',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94))
    fig.savefig(folder/'development_curve.png',dpi=180)
    fig.savefig(folder/'development_curve.svg')
    lines=['# Entwicklung v2: getrennte Ablationen und Kombination','',
           'Alle Werte stammen aus denselben drei Quellengruppen-Folds **innerhalb des ursprünglichen Train-Splits**. Die alten Testzahlen werden nicht in diese Kurve gemischt. Val und ursprünglicher Test blieben unberührt.','',
           '| Stufe | macro-F1 Mittel ± Fold-SD | DE | EN | Negativ-Recall | ms/Text | MiB | HTML exakt |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['stage']} | {r['mean_macro_f1']:.4f} ± {r['sd_macro_f1']:.4f} | {r['mean_macro_f1_de']:.4f} | {r['mean_macro_f1_en']:.4f} | {r['mean_negative_recall']:.4f} | {r['median_latency_ms']:.3f} | {r['model_size_bytes']/2**20:.1f} | {r['html_fixture_exact_match']:.0%} |")
    lines+=['','![Gemessene Entwicklung](development_curve.png)','',f"**Vor dem Kollegenvergleich ausgewählt: `{decision['selected']}`.** Regel: höchstes mittleres Quellen-CV-macro-F1; innerhalb 0,005 gewinnen Latenz, dann Größe.",'',
       'B0 ist je Fold neu gefittet; R0 ist der ursprüngliche Fine-Tune-Ansatz, ebenfalls ausschließlich auf dem jeweiligen Fit-Fold neu trainiert. Das ursprüngliche Volltrain-Modell wurde nicht auf Trainings-Folds bewertet. R1 ändert nur die Gewichtung von Klassen zu Sprache×Klasse; R2 nutzt R0-Gewichte mit tokengewichteter Chunk-Aggregation und verbesserter HTML-Extraktion; R3 speichert und rechnet R0 in BF16. R4 kombiniert R1, R2 und R3.','']
    by={r['stage']:r for r in rows}
    for stage,parent in [('R0_reference','B0_frozen'),('R1_quality','R0_reference'),('R2_article','R0_reference'),('R3_deploy','R0_reference'),('R4_combined','R0_reference')]:
        delta=by[stage]['mean_macro_f1']-by[parent]['mean_macro_f1']
        latency=by[stage]['median_latency_ms']-by[parent]['median_latency_ms']
        lines.append(f"- `{stage}` gegenüber `{parent}`: Δmacro-F1 {delta:+.4f}, Δms/Text {latency:+.3f}. Kein automatischer Vorteil durch Kombination unterstellt.")
    lines+=['','Die 18 synthetischen Artikeltexte und sechs HTML-Fixtures sind kontrollierte technische Prüfungen, keine unabhängig von Menschen gelabelten Nachrichtenartikel. Deshalb trägt deren hohe Trefferquote keinen Anspruch auf reale Artikelqualität. Quellen-CV verhindert dieselbe Quelle in Fit/Dev, aber nicht dasselbe Ereignis über verschiedene Verlage. Drei Folds mit einem festen Seed ergeben Fold-Streuung, keinen Signifikanznachweis. Ein unabhängiger Benchmark folgt erst nach dieser Auswahl.','',
       'Der ursprüngliche v1-Test und seine einmalige Auswertung bleiben als historischer Meilenstein erhalten. Zusätzliche Fine-Tuning-Ablationen sind die ausdrücklich angeforderte Erweiterung des ursprünglichen Ein-Modell-Protokolls. Der Harness-Abbruch ist in `harness_incident.json` dokumentiert und als fehlgeschlagener MLflow-Lauf erhalten.','',
       'Reproduktion: `python -m src.development run`, danach `summarize`, `release`, `activate`, `python -m src.service_benchmark` und `python -m src.development_report`. Auf diesem Tower sind abgeschlossene Läufe gesperrt; ein frischer Clone hat keine lokalen Daten/Modelle. Vorher Daten und die ursprüngliche eingefrorene Baseline mit `src.prepare` / `src.train frozen` herstellen. Für frische Reproduktionen `SENTIMENT_DEV_REPORTS=reports/reproduction` exportieren; so bleiben eingecheckte historische Messergebnisse unberührt. Die ursprünglichen v1-Test- und Fine-Tuning-Befehle sind für v2 nicht nötig.','',
       'MLflow: Experiment `article-sentiment-second-path`, Tags `phase=development-v2`, `stage`, `fold`. Jede Fold-Auswertung enthält Accuracy, Klassen-F1, macro-F1, Sprachwerte, Negativ-Precision/Recall, Latenz, Bytes und Konfusionsmatrix. `decision.json`, `runs.json`, `curve.csv`, `folds.json` und vollständige lokale Modellartefakte machen jede Entscheidung nachvollziehbar.','',
       'Methodische Referenzen: [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [PyTorch mixed precision](https://docs.pytorch.org/docs/stable/amp.html).']
    if (folder/'service.json').exists():
        service=json.loads((folder/'service.json').read_text())
        lines+=['','## Lokaler Last- und Erholungstest','',service['note'],'',
                '| Profil | Steady p50/p95 ms | Burst 200 / 503 | Burst p95 ms | Erholt |',
                '| --- | ---: | ---: | ---: | --- |']
        for r in service['results']:
            lines.append(f"| {r['profile']} | {r['steady_p50_ms']:.2f} / {r['steady_p95_ms']:.2f} | {r['burst_success_200']} / {r['burst_busy_503']} | {r['burst_p95_all_ms']:.2f} | {r['recovered']} |")
    (folder/'development.md').write_text('\n'.join(lines)+'\n')
    if (folder/'branch_comparison.json').exists():
        branch_report(folder)
    # Matplotlib path lines contain trailing spaces; keep generated text diffs clean.
    for name in ['development_curve.svg','branch_comparison.svg']:
        svg=folder/name
        if svg.exists():
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    print('Measured chart and development.md written')

def branch_report(folder):
    result=json.loads((folder/'branch_comparison.json').read_text())
    rows=result['results']
    by={r['model']:r for r in rows}
    order=['colleague_main_default','second_path_v1','second_path_v2']
    labels=['Colleague main\nreconstructed default','Our v1\noriginal release','Our v2\nselected release']
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    x=np.arange(3)
    ax=axes[0]
    en=[by[k]['macro_f1_en'] for k in order]
    de=[by[k]['macro_f1_de'] for k in order]
    bars=ax.bar(x-.18,en,.36,color='#2166ac',label='Native English (human labels)')
    other=ax.bar(x+.18,de,.36,color='#e6a54a',hatch='//',label='German machine translation')
    for b in list(bars)+list(other):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+.012,f'{b.get_height():.3f}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=labels,ylabel='macro-F1',ylim=(0,.9),title='Same final external headlines')
    ax.legend(fontsize=8,loc='upper left')
    ax=axes[1]
    for key,label,color in zip(order,['Colleague main','Our v1','Our v2'],['#8894a2','#2166ac','#00876c']):
        r=by[key]
        ax.scatter(r['inference_ms_per_text'],r['model_size_bytes']/2**20,s=100,color=color)
        ax.annotate(label,(r['inference_ms_per_text'],r['model_size_bytes']/2**20),xytext=(7,5),textcoords='offset points')
    ax.set(xscale='log',xlim=(.15,4),ylim=(-20,520),xlabel='ms/text (log scale; CPU for main, CUDA for ours)',ylabel='Complete artifact MiB',title='Measured deployment trade-off')
    axes[1].set_xticks([.25,.5,1,2,4],['0.25','0.5','1','2','4'])
    axes[1].minorticks_off()
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Branch comparison — only after our choice was frozen',fontweight='bold')
    fig.text(.02,.02,'600 native English headlines + their German translations. Not 1200 independent articles. Different training corpora; compares delivered methods, not encoder architecture alone.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.94))
    fig.savefig(folder/'branch_comparison.png',dpi=180)
    fig.savefig(folder/'branch_comparison.svg')
    ci=result['native_en_paired_bootstrap_95_percentile_interval']
    decision=json.loads((folder/'decision.json').read_text())
    chosen=next(r for r in decision['results'] if r['stage']==decision['selected'])
    best=max(decision['results'],key=lambda r:r['mean_macro_f1'])
    lines=['# Abschließender Vergleich mit dem anderen Branch','',
        '**Erst nach Abschluss der Ablationen, Auswahl und vollständigem Release-Training ausgeführt.** Die externe Bewertung änderte weder Auswahl noch Modelle.','',
        '| Lösung | macro-F1 EN nativ | macro-F1 DE übersetzt | macro-F1 gemeinsam (sekundär) | ms/Text | MiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key in order:
        r=by[key]
        lines.append(f"| {key} | {r['macro_f1_en']:.4f} | {r['macro_f1_de']:.4f} | {r['macro_f1']:.4f} | {r['inference_ms_per_text']:.3f} | {r['model_size_bytes']/2**20:.2f} |")
    lines+=['','![Gemeinsamer externer Vergleich](branch_comparison.png)','',
        f"Auf den 600 nativen englischen Headlines liegt v2 um {result['native_en_delta_v2_minus_peer']:+.4f} macro-F1 vor dem rekonstruierten Kollegen-Standard. Ein gepaarter Bootstrap über diese 600 bereits vorhergesagten Headlines (1000 Ziehungen, Seed 20261019) ergibt ein 95-%-Perzentilintervall der Differenz [{ci[0]:.4f}, {ci[1]:.4f}]. Keine zusätzlichen Modellvorhersagen oder Modellwahl durch Bootstrap.",'',
        f"**Kein pauschaler Genauigkeitsgewinn von v2 gegenüber v1 behauptet:** Englisch bewegt sich von {by['second_path_v1']['macro_f1_en']:.4f} auf {by['second_path_v2']['macro_f1_en']:.4f}; übersetztes Deutsch von {by['second_path_v1']['macro_f1_de']:.4f} auf {by['second_path_v2']['macro_f1_de']:.4f}. Die Artefaktgröße bewegt sich von {by['second_path_v1']['model_size_bytes']/2**20:.1f} auf {by['second_path_v2']['model_size_bytes']/2**20:.1f} MiB. Der Unterschied Englisch v1/v2 wird hier nicht als signifikant bezeichnet.",'',
        'Der Kollegen-Standard ist erheblich kleiner und schneller auf CPU. Unser Encoder nutzt CUDA und ist trotz BF16 größer und langsamer. Für das vorgegebene Qualitätsziel spricht dieser externe Benchmark für den Encoder; für enge CPU-/Speicherbudgets bleibt LinearSVC ein klarer Trade-off. Das ist keine allgemeine Rangliste für andere Nachrichtendomänen.','',
        '## Herkunft, Filter, Grenzen','',
        '[SEntFiN](https://github.com/pyRis/SEntFiN), Revision `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`, ursprüngliches Repo unter [MIT](https://github.com/pyRis/SEntFiN/blob/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/LICENSE); Sinha, Kedas, Kumar, Malo. Von 10.753 englischen Headlines bleiben 9.512 mit einem einheitlichen Sentiment für sämtliche genannten Entitäten und gültiger Länge. Nach normalisierten Duplikaten und Prüfung gegen beide Trainingskorpora bleiben 9.453; kein exakter normalisierter Trainingsüberlapp wurde gefunden. Je 200 negative, neutrale und positive Headlines werden ohne Modellvorhersagen mit Seed 20261019 gezogen.','',
        'Die englischen Labels stammen aus der menschlichen Entitätsannotation. Das Übertragen auf den ganzen Titel erfolgt nur bei einstimmigen Entitätslabels; das bleibt eine Operationalisierung, keine neue Annotation. Die 600 deutschen Texte werden aus genau diesen Titeln mit [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de), CC-BY-4.0, Revision `6183067f769a302e3861815543b9f312c71b0ca4` übersetzt (Beam 4, kein Sampling, Kontext 256, max. 128 neue Token). Kein englischer Quelltext wurde durch den Kontext gekürzt. Deutsche Labels sind übernommen: Übersetzungsfehler und eine veränderte Tonalität wurden nicht unabhängig geprüft. **DE ist deshalb ein MT-Transfercheck, kein nativer deutscher Goldtest.**','',
        'Der Benchmark ist klassenbalanciert, besteht aus Headlines und hat 600 unterschiedliche Titelpaare. Er repräsentiert weder natürliche Nachrichtenhäufigkeiten noch vollständig gelabelte Artikel. Semantischer Ereignisüberlapp und Vortrainingskontamination sind unbekannt. Die alte Testzahl 0,720 und die alte eigene Zahl 0,7187 werden nicht zur Rangbildung benutzt.','',
        '## Unveränderte Referenz, kein nachgetragener Lauf auf main','',
        'Der Referenzcode liegt ausschließlich in einem separaten detached Checkout auf `a840bb7bc8f1656f1700ccabaa89837ce347ca2b`. Der Adapter importiert `new_linearsvc` und die originale `prepare()`-Funktion. Er kopiert keine Modellimplementierung in second-path und ruft niemals `src.train` oder das ursprüngliche MLflow-Experiment auf. Referenz-Trainingsdaten: 5.923 Zeilen aus Phrasebank EN und Kenpache DE; pandas 3.0.6, scikit-learn 1.9.1 und joblib 1.6.0 wie in deren Requirements. Unsere Modelle trainieren auf 8.335 Kenpache-DE/EN-Zeilen. Der Vergleich trennt daher nicht den Effekt der Architektur von der Trainingsdatenwahl.','',
        'Die ursprünglichen trainierten Gewichte des Kollegen waren nicht verfügbar. Das Ergebnis ist eine Rekonstruktion des gelieferten Standardverfahrens mit unverändertem Code und erneut hergestelltem Trainingssplit; es wird weder Bitgleichheit zum damaligen Artefakt noch zum damaligen Testsplit behauptet. Code-, Daten- und Artefakthashes stehen in `peer_provenance.json`, die Referenz-Abhängigkeiten in `reference_requirements.txt`.','',
        'Alle drei neuen gemeinsamen Auswertungen liegen ausschließlich im Experiment `article-sentiment-second-path`, Tag `phase=final-branch-comparison`. Die Hauptbranch-Referenz und ihr MLflow-Verzeichnis bleiben unverändert.','',
        '## Entscheidungen nach Themen','',
        '| Thema | Beobachtung / Entscheidung |',
        '| --- | --- |',
        '| Qualität der Evidenz | Quellengetrennte Train-CV und zusätzlicher externer Audit statt eines weiteren Blicks auf den alten Test. Kein Beleg für allgemeine Artikelqualität. |',
        '| Trainingsgewichtung | Sprache×Klasse verändert die CV-Werte nur gering; kein belastbarer großer Genauigkeitsgewinn. |',
        '| Artikelverarbeitung | R2/R4 bestehen sechs statt zwei HTML-Fixtures; technische Verbesserung, kein menschlich gelabelter Artikelnachweis. Nicht Bestandteil des nach Zeit-Tie-Regel gewählten R3-Standards. |',
        '| Betrieb | R3/BF16 liefert ca. 241 statt 465 MiB; begrenztes Warten und Prüfung endlicher Scores verhindern zwei konkrete Betriebsfehler. Geringe Latenzunterschiede sind keine Signifikanzbehauptung. |',
        f"| Kombination / Auswahl | Höchster CV-Punktwert: {best['stage']} mit {best['mean_macro_f1']:.4f}. Gewählt nach der vorab erklärten Nähe-/Latenzregel: {chosen['stage']} mit {chosen['mean_macro_f1']:.4f} und {chosen['median_latency_ms']:.3f} ms. Kein nachträgliches Umdefinieren, um alles kombiniert auszuliefern. |",
        '| Kollege vs. second-path | Auf diesem gemeinsamen nativen EN-Benchmark bessere Encoder-Qualität gegen deutlich kleineren/schnelleren CPU-Standard. Andere Daten, Kosten und Sprachdomänen können die Entscheidung ändern. |']
    lines+=['','## Frisch reproduzieren (erst nach v2-Auswahl)','',
        'In einem frischen Clone nach den v2-Befehlen im README. Der v1-Trainingsaufruf erzeugt nur den Vergleichsstand, keine alte Testbewertung. Auf dem aktuellen Tower sind die Artefakte bereits vorhanden und werden nicht überschrieben. Neue Berichte bleiben von den historischen getrennt.','',
        '```bash',
        'export SENTIMENT_DEV_REPORTS=reports/reproduction',
        'python -m pip install -r requirements-evaluation.txt',
        'python -m src.train finetuned',
        'git clone --no-checkout https://github.com/Rasvan-Teofil/ai-operations-projekt.git ../colleague-reference',
        'git -C ../colleague-reference checkout --detach a840bb7bc8f1656f1700ccabaa89837ce347ca2b',
        'python3.12 -m venv ../colleague-reference/.venv',
        '../colleague-reference/.venv/bin/python -m pip install -r reports/development/reference_requirements.txt',
        '../colleague-reference/.venv/bin/python src/peer_reference.py prepare --checkout ../colleague-reference --report reports/reproduction/peer_provenance.json',
        'python -m src.branch_benchmark prepare',
        'python -m src.branch_benchmark audit',
        'python -m src.development_report',
        '```','',
        'Referenzdaten werden durch den Adapter auf die protokollierten HF-Snapshots gepinnt; die originale Filter-/Split-/Modellimplementierung bleibt unangetastet. Der Benchmark besitzt eine exklusive lokale Audit-Sperre. Originale Test-/Val-Dateien werden von unserer Entwicklung nicht als Bewertungsdaten verwendet.']
    (folder/'branch_comparison.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
