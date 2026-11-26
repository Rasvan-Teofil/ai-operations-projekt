"""Render the audited optimizer recipe/clipping comparison; no model inference."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/optimizer_validation'
NAMES={'baseline':'Bisheriges Rezept','candidate':'Neuer Kandidat','candidate_no_clip':'Neues Rezept ohne Clipping'}


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    summary=json.loads((OUT/'summary-v1.json').read_text())
    verification=json.loads((OUT/'verification-v1.json').read_text())
    if summary['state']!='complete' or verification['status']!='complete' or not verification.get('bootstrap_2000_paired_group_draws_independently_reproduced_with_sklearn_weights'):
        raise RuntimeError('Render only after complete independent evidence audit')
    a=summary['analysis'];seeds=list(a['per_seed']);recipes=list(NAMES)
    metrics=['primary_score','macro_f1_de','macro_f1_en','macro_f1_ja'];metric_names=['DE/EN','DE','EN','JA']
    means={recipe:{m:float(np.mean([a['per_seed'][s]['oof'][recipe][m] for s in seeds])) for m in metrics} for recipe in recipes}
    gain=a['mean_contrasts'];intervals=a['conditional_group_bootstrap_95_intervals']
    supported=a['clipping_supported_in_this_exploratory_cv'];gate=all(a['practical_recipe_gate'].values())
    clip_statement=('Der vorab festgelegte deskriptive Clipping-Nachweis ist in dieser explorativen CV erfüllt.' if supported else
                    'Der vorab festgelegte deskriptive Clipping-Nachweis ist in dieser explorativen CV nicht erfüllt.')
    recipe_statement=('Das neue Gesamtrezept erfüllt die vorab festgelegte praktische Konsistenzregel.' if gate else
                      'Das neue Gesamtrezept erfüllt die vorab festgelegte praktische Konsistenzregel nicht; der bisherige API-Standard bleibt die Empfehlung.')
    lines=['# Optimizer-Validierung: Ergebnis und Modellentscheidung','',clip_statement,recipe_statement,'',
        '**27/27 Auswertungen abgeschlossen: 21 neue Fits und sechs exakt übernommene bestehende Fits.** Drei gruppierte Folds, drei feste Seeds, 240 Train-Artikel in 235 semantischen Gruppen. Kein erneuter Abschlusstest, kein automatischer Modellwechsel.','',
        '**Was verglichen wurde:** Bisheriges Rezept und neuer Kandidat verwenden beide Clipping 1,0. Ihr Unterschied betrifft Lernrate, Weight Decay und Warmup. Der Vergleich des neuen Rezepts mit und ohne Clipping isoliert den Clipping-Schalter.','',
        '![Gepaarte Optimizer-Validierung](optimizer-validation-v1.svg)','',
        '## Gepoolte Out-of-fold-Güte','',
        'Pro Seed und Rezept werden zunächst alle 240 zurückgehaltenen Vorhersagen aus den drei Folds gepoolt. Erst danach wird Macro-F1 berechnet. Die folgende Tabelle mittelt die drei festen Seed-Werte. Der DE/EN-Wert gewichtet beide Sprachen gleich.','',
        '| Rezept | DE/EN Macro-F1 | DE | EN | JA |','|---|---:|---:|---:|---:|']
    for recipe in recipes:lines.append('| '+NAMES[recipe]+' | '+' | '.join(f'{means[recipe][m]:.4f}' for m in metrics)+' |')
    lines+=['','## Gepaarte Unterschiede','',
        'Absolute F1-Prozentpunkte; positive Werte sprechen für den neuen Kandidaten. Intervalle sind bedingte 95%-Gruppen-Bootstrap-Intervalle mit 2000 festen Ziehungen, keine um Auswahlverzerrung bereinigten Bestätigungsintervalle.','',
        '| Vergleich | DE/EN Δ [Intervall] | DE Δ | EN Δ | JA Δ |','|---|---:|---:|---:|---:|']
    for name,label in [('recipe','Neues Gesamtrezept − bisheriges Rezept'),('clipping','Neues Rezept mit − ohne Clipping')]:
        lo,hi=intervals[name]['primary_score']
        lines.append(f'| {label} | {gain[name]["primary_score"]*100:+.2f} [{lo*100:+.2f}; {hi*100:+.2f}] | '+' | '.join(f'{gain[name][m]*100:+.2f}' for m in metrics[1:])+' |')
    lines+=['','| Seed | Gesamtrezept Δ DE/EN (pp) | Clipping Δ DE/EN (pp) |','|---|---:|---:|']
    for seed in seeds:
        v=a['per_seed'][seed]['contrasts'];lines.append(f'| {seed} | {v["recipe"]["primary_score"]*100:+.2f} | {v["clipping"]["primary_score"]*100:+.2f} |')
    lines+=['','## Kontrolle außerhalb des Auswahlfolds','',
        'Fold 0 war bereits zur Rezeptwahl verwendet worden. Nur aus den übrigen Folds 1 und 2 gepoolt ergeben sich folgende Seed-gemittelten Unterschiede. Auch diese wiederverwendeten Train-Daten sind kein unabhängiger Abschlusstest.','',
        '| Vergleich | DE/EN Δ (pp) | DE Δ | EN Δ | JA Δ |','|---|---:|---:|---:|---:|']
    for name,label in [('recipe','Gesamtrezept'),('clipping','Clipping')]:lines.append('| '+label+' | '+' | '.join(f'{a["excluding_selection_fold_mean_contrasts"][name][m]*100:+.2f}' for m in metrics)+' |')
    lines+=['','## Vorab festgelegte Entscheidung','',
        '| Regel für das neue Gesamtrezept | Erfüllt |','|---|---|']
    gate_labels={'mean_primary_gain_at_least_one_pp':'Mindestens +1 pp im mittleren DE/EN-Wert','nonnegative_primary_each_seed':'Kein negativer DE/EN-Unterschied in einem der drei Seeds','no_mean_language_regression_over_one_pp':'Keine Sprache verliert im Mittel mehr als 1 pp'}
    for key,label in gate_labels.items():lines.append(f'| {label} | {"ja" if a["practical_recipe_gate"][key] else "nein"} |')
    lines+=['',
        'Clipping-Unterstützung verlangt gleichzeitig einen positiven mittleren DE/EN-Unterschied, positive DE/EN-Unterschiede in allen drei Seeds und eine positive untere bedingte Bootstrap-Grenze. Ein Nichterfüllen ist kein Beweis für Wirkungslosigkeit, sondern ein nicht ausreichend robuster positiver Nachweis nach dieser Regel.','',
        '## Was das für das gespeicherte Modell bedeutet','',
        'Der bereits trainierte Adapter `article-qwen-lora-optimizer-v1` ist technisch funktionsfähig und bleibt optional über `article_qwen_lora_optimizer_v1` ausführbar. Seine Gewichte wurden durch diese Prüfung nicht verändert. Gemessen wurde die Trainingsrezeptur in 159/160/161-Artikel-Fits; die obigen F1-Werte sind keine direkt gemessene unabhängige Testgüte des auf allen 240 Artikeln trainierten Adapters.','',
        'Die ursprüngliche Auswahl und der frühere gemeinsame Test bleiben erhalten. Es gibt keinen nachträglichen Austausch der Testwerte. Ein unabhängiger Qualitätsnachweis für den Volltrain-Adapter erfordert eine gesondert geplante frische Evaluation.','',
        '## Was Clipping technisch beeinflusst','',
        'Alle Fits verwenden AdamW. Der frühere Faktorvergleich untersucht dessen Weight Decay, keinen Vergleich mit einer anderen Optimizerfamilie.','',
        'Clipping begrenzt hier die gemeinsame L2-Norm der akkumulierten LoRA-Gradienten vor dem AdamW-Schritt. Der protokollierte Gradientenwert ist die Norm vor dem Eingriff. [PyTorch 2.11: clip_grad_norm_](https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.clip_grad_norm_.html).','',
        'AdamW verwendet laufende erste und zweite Gradientenmomente sowie einen getrennten Weight-Decay-Schritt. Deshalb ist ein kleinerer Gradient keine Garantie für eine entsprechend kleinere Parameteränderung oder höhere F1. Die gemessenen Update-Normen und Clipping-Anteile sind Diagnostik; sie beweisen den Mechanismus eines Güteunterschieds nicht. [PyTorch 2.11: AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html).','',
        '| Rezept | Mittlerer Anteil geclippter Schritte | Mittlere Update-L2-Norm |','|---|---:|---:|']
    for recipe in recipes:
        rows=[r for r in summary['runs'] if r['config']['recipe']==recipe]
        lines.append(f'| {NAMES[recipe]} | {np.mean([r["metrics"]["clip_fraction"] for r in rows]):.1%} | {np.mean([r["metrics"]["mean_adapter_update_l2"] for r in rows]):.5f} |')
    lines+=['','## Nachweise und Grenzen','',
        '- [Festes Protokoll](protocol-v1.json), [Korrektur vor erstem Trainingsschritt](engineering-amendment-v1.json), [vollständige Ergebnisse](summary-v1.json), [separate Verifikation](verification-v1.json).',
        '- [Öffentliches Original-MLflow · Validierung](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/4), [Volltrain-Modelllauf](https://aquarium-markers-messaging-specialists.trycloudflare.com/#/experiments/3/runs/5230309ca351479980869ef61fc239f0).',
        '- Unter `runs/`, `predictions/` und `telemetry/` stehen alle 27 Auswertungen; sechs verweisen auf ihre ursprünglichen Fits. Keine Rohartikel oder Modellgewichte in Git.',
        '- Ein zusätzlicher fehlgeschlagener MLflow-Setup-Eintrag enthält null Trainingsschritte und keine Vorhersagen. Er ist kein weiterer Fit und bleibt als Ausführungshistorie erhalten.',
        '- Kleine, wiederverwendete Stichprobe; bereits bekannte Projektergebnisse; Auswahl des Rezepts auf Fold 0; drei feste Seeds und überlappende Trainingsmengen. Keine unabhängigen Fold-/Seed-Stichproben, keine p-Werte, keine kausale Mechanismusaussage.',
        '- Gruppen-Bootstrap resampelt dieselben semantischen Gruppen gemeinsam für alle Arme und Seeds. Er beschreibt bedingte Artikelunsicherheit und nicht die gesamte Unsicherheit aus Rezeptwahl und Modelltraining.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    table=['fold,seed,recipe,primary_score,macro_f1_de,macro_f1_en,macro_f1_ja,evidence_origin,mlflow_run_id']
    for r in summary['runs']:
        c=r['config'];table.append(','.join(map(str,[c['fold'],c['seed'],c['recipe'],*[r['metrics'][m] for m in metrics],r['evidence_origin'],r['mlflow_run_id']])))
    (OUT/'fold-results-v1.csv').write_text('\n'.join(table)+'\n')
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={'width_ratios':[1.05,1]})
    colors=['#596773','#2463ad','#d28130'];x=np.arange(4);width=.23
    for i,recipe in enumerate(recipes):
        ax.bar(x+(i-1)*width,[means[recipe][m] for m in metrics],width,color=colors[i],label=NAMES[recipe],alpha=.85)
        for j,seed in enumerate(seeds):ax.scatter(x+(i-1)*width+(j-1)*.035,[a['per_seed'][seed]['oof'][recipe][m] for m in metrics],color='black',s=12,alpha=.5,zorder=3)
    ax.set(xticks=x,xticklabels=metric_names,ylim=(0,1),ylabel='Macro-F1',title='Gepoolte OOF-Güte · Mittel über 3 Seeds')
    ax.legend(loc='lower left',fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for j,(name,label,color) in enumerate([('recipe','Gesamtrezept','#2463ad'),('clipping','Clipping isoliert','#d28130')]):
        pos=x+(j-.5)*.2;point=np.array([gain[name][m]*100 for m in metrics]);ci=np.array([intervals[name][m] for m in metrics]).T*100
        bx.vlines(pos,ci[0],ci[1],color=color,linewidth=2);bx.scatter(pos,point,color=color,s=40,label=label,zorder=3)
    bx.axhline(0,color='#555',linewidth=.9);bx.set(xticks=x,xticklabels=metric_names,ylabel='Δ Macro-F1 (Prozentpunkte)',title='Gepaarte Unterschiede · bedingte 95%-Intervalle')
    bx.grid(axis='y',alpha=.2);bx.legend(fontsize=9)
    fig.suptitle('Optimizer-Validierung · 3 Folds × 3 Seeds × 3 Rezepte',fontsize=14)
    fig.text(.5,.02,'Explorative Train-CV nach Rezeptwahl; kein unabhängiger Test des 240-Artikel-Adapters. Punkte links: einzelne Seeds.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93))
    fig.savefig(OUT/'optimizer-validation-v1.svg');fig.savefig(OUT/'optimizer-validation-v1.png',dpi=160);plt.close(fig)
    print(clip_statement,recipe_statement)


if __name__=='__main__':main()
