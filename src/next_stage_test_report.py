"""Paired group-bootstrap and error analysis from one saved fresh-Test pass.

No model calls, relabelling, selection changes or additional scored conditions.
"""
import csv
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from src.config import ROOT, LABELS
from src.next_stage_collect import write_json
from src.next_stage_experiment import OUT, ART
from src.tracking import tracking


def macro_f1(y,pred):
    matrix=np.bincount(3*y+pred,minlength=9).reshape(3,3)
    denominator=matrix.sum(0)+matrix.sum(1)
    return float(np.divide(2*np.diag(matrix),denominator,out=np.zeros(3,dtype=float),where=denominator!=0).mean())


def reference_ambiguous(annotation):
    """Self-review disagreement remains a sensitivity flag after resolution."""
    first,second=annotation['first_pass'],annotation['second_pass']
    return bool(first['ambiguous'] or second['ambiguous'] or first['label']!=second['label'])


def group_bootstrap(table,predictions,selected,replicates=2000,seed=20261110):
    y=np.array([LABELS.index(x) for x in table.label]);languages=table.language.to_numpy()
    groups=[np.flatnonzero(table['group'].to_numpy()==group) for group in table['group'].unique()]
    rng=np.random.default_rng(seed);values={name:[] for name in predictions}
    for _ in range(replicates):
        indices=np.concatenate([groups[int(i)] for i in rng.integers(0,len(groups),len(groups))])
        for name,pred in predictions.items():
            per=[macro_f1(y[indices][languages[indices]==language],pred[indices][languages[indices]==language]) for language in ('de','en','ja')]
            values[name].append([per[0],per[1],per[2],(per[0]+per[1])/2])
    arrays={name:np.asarray(value) for name,value in values.items()}
    intervals={name:{metric:np.percentile(array[:,i],[2.5,97.5]).tolist() for i,metric in enumerate(['macro_f1_de','macro_f1_en','macro_f1_ja','primary_score'])} for name,array in arrays.items()}
    paired={name:np.percentile(arrays[selected][:,3]-array[:,3],[2.5,97.5]).tolist() for name,array in arrays.items() if name!=selected}
    return {'method':'Paired percentile bootstrap of whole observed Test event/source components; same resamples for all candidates',
        'replicates':replicates,'seed':seed,'observed_groups':len(groups),'confidence_level':.95,'intervals':intervals,'selected_minus_other_primary_delta_intervals':paired,
        'limits':'Conditional sampling uncertainty for the observed source population and fixed AI references. Does not quantify annotation error, unknown pretraining overlap or all grouping errors.'}


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=ROOT/'data/next_stage/final-test-v1';table=pd.read_csv(data/'test.csv');manifest=json.loads((OUT/'final-test-v1/test-manifest.json').read_text());results=json.loads((OUT/'final-test-v1.json').read_text())
    if json.loads((OUT/'final-test-evaluation.lock').read_text())['status']!='complete':raise RuntimeError('Finish all one-pass candidate evaluations before reporting')
    names=[r['model'] for r in results['models']];predictions={};by_model={}
    for name in names:
        rows=[json.loads(x) for x in (ART/'final-predictions'/(name+'.jsonl')).read_text().splitlines()]
        indexed={r['id']:r for r in rows}
        if len(indexed)!=len(rows) or set(indexed)!=set(table.id):raise RuntimeError('Saved candidate Test membership differs')
        for row in table.itertuples():
            r=indexed[row.id]
            if r['text_sha256']!=row.text_sha256 or r['reference']!=row.label:raise RuntimeError('Saved Test input/reference differs')
            scores=np.array(r['scores'])
            if scores.shape!=(3,) or not np.isfinite(scores).all() or (scores<0).any() or not np.isclose(scores.sum(),1.,atol=1e-4):raise RuntimeError('Invalid saved score distribution')
        by_model[name]=indexed;predictions[name]=np.array([LABELS.index(indexed[identifier]['prediction']) for identifier in table.id])
    intervals=group_bootstrap(table,predictions,results['selected_on_development']);write_json(OUT/'final-test-uncertainty-v1.json',intervals)
    annotations={a['id']:a for a in map(json.loads,(data/'annotations.jsonl').read_text().splitlines())}
    ambiguous=np.array([reference_ambiguous(annotations[identifier]) for identifier in table.id]);truth=np.array([LABELS.index(x) for x in table.label])
    sensitivity={name:{'all_reference_cases':len(table),'ambiguous_reference_cases':int(ambiguous.sum()),'clear_reference_cases':int((~ambiguous).sum()),
        'clear_de_en_primary_score':sum(macro_f1(truth[(~ambiguous)&(table.language.to_numpy()==lang)],pred[(~ambiguous)&(table.language.to_numpy()==lang)]) for lang in ('de','en'))/2,
        'ambiguity_rule':'Either review explicitly ambiguous or first/second class differs, even after resolution.',
        'interpretation':'Descriptive reference-ambiguity sensitivity from the same saved predictions; no relabel, rerun or selection change.'} for name,pred in predictions.items()}
    write_json(OUT/'final-test-reference-sensitivity-v1.json',sensitivity)
    tracking();client=MlflowClient();enriched=[]
    for result in results['models']:
        fit_metrics=client.get_run(result['run_id']).data.metrics
        per_language={}
        for language in ('de','en','ja'):
            mask=table.language.to_numpy()==language;y=truth[mask];p=predictions[result['model']][mask];matrix=np.bincount(3*y+p,minlength=9).reshape(3,3)
            per_language[language]={'rows':int(mask.sum()),'confusion_matrix':matrix.tolist(),'accuracy':float((y==p).mean()),'macro_f1':macro_f1(y,p)}
        enriched.append({**result,'training_seconds':fit_metrics.get('training_seconds',0.),'peak_training_allocated_bytes':fit_metrics.get('peak_training_allocated_bytes'),
                         'optimizer_steps':fit_metrics.get('optimizer_steps',0.),'per_language':per_language,'confidence_intervals_95':intervals['intervals'][result['model']]})
    final={**results,'models':enriched,'test_rows':len(table),'test_language_label_counts':manifest['language_label_counts'],
        'reference_scope':'AI-labelled · Codex-reviewed; two passes by the same AI annotator, no independent human validation',
        'input_scope':'Native quote-restricted Wikinews archive full texts, <=20000 characters and <=4096 Qwen article tokens',
        'source_rows':manifest['source_rows'],'excluded_source_rows':manifest['source_rows']-len(table),'retained_source_fraction':len(table)/manifest['source_rows'],
        'uncertainty':intervals,'reference_sensitivity':sensitivity,'selection':'Frozen development choice remains unchanged after Test',
        'test_inference':'One successful saved prediction per candidate and exact Test article; no repeated Test timing passes',
        'limits':['Restricted archive population and relatively short training articles; not all live publisher layouts or long interviews.',
                  'Unknown public pretraining overlap; observed grouping cannot prove complete semantic isolation.',
                  'Bootstrap intervals do not measure correctness of the AI reference labels; scores are uncalibrated.']}
    write_json(OUT/'comparison-v1.json',final)
    fields=['model','accuracy','primary_score','macro_f1_de','macro_f1_en','macro_f1_ja','negative_precision','negative_recall','positive_recall','inference_ms_per_text','latency_p50_ms','latency_p95_ms','full_model_size_bytes','training_seconds']
    with (OUT/'comparison-v1.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({key:r[key] for key in fields} for r in enriched)
    errors=[]
    for i,row in enumerate(table.itertuples()):
        if any(pred[i]!=truth[i] for pred in predictions.values()):
            review=annotations[row.id]['second_pass'];errors.append({'id':row.id,'title':row.title,'revision_url':row.revision_url,'reference':row.label,
                'reference_target':review['target'],'reference_ambiguous':reference_ambiguous(annotations[row.id]),'reference_ambiguity_note':review['ambiguity_note'],
                'reference_resolution_note':annotations[row.id]['resolution_note'],
                'predictions':{name:LABELS[int(pred[i])] for name,pred in predictions.items()},'annotation_unchanged':True})
    write_json(OUT/'error-analysis-v1.json',{'scope':'Saved Test predictions only, no annotation correction or tuning','cases':errors})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');positions=np.arange(len(enriched));scores=[r['primary_score'] for r in enriched]
    low=[scores[i]-r['confidence_intervals_95']['primary_score'][0] for i,r in enumerate(enriched)];high=[r['confidence_intervals_95']['primary_score'][1]-scores[i] for i,r in enumerate(enriched)]
    colors=['#285d39' if r['model']==results['selected_on_development'] else '#7198ad' for r in enriched]
    axes[0].bar(positions,scores,color=colors);axes[0].errorbar(positions,scores,yerr=np.maximum(0,np.array([low,high])),fmt='none',color='#243849',capsize=4);axes[0].set_ylim(0,1);axes[0].set_ylabel('Mittlerer DE/EN-macro-F1');axes[0].set_title('Ein gemeinsamer frischer Test · 95%-Bootstrapintervall')
    axes[1].bar(positions,[r['inference_ms_per_text'] for r in enriched],color=colors);axes[1].set_yscale('log');axes[1].set_ylabel('Warme ms/Artikel · logarithmisch');axes[1].set_title('Vollständiger Text · Batch 1 · ohne URL-Abruf')
    for ax in axes:ax.set_xticks(positions,names,rotation=20);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'{len(table)} native Vollartikel · KI-Referenzlabels · Auswahl vor Test eingefroren',fontsize=12);fig.savefig(OUT/'comparison-v1.png',dpi=180);fig.savefig(OUT/'comparison-v1.svg');plt.close(fig)
    lines=['# Vollartikel: gemeinsamer frischer Abschlusstest','',f"{len(table)} Testartikel aus {manifest['source_rows']} vorab erhobenen Quellenfällen. Alle lernenden Kandidaten sehen dieselben 240 Train-Artikel. Ausgewählt vor der Testbewertung: **{results['selected_on_development']}**. Kein Nachoptimieren am Test.",'',
        '**AI-labelled · Codex-reviewed**: zwei Durchgänge desselben KI-Annotators. Die Referenzlabels sind keine unabhängig bestätigte Human-Ground-Truth.','',
        '| Kandidat | DE/EN-macro-F1 | 95%-Intervall | DE | EN | JA | warme ms/Artikel | vollständige MiB |', '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in enriched:
        ci=r['confidence_intervals_95']['primary_score'];lines.append(f"| {r['model']} | {r['primary_score']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {r['macro_f1_de']:.4f} | {r['macro_f1_en']:.4f} | {r['macro_f1_ja']:.4f} | {r['inference_ms_per_text']:.3f} | {r['full_model_size_bytes']/1048576:.2f} |")
    lines+=['','![Gemeinsamer Testvergleich](comparison-v1.png)','','## Reichweite','',
        'Einmalige erfolgreiche Inferenz pro Testartikel und Kandidat; Aufwärmen ausschließlich mit Train-Texten. Latenz enthält Tokenisierung, Transfer, Kontext und Aggregation. Laden und URL-Abruf sind getrennt. Qwen-Größe enthält Basisgewichte und Adapter. Trainingszeit und Speicher stehen mit Laufkennungen, Klassenwerten und Konfusionsmatrizen im JSON/CSV.','',
        'Die Bootstrapintervalle verwenden gemeinsam gezogene vollständige beobachtete Ereignis-/Quellengruppen. Sie beschreiben Stichprobenunsicherheit bedingt auf diese KI-Referenzen; sie erfassen weder deren systematische Fehler noch unbekannte Vortrainingsüberschneidungen. Die kleine Train-Mengenkurve zeigte keine Sättigung; 240 ist kein belegtes Datenoptimum.','',
        'Die Population ist ein vorab eingeschränktes Archiv: native Artikel, begrenzte direkte Zitate, maximal 20000 Zeichen und 4096 Qwen-Artikeltokens. Offene Dritttextrechte, unauflösbare Labels und geschützte Ereignisverbindungen bleiben mit Ausschlussgrund erhalten. Die erste Auswahl bis zur Sprachquote verwendet Quellenreihenfolge und ganze Gruppen, keine nachträgliche Klassenbalance oder Modellfehler.','',
        'Alle früheren Vergleichsgruppen bleiben eigenständig. Diese Zahlen werden weder mit v1–v5 zu einer Fortschrittskurve verbunden noch als 95%-Qualitätsgarantie ausgegeben. Scores sind nicht kalibriert. Satzfarben in der Anwendung sind zusätzliche Vorhersagen und keine kausalen Worterklärungen.','',
        'Referenz-Mehrdeutigkeit und Fehlerfälle werden aus denselben gespeicherten Vorhersagen beschrieben. Labels und Kandidaten bleiben dabei unverändert.','','## Dateien','',
        '- `comparison-v1.json`: vollständige Ergebnisse, Laufkennungen, Kosten und Konfusionsmatrizen je Sprache.','- `final-test-v1/test-manifest.json`: exakte Hashes, Mitgliedschaft, Klassen, Quellenverluste und Schutzdateien.','- `final-test-uncertainty-v1.json`: gepaarte Gruppen-Bootstrapintervalle.','- `final-test-reference-sensitivity-v1.json`: Auswertung eindeutiger Referenzfälle ohne neue Inferenz.','- `error-analysis-v1.json`: unveränderte Referenzen, Bezugsgegenstände und Kandidatenfehler ohne Rohtextveröffentlichung.','']
    (OUT/'comparison-v1.md').write_text('\n'.join(lines));print('Reported one-pass Test results, paired group uncertainty and unchanged-reference error analysis.',flush=True)


if __name__=='__main__':report()
