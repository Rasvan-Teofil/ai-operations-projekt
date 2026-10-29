"""Render measured ablations; no fabricated upward trend or test/CV mixing."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from second_path.config import REPORTS


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
    lines=['# Development v2: separate ablations and combination','',
           'All values come from the same three source-group folds **inside the original train split**. The old test scores are not mixed into this curve. Validation and the original test stayed untouched.','',
           '| Stage | macro-F1 mean ± fold SD | DE | EN | Negative recall | ms/text | MiB | HTML exact |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['stage']} | {r['mean_macro_f1']:.4f} ± {r['sd_macro_f1']:.4f} | {r['mean_macro_f1_de']:.4f} | {r['mean_macro_f1_en']:.4f} | {r['mean_negative_recall']:.4f} | {r['median_latency_ms']:.3f} | {r['model_size_bytes']/2**20:.1f} | {r['html_fixture_exact_match']:.0%} |")
    lines+=['','![Measured development](development_curve.png)','',f"**Selected before the comparison with the first iteration: `{decision['selected']}`.** Rule: highest mean source-CV macro-F1; within 0.005, latency wins, then size.",'',
       'B0 is refitted per fold; R0 is the original fine-tuning approach, also retrained only on the respective fit fold. The original full-train model was not evaluated on training folds. R1 only changes the weighting from class to language×class; R2 uses R0 weights with token-weighted chunk aggregation and improved HTML extraction; R3 stores and computes R0 in BF16. R4 combines R1, R2 and R3.','']
    by={r['stage']:r for r in rows}
    for stage,parent in [('R0_reference','B0_frozen'),('R1_quality','R0_reference'),('R2_article','R0_reference'),('R3_deploy','R0_reference'),('R4_combined','R0_reference')]:
        delta=by[stage]['mean_macro_f1']-by[parent]['mean_macro_f1']
        latency=by[stage]['median_latency_ms']-by[parent]['median_latency_ms']
        lines.append(f"- `{stage}` vs. `{parent}`: Δmacro-F1 {delta:+.4f}, Δms/text {latency:+.3f}. No automatic benefit from combining is assumed.")
    lines+=['','The 18 synthetic article texts and six HTML fixtures are controlled technical checks, not news articles labelled independently by humans. Their high hit rate therefore makes no claim about real-world article quality. Source CV keeps the same source out of both fit and dev, but not the same event across different publishers. Three folds with one fixed seed give fold spread, not a significance test. An independent benchmark follows only after this selection.','',
       'The original v1 test and its one-off evaluation are kept as a historical milestone. The additional fine-tuning ablations are the explicitly requested extension of the original single-model protocol. The harness abort is documented in `harness_incident.json` and kept as a failed MLflow run.','',
       'Reproduction: `python -m second_path.development run`, then `summarize`, `release`, `activate`, `python -m second_path.service_benchmark` and `python -m second_path.development_report`. On this tower, completed runs are locked; a fresh clone has no local data or models. First create the data and the original frozen baseline with `second_path.prepare` / `second_path.train frozen`. For fresh reproductions export `SENTIMENT_DEV_REPORTS=reports/reproduction` so the committed historical results stay untouched. The original v1 test and fine-tuning commands are not needed for v2.','',
       'MLflow: experiment `article-sentiment-second-path`, tags `phase=development-v2`, `stage`, `fold`. Every fold evaluation contains accuracy, per-class F1, macro-F1, per-language scores, negative precision/recall, latency, bytes and confusion matrix. `decision.json`, `runs.json`, `curve.csv`, `folds.json` and complete local model artefacts make every decision traceable.','',
       'Methodological references: [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html), [PyTorch mixed precision](https://docs.pytorch.org/docs/stable/amp.html).']
    if (folder/'service.json').exists():
        service=json.loads((folder/'service.json').read_text())
        lines+=['','## Local load and recovery test','',service['note'],'',
                '| Profile | Steady p50/p95 ms | Burst 200 / 503 | Burst p95 ms | Recovered |',
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
    labels=['Iteration 1\nLinearSVC (reconstructed)','Iteration 2 v1\noriginal release','Iteration 2 v2\nselected release']
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
    for key,label,color in zip(order,['Iteration 1','Iteration 2 v1','Iteration 2 v2'],['#8894a2','#2166ac','#00876c']):
        r=by[key]
        ax.scatter(r['inference_ms_per_text'],r['model_size_bytes']/2**20,s=100,color=color)
        ax.annotate(label,(r['inference_ms_per_text'],r['model_size_bytes']/2**20),xytext=(7,5),textcoords='offset points')
    ax.set(xscale='log',xlim=(.15,4),ylim=(-20,520),xlabel='ms/text (log scale; CPU for iteration 1, CUDA for iteration 2)',ylabel='Complete artifact MiB',title='Measured deployment trade-off')
    axes[1].set_xticks([.25,.5,1,2,4],['0.25','0.5','1','2','4'])
    axes[1].minorticks_off()
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Iteration 1 vs. 2 — only after our choice was frozen',fontweight='bold')
    fig.text(.02,.02,'600 native English headlines + their German translations. Not 1200 independent articles. Different training corpora; compares delivered methods, not encoder architecture alone.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.94))
    fig.savefig(folder/'branch_comparison.png',dpi=180)
    fig.savefig(folder/'branch_comparison.svg')
    ci=result['native_en_paired_bootstrap_95_percentile_interval']
    decision=json.loads((folder/'decision.json').read_text())
    chosen=next(r for r in decision['results'] if r['stage']==decision['selected'])
    best=max(decision['results'],key=lambda r:r['mean_macro_f1'])
    lines=['# Final comparison of the first and second iteration','',
        '**Run only after the ablations, the selection and the full release training were finished.** The external evaluation changed neither the selection nor the models.','',
        '| Solution | macro-F1 EN native | macro-F1 DE translated | macro-F1 combined (secondary) | ms/text | MiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key in order:
        r=by[key]
        lines.append(f"| {key} | {r['macro_f1_en']:.4f} | {r['macro_f1_de']:.4f} | {r['macro_f1']:.4f} | {r['inference_ms_per_text']:.3f} | {r['model_size_bytes']/2**20:.2f} |")
    lines+=['','`colleague_main_default` is our first iteration (TF-IDF + LinearSVC, commit `a840bb7`), reconstructed. The name is the key in the JSON/CSV files.','','![Shared external comparison](branch_comparison.png)','',
        f"On the 600 native English headlines, v2 is {result['native_en_delta_v2_minus_peer']:+.4f} macro-F1 ahead of our reconstructed first iteration (LinearSVC). A paired bootstrap over these 600 already predicted headlines (1,000 resamples, seed 20261019) gives a 95 % percentile interval for the difference of [{ci[0]:.4f}, {ci[1]:.4f}]. No additional model predictions or model selection via bootstrap.",'',
        f"**No general accuracy gain of v2 over v1 is claimed:** English moves from {by['second_path_v1']['macro_f1_en']:.4f} to {by['second_path_v2']['macro_f1_en']:.4f}; translated German from {by['second_path_v1']['macro_f1_de']:.4f} to {by['second_path_v2']['macro_f1_de']:.4f}. Artefact size moves from {by['second_path_v1']['model_size_bytes']/2**20:.1f} to {by['second_path_v2']['model_size_bytes']/2**20:.1f} MiB. The English v1/v2 difference is not called significant here.",'',
        'Our first iteration (LinearSVC) is considerably smaller and faster on CPU. Our encoder uses CUDA and, despite BF16, is larger and slower. For the given quality goal this external benchmark favours the encoder; for tight CPU/memory budgets LinearSVC remains a clear trade-off. This is not a general ranking for other news domains.','',
        '## Provenance, filters, limits','',
        '[SEntFiN](https://github.com/pyRis/SEntFiN), revision `eba43610fe3cb57a3e5773cd6c037da1f8992ad5`, original repo under [MIT](https://github.com/pyRis/SEntFiN/blob/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/LICENSE); Sinha, Kedas, Kumar, Malo. Of 10,753 English headlines, 9,512 remain with one consistent sentiment for all mentioned entities and a valid length. After removing normalised duplicates and checking against both training corpora, 9,453 remain; no exact normalised training overlap was found. 200 negative, 200 neutral and 200 positive headlines are drawn without model predictions, seed 20261019.','',
        'The English labels come from human entity annotation. They are transferred to the whole headline only when all entity labels agree; this remains an operationalisation, not a new annotation. The 600 German texts are translated from exactly these headlines with [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de), CC-BY-4.0, revision `6183067f769a302e3861815543b9f312c71b0ca4` (beam 4, no sampling, context 256, max. 128 new tokens). No English source text was truncated by the context. German labels are carried over: translation errors and shifted tone were not checked independently. **DE is therefore an MT transfer check, not a native German gold test.**','',
        'The benchmark is class-balanced, consists of headlines and has 600 distinct headline pairs. It represents neither natural news frequencies nor fully labelled articles. Semantic event overlap and pre-training contamination are unknown. The old test score 0.720 and our old own score 0.7187 are not used for ranking.','',
        '## Unchanged reference, no retrofitted run on main','',
        'The reference code lives only in a separate detached checkout at `a840bb7bc8f1656f1700ccabaa89837ce347ca2b`. The adapter imports `new_linearsvc` and the original `prepare()` function. It copies no model implementation into second-path and never calls `src.train` or the original MLflow experiment. Reference training data: 5,923 rows from Phrasebank EN and Kenpache DE; pandas 3.0.6, scikit-learn 1.9.1 and joblib 1.6.0 as in its requirements. Our models train on 8,335 Kenpache DE/EN rows. The comparison therefore does not separate the effect of the architecture from the choice of training data.','',
        'The trained weights of the first iteration were not available. The result is a reconstruction of the delivered default procedure with unchanged code and a re-created training split; bit-identity with the original artefact or test split is not claimed. Code, data and artefact hashes are in `peer_provenance.json`, the reference dependencies in `reference_requirements.txt`.','',
        'All three new shared evaluations live only in the experiment `article-sentiment-second-path`, tag `phase=final-branch-comparison`. The main-branch reference and its MLflow directory stay unchanged.','',
        '## Decisions by topic','',
        '| Topic | Observation / decision |',
        '| --- | --- |',
        '| Quality of evidence | Source-separated train CV plus an external audit instead of another look at the old test. No evidence of general article quality. |',
        '| Training weights | Language×class changes the CV scores only slightly; no robust large accuracy gain. |',
        '| Article processing | R2/R4 pass six instead of two HTML fixtures; a technical improvement, not evidence from human-labelled articles. Not part of the R3 default chosen by the latency tie rule. |',
        '| Operations | R3/BF16 takes about 241 instead of 465 MiB; bounded waiting and a finite-score check prevent two concrete operational failures. Small latency differences are not a significance claim. |',
        f"| Combination / selection | Highest CV point estimate: {best['stage']} with {best['mean_macro_f1']:.4f}. Chosen by the pre-declared tie/latency rule: {chosen['stage']} with {chosen['mean_macro_f1']:.4f} and {chosen['median_latency_ms']:.3f} ms. No after-the-fact redefinition to ship everything combined. |",
        '| First vs. second iteration | On this shared native EN benchmark, better encoder quality vs. a much smaller/faster CPU default. Other data, costs and language domains can change the decision. |']
    lines+=['','## Reproduce from scratch (only after the v2 selection)','',
        'In a fresh clone, after the v2 commands in the README. The v1 training call only produces the comparison state, not an old test evaluation. On the current tower the artefacts already exist and are not overwritten. New reports stay separate from the historical ones.','',
        '```bash',
        'export SENTIMENT_DEV_REPORTS=reports/reproduction',
        'python -m pip install -r requirements-evaluation.txt',
        'python -m second_path.train finetuned',
        'git clone --no-checkout https://github.com/Rasvan-Teofil/ai-operations-projekt.git ../colleague-reference',
        'git -C ../colleague-reference checkout --detach a840bb7bc8f1656f1700ccabaa89837ce347ca2b',
        'python3.12 -m venv ../colleague-reference/.venv',
        '../colleague-reference/.venv/bin/python -m pip install -r reports/development/reference_requirements.txt',
        '../colleague-reference/.venv/bin/python second_path/peer_reference.py prepare --checkout ../colleague-reference --report reports/reproduction/peer_provenance.json',
        'python -m second_path.branch_benchmark prepare',
        'python -m second_path.branch_benchmark audit',
        'python -m second_path.development_report',
        '```','',
        'The adapter pins reference data to the logged HF snapshots; the original filter/split/model implementation stays untouched. The benchmark has an exclusive local audit lock. Original test/val files are not used as evaluation data by our development.']
    (folder/'branch_comparison.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__': main()
