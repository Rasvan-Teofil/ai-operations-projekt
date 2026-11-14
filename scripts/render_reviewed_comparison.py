"""Render an updated presentation caption from saved results; never run a model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    source = ROOT / 'reports/next_stage/comparison-v1.json'
    review_path = ROOT / 'reports/data-review/project-review-v1.json'
    report = json.loads(source.read_text()); review = json.loads(review_path.read_text())
    rows = report['models']; positions = np.arange(len(rows))
    scores = np.array([r['primary_score'] for r in rows])
    lower = scores - [r['confidence_intervals_95']['primary_score'][0] for r in rows]
    upper = [r['confidence_intervals_95']['primary_score'][1] for r in rows] - scores
    colors = ['#285d39' if r['model'] == report['selected_on_development'] else '#7198ad' for r in rows]
    matplotlib.rcParams['svg.hashsalt'] = 'project-review-presentation-v1'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    axes[0].bar(positions, scores, color=colors)
    axes[0].errorbar(positions, scores, yerr=np.maximum(0, np.array([lower, upper])),
                     fmt='none', color='#243849', capsize=4)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel('Mittlerer DE/EN-macro-F1')
    axes[0].set_title('Gespeicherter Abschlusstest · 95%-Bootstrapintervall')
    axes[1].bar(positions, [r['inference_ms_per_text'] for r in rows], color=colors)
    axes[1].set_yscale('log'); axes[1].set_ylabel('Warme ms/Artikel · logarithmisch')
    axes[1].set_title('Vollständiger Text · Batch 1 · ohne URL-Abruf')
    for ax in axes:
        ax.set_xticks(positions, [r['model'] for r in rows], rotation=20)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('691 native Vollartikel · Projektseitig geprüfte Referenzen\n'
                 'Unveränderte Messung · Reviewbestätigung 14.11.2026 · Auswahl vor Test', fontsize=11)
    out = ROOT / 'reports/data-review'; out.mkdir(exist_ok=True)
    png = out / 'comparison-reviewed-v1.png'; svg = out / 'comparison-reviewed-v1.svg'
    fig.savefig(png, dpi=180, metadata={'Software': 'Article Sentiment: stored-results presentation v1'})
    fig.savefig(svg, metadata={'Date': None}); plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    manifest = {
        'version': 'reviewed-comparison-presentation-v1', 'model_calls': 0,
        'source_results': str(source.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'review_record': str(review_path.relative_to(ROOT)),
        'review_record_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
        'reviewed_on': review['reviewed_on'],
        'scope': 'Presentation caption updated after project-owner review. Same saved metrics and intervals; original figure and result files unchanged.',
        'output_files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (png, svg)},
    }
    (out / 'comparison-presentation-v1.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('Reviewed comparison rendered from saved metrics; zero inference, original report unchanged.')


if __name__ == '__main__':
    main()
