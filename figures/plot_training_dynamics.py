"""Reproduce the six-panel final-run figure from committed training/probe records."""
from pathlib import Path
import hashlib
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
ARCHIVE_ROOT = Path(os.environ.get('C3_EXPERIMENT_ARCHIVE', ROOT / 'experiments')).resolve()
RUNS = {
    'C3-OPD': 'c3_search_qrho1_beta01_50u_20261011',
    'GRPO+OPSD': 'grpo_opsd_no_verifier_50u_20261013',
}
KEYS = ['episode/success_rate', 'c3/delta_mean', 'actor/grad_norm']
VERIFIER_KEYS = ['c3/c_dist_mean', 'c3/v_dist_mean']
COLORS = ['#0072B2', '#D55E00']
STYLES = ['-', (0, (4, 2))]
MARKERS = ['o', 's']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moving_mean(values, window=5):
    """Trailing mean; the first four updates use the available preceding values."""
    return np.asarray([values[max(0, i-window+1):i+1].mean()
                       for i in range(len(values))])


def atomic_json(path, value):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_data():
    data = {'smoothing': 'trailing 5-update mean; available prefix at the boundary',
            'single_seed': True, 'uncertainty_bands': False,
            'verifier_metric_scope': 'unique non-padding turns; c includes only locally scored turns',
            'ablation_verifier_metrics': 'absent, not plotted or replaced by zero',
            'source_sha256': {}, 'models': {}}
    for label, run in RUNS.items():
        base = ARCHIVE_ROOT / run
        source = base / 'key_metrics.json'
        metrics = sorted(json.loads(source.read_text(encoding='utf-8'))['update_metrics'],
                         key=lambda item: item['step'])
        assert [item['step'] for item in metrics] == list(range(1, 51))
        values = {key: np.asarray([item[key] for item in metrics], dtype=float)
                  for key in KEYS}
        if label == 'C3-OPD':
            for key in VERIFIER_KEYS:
                values[key] = np.asarray([item[key] for item in metrics], dtype=float)
                assert ((values[key] >= 0) & (values[key] <= 1)).all()
        else:
            assert all(key not in item for item in metrics for key in VERIFIER_KEYS)
        assert all(np.isfinite(array).all() for array in values.values())
        assert ((values['episode/success_rate'] >= 0) &
                (values['episode/success_rate'] <= 1)).all()
        data['source_sha256'][str(source)] = digest(source)
        probe = []
        for step in [10, 20, 30, 40, 50]:
            path = base / 'probe_results' / f'global_step_{step}.json'
            record = json.loads(path.read_text(encoding='utf-8'))
            assert record['evaluated'] == len(record['questions']) == 350
            assert record['correct'] == sum(item['correct'] for item in record['questions'])
            probe.append({'step': step, 'correct': record['correct'],
                          'evaluated': 350, 'em_percent': record['correct'] / 350 * 100})
            data['source_sha256'][str(path)] = digest(path)
        data['models'][label] = {
            'run': run, 'updates': list(range(1, 51)),
            'raw_metrics': {key: value.tolist() for key, value in values.items()},
            'probe': probe,
        }
    assert [p['correct'] for p in data['models']['C3-OPD']['probe']] == [73, 91, 83, 93, 97]
    assert [p['correct'] for p in data['models']['GRPO+OPSD']['probe']] == [76, 58, 41, 47, 57]
    return data


def draw(data):
    try:
        font_manager.findfont('Times New Roman', fallback_to_default=False)
        serif = 'Times New Roman'
    except ValueError:
        serif = 'DejaVu Serif'
    plt.rcParams.update({
        'font.family': serif, 'font.size': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'axes.linewidth': .65, 'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'axes.unicode_minus': True,
    })
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.65), sharex=True)
    fig.subplots_adjust(left=.105, right=.986, bottom=.08, top=.905,
                        wspace=.32, hspace=.48)
    for ax, letter, title in zip(axes.flat, 'abcdef',
            ['Training success', 'Fixed-probe EM', 'Teacher–student gap', 'Gradient norm',
             r'Local correctness ($c$)', r'Context validity ($v$)']):
        ax.text(0, 1.10, letter, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=12, fontweight='bold')
        ax.text(.075, 1.10, title, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#444444')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E1E1E1', linewidth=.55)
        ax.set_xlim(0, 51)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.tick_params(colors='#333333', pad=3)

    for index, (label, model) in enumerate(data['models'].items()):
        x = np.asarray(model['updates'])
        series = [np.asarray(model['raw_metrics'][KEYS[0]]) * 100,
                  np.asarray(model['raw_metrics'][KEYS[1]]),
                  np.asarray(model['raw_metrics'][KEYS[2]])]
        for ax, values in zip([axes[0, 0], axes[1, 0], axes[1, 1]], series):
            ax.plot(x, values, color=COLORS[index], linestyle=STYLES[index],
                    linewidth=.65, alpha=.23, solid_capstyle='round')
            ax.plot(x, moving_mean(values), color=COLORS[index], linestyle=STYLES[index],
                    linewidth=1.65, solid_capstyle='round')
        probes = model['probe']
        axes[0, 1].plot([p['step'] for p in probes], [p['em_percent'] for p in probes],
                       color=COLORS[index], linestyle=STYLES[index], linewidth=1.65,
                       marker=MARKERS[index], markersize=4.3, markerfacecolor='white',
                       markeredgewidth=1.15)

    c3_model = data['models']['C3-OPD']
    for ax, key in zip(axes[2], VERIFIER_KEYS):
        values = np.asarray(c3_model['raw_metrics'][key])
        ax.plot(c3_model['updates'], values, color=COLORS[0], linewidth=.65, alpha=.23)
        ax.plot(c3_model['updates'], moving_mean(values), color=COLORS[0], linewidth=1.65)
        ax.set(ylabel='Score', ylim=(.4, .82), yticks=[.4, .5, .6, .7, .8])

    axes[0, 0].set(ylabel='Success (%)', ylim=(0, 50), yticks=[0, 10, 20, 30, 40, 50])
    axes[0, 1].set(ylabel='EM (%)', ylim=(9, 31), yticks=[10, 15, 20, 25, 30])
    axes[1, 0].set(ylabel=r'$\Delta\,\log p$ (nats/token)',
                   ylim=(-.80, -.48), yticks=[-.8, -.7, -.6, -.5])
    axes[1, 1].set(ylabel='Norm', ylim=(.55, 1.27), yticks=[.6, .8, 1., 1.2])
    for ax in axes[2]:
        ax.set_xlabel('Optimizer update', labelpad=5)
    handles = [Line2D([0], [0], color=COLORS[i], linestyle=STYLES[i], linewidth=1.8,
                      label=label) for i, label in enumerate(data['models'])]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .995),
               frameon=False, ncol=2, fontsize=10, handlelength=2.6,
               columnspacing=2.8, handletextpad=.7)
    for extension in ['png', 'pdf']:
        path = OUT / f'training_dynamics.{extension}'
        tmp = OUT / f'training_dynamics.tmp.{extension}'
        kwargs = {'dpi': 400} if extension == 'png' else {
            'metadata': {'Title': 'Training dynamics: C3-OPD and GRPO+OPSD',
                         'Subject': 'Two single-seed, 50-update Search-QA runs'}}
        fig.savefig(tmp, format=extension, **kwargs)
        os.replace(tmp, path)
    plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    values = load_data()
    atomic_json(OUT / 'training_dynamics_data.json', values)
    draw(values)
    print('Six panels exported: training_dynamics.png and training_dynamics.pdf')
