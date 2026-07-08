from gt_kmeans import GTKmeans
import os
import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon
from sklearn.cluster import KMeans
import h5py
import matplotlib.pyplot as plt
import re
from scipy.stats import spearmanr

epsilons = [0.05, 0.1]
n_reps = 200
baselines = [10, 20, 50, 100]
inertia_decimal = 2

def load_dataset(path):
    try:
        mat = loadmat(path)
        data = mat['Data'].astype(float)
        k = len(np.unique(mat['y'].ravel()))
    except NotImplementedError:
        # v7.3 HDF5 format
        with h5py.File(path, 'r') as f:
            data = np.array(f['Data']).T.astype(float)  # transpose: h5py loads transposed
            y = np.array(f['y']).ravel()
            k = len(np.unique(y))
    return data, k



def run_all(data_path:str):
    cwd = os.getcwd()
    os.chdir(data_path)
    all_files = os.listdir()
    for file in all_files:
        if file.endswith('.mat'):
            data, k = load_dataset(file)
            filename = file.split('.')[0]
            print('Processing: ' + filename)

            gt_results={}
            gt_runs={}
            gt_stop_crit={}
            gt_distinct_partitions={}
            gt_lb_prob_km_converges_to_best_found={}
            km_results={}

            #GTKmeans
            for eps in epsilons:
                inertias = []
                runs = []
                stop_crit = np.array([0,0,0])
                gt_distinct_partitions[eps] = np.zeros(n_reps)
                gt_lb_prob_km_converges_to_best_found[eps] = np.zeros(n_reps) #this is a lower bound
                for rep_i in range(n_reps):
                    model = GTKmeans(k=k, epsilon=eps)
                    gt_km = model.fit(data)
                    inertias.append(gt_km.inertia_)
                    runs.append(model.run_i)
                    stop_crit += model.stop_criteria
                    gt_distinct_partitions[eps][rep_i]=len(model.partition_counts)
                    gt_lb_prob_km_converges_to_best_found[eps][rep_i] = model.p_lo
                gt_results[eps] = np.array(inertias)
                gt_runs[eps] = np.array(runs)
                gt_stop_crit[eps] = stop_crit/n_reps
                print(filename + '&' + str(eps) + '&' +
                      str(round(gt_results[eps].mean(), inertia_decimal)) + '&' +
                      str(round(gt_results[eps].std(), 2)) + '&' +
                      str(round(gt_runs[eps].mean(), inertia_decimal)) + '&' +
                      str(round(gt_runs[eps].std(), 2)))
                print('Criteria used')
                print(gt_stop_crit[eps])
                
                print('Distinct partitions found')
                print('avg: ' + str(round(gt_distinct_partitions[eps].mean(),2)) + ' std: '+str(round(gt_distinct_partitions[eps].std(),2)))
                
                print('Clopper-Pearson lower confidence bound on the true prob km converges to best found in 1 run')
                print('avg: ' + str(round(gt_lb_prob_km_converges_to_best_found[eps].mean(),2)) + ' std: '+str(round(gt_lb_prob_km_converges_to_best_found[eps].std(),2)))

            #kmeans
            for runs in baselines:
                inertias = []
                for rep_i in range(n_reps):
                    km=KMeans(n_clusters = k, n_init = runs).fit(data)
                    inertias.append(km.inertia_)
                km_results[runs] = np.array(inertias)
                print(filename + '&' +
                      str(round(km_results[runs].mean(), inertia_decimal)) + '&' +
                      str(round(km_results[runs].std(), 2)) + '&' +
                      str(runs))

            #Wilcoxon signed-ran tests: GT vs each baseline
            print('\n--- Wilcoxon tests for ' + filename + ' ---')
            wilcoxon_results = {}
            for eps in epsilons:
                wilcoxon_results[eps] = {}
                for b in baselines:
                    x = np.round(gt_results[eps], inertia_decimal)
                    y = np.round(km_results[b], inertia_decimal)
                    # check if there's any difference to test
                    if np.all(x == y):
                        print(f'GT(eps={eps}) vs fixed({b}): identical results, no test needed')
                        wilcoxon_results[eps][b] = {'stat': np.nan, 'p': np.nan, 'identical': True}
                        continue
                    stat, p = wilcoxon(x, y, alternative='two-sided')
                    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                    print(f'GT(eps={eps}) vs fixed({b}): W={stat:.1f}, p={p:.4f} {sig}')
                    wilcoxon_results[eps][b] = {'stat': stat, 'p': p, 'identical': False}
            
            np.savez(f'results_{filename}.npz',
                     gt_results=gt_results,
                     gt_runs=gt_runs,
                     gt_stop_crit=gt_stop_crit,
                     gt_distinct_partitions=gt_distinct_partitions,
                     gt_p_lo=gt_lb_prob_km_converges_to_best_found,
                     km_results=km_results,
                     wilcoxon_results=wilcoxon_results)
            #print(os.path.abspath(f'results_{filename}.npz'))
    os.chdir(cwd)



def plot_median_gap(results_path: str, epsilons=(0.05, 0.1), baselines=(10, 20, 50, 100)):
    """
    For each dataset, computes the % gap of each method's objective relative
    to the best observed value (minimum across GTRC and all fixed
    baselines), then plots the median restart count vs median % gap across
    all datasets, for each method, with GTRC shown separately per epsilon.
    """
    cwd = os.getcwd()
    os.chdir(results_path)
    all_files = os.listdir()

    baselines = sorted(baselines)
    gaps = {eps: {'GT': [], **{b: [] for b in baselines}} for eps in epsilons}
    restarts = {eps: {'GT': [], **{b: [] for b in baselines}} for eps in epsilons}

    for file in all_files:
        if file.endswith('.npz'):
            f = np.load(file, allow_pickle=True)
            gt_results = f['gt_results'].item()
            gt_runs = f['gt_runs'].item()
            km_results = f['km_results'].item()

            for eps in epsilons:
                gt_mean = gt_results[eps].mean()
                r_mean = gt_runs[eps].mean()
                km_means = {b: km_results[b].mean() for b in baselines}
                best = min([gt_mean] + list(km_means.values()))

                gaps[eps]['GT'].append(100 * (gt_mean - best) / best if best != 0 else 0.0)
                restarts[eps]['GT'].append(r_mean)
                for b in baselines:
                    gaps[eps][b].append(100 * (km_means[b] - best) / best if best != 0 else 0.0)
                    restarts[eps][b].append(b)

    os.chdir(cwd)

    points = {
        eps: {m: (np.median(restarts[eps][m]), np.median(gaps[eps][m]))
              for m in ['GT'] + baselines}
        for eps in epsilons
    }

    fig, ax = plt.subplots(figsize=(6.5, 5))

    # fixed k-means++ frontier is identical across epsilons, plot once using the first epsilon
    ref_eps = epsilons[0]
    fixed_x = [points[ref_eps][b][0] for b in baselines]
    fixed_y = [points[ref_eps][b][1] for b in baselines]
    ax.plot(fixed_x, fixed_y, color='0.4', linestyle='--', marker='o', markersize=7,
             markerfacecolor='0.7', markeredgecolor='black', label='fixed k-means++', zorder=2)
    for b, x, y in zip(baselines, fixed_x, fixed_y):
        ax.annotate(str(b), (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

    markers_gray = ['0', '0.55', '0.8']
    for i, eps in enumerate(epsilons):
        gx, gy = points[eps]['GT']
        shade = markers_gray[min(i, len(markers_gray) - 1)]
        ax.scatter([gx], [gy], marker='*', s=280, color=shade, edgecolor='black',
                    linewidth=0.8, zorder=3, label=rf'GTRC ($\varepsilon={eps}$)')
        ax.annotate(f'{gx:.0f}', (gx, gy), textcoords="offset points", xytext=(10, -4), fontsize=9)

    ax.set_xlabel('median restarts used')
    ax.set_ylabel('median % gap to best observed objective')
    ax.legend(frameon=False, loc='upper right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'median_gap.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return points
        
def plot_restart_spread(results_path: str, epsilon: float = 0.05):
    cwd = os.getcwd()
    os.chdir(results_path)
    all_files = os.listdir()

    rows = []
    for file in all_files:
        if file.endswith('.npz'):
            name = file.replace('results_', '').replace('.npz', '').replace('_std', '').replace('_Std', '')
            name = re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', ' ', name)
            f = np.load(file, allow_pickle=True)
            gt_runs = f['gt_runs'].item()
            rows.append((name, gt_runs[epsilon].mean()))

    os.chdir(cwd)

    rows.sort(key=lambda r: r[1])
    median_r = np.median([r[1] for r in rows])

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ['0.8' if v < median_r else '0.4' for v in vals]

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.barh(range(len(names)), vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(median_r, color='black', linestyle='--', linewidth=1)
    ax.text(median_r + 0.5, len(names) - 0.5, f'median = {median_r:.0f}', fontsize=9, va='top')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f'restarts used by GTRC ($\\varepsilon={epsilon}$)')
    ax.set_xlim(0, max(vals) + 5)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'restart_spread_eps{epsilon}.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return rows, median_r


def plot_p_lo_vs_runs(results_path: str, epsilon: float = 0.05):
    cwd = os.getcwd()
    os.chdir(results_path)
    all_files = os.listdir()

    rows = []
    for file in all_files:
        if file.endswith('.npz'):
            f = np.load(file, allow_pickle=True)
            gt_runs = f['gt_runs'].item()
            gt_p_lo = f['gt_p_lo'].item()
            rows.append((gt_runs[epsilon].mean(), gt_p_lo[epsilon].mean()))

    os.chdir(cwd)

    r_vals = np.array([r[0] for r in rows])
    p_vals = np.array([r[1] for r in rows])
    rho, pval = spearmanr(r_vals, p_vals)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(r_vals, p_vals, color='0.3', edgecolor='black', s=45, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel(f'restarts used by GTRC ($\\varepsilon={epsilon}$)')
    ax.set_ylabel('$p_{lo}$ (lower confidence bound on $p^*$)')
    ax.text(0.97, 0.95, rf'Spearman $\rho={rho:.2f}$', transform=ax.transAxes,
            ha='right', va='top', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f'p_lo_vs_runs_eps{epsilon}.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return rows, rho, pval

def plot_bound_rates(results_path: str, epsilons=(0.05, 0.1)):
    cwd = os.getcwd()
    os.chdir(results_path)
    all_files = os.listdir()

    avg_rates = {eps: {'GT': [], 'worst': [], 'emp': []} for eps in epsilons}
    for file in all_files:
        if file.endswith('.npz'):
            f = np.load(file, allow_pickle=True)
            stop_crit = f['gt_stop_crit'].item()
            for eps in epsilons:
                gt, worst, emp = stop_crit[eps]
                avg_rates[eps]['GT'].append(gt)
                avg_rates[eps]['worst'].append(worst)
                avg_rates[eps]['emp'].append(emp)

    os.chdir(cwd)

    categories = ['GT', 'worst', 'emp']
    labels_display = ['Good-Turing', 'unconditional bound', 'confidence bound']

    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(categories))
    width = 0.35 if len(epsilons) == 2 else 0.6 / len(epsilons)

    hatches = ['///', 'xxx', '...', '\\\\\\']
    shades = ['0.75', '0.4', '0.6', '0.3']

    for i, eps in enumerate(epsilons):
        offset = (i - (len(epsilons) - 1) / 2) * width
        vals = [np.mean(avg_rates[eps][c]) for c in categories]
        ax.bar(x + offset, vals, width, label=rf'$\varepsilon={eps}$',
               color=shades[i % len(shades)], edgecolor='black', hatch=hatches[i % len(hatches)])
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_display, fontsize=9)
    ax.set_ylabel('average trigger rate across data sets')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, 'bound_rates.png'), dpi=150, bbox_inches='tight')
    plt.show()

    return avg_rates


