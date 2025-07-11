import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import deque, defaultdict
import networkx as nx
import io

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# --- Function to sample duration based on distribution type ---
def sample_duration(min_d, most_likely, max_d, dist):
    if dist == 'UNI':
        return np.random.uniform(min_d, max_d)
    elif dist == 'TRI':
        return np.random.triangular(min_d, most_likely, max_d)
    elif dist == 'PERT':
        mean = (min_d + 4 * most_likely + max_d) / 6
        std_dev = (max_d - min_d) / 6
        alpha = ((mean - min_d) * (2 * (most_likely - min_d) / (max_d - min_d) - 1)) / (std_dev ** 2)
        beta = alpha * (max_d - mean) / (mean - min_d)
        return np.random.beta(alpha, beta) * (max_d - min_d) + min_d if alpha > 0 and beta > 0 else most_likely
    elif dist == 'FRONT':
        return np.random.beta(4, 2) * (max_d - min_d) + min_d
    elif dist == 'BACK':
        return np.random.beta(2, 4) * (max_d - min_d) + min_d
    else:
        return most_likely

# --- Function to compute topological order ---
def compute_topological_order(predecessors, successors):
    in_degree = defaultdict(int)
    for to_act, preds in predecessors.items():
        in_degree[to_act] = len(preds)

    queue = deque([act for act in predecessors if in_degree[act] == 0])
    topo_order = []

    while queue:
        act = queue.popleft()
        topo_order.append(act)
        for succ, _, _ in successors[act]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)
    return topo_order

# --- Backward pass to compute Late Start and Late Finish ---
def run_backward_pass(G, early_start, early_finish, project_duration):
    late_finish = {}
    late_start = {}

    for node in reversed(list(nx.topological_sort(G))):
        successors = list(G.successors(node))
        if not successors:
            late_finish[node] = project_duration
        else:
            candidate_lfs = []
            for succ in successors:
                edge = G.get_edge_data(node, succ)
                rel_type = edge.get('type', 'FS')
                lag = edge.get('lag', 0)
                succ_ls = late_start.get(succ, project_duration)
                succ_lf = late_finish.get(succ, project_duration)
                succ_dur = G.nodes[succ].get('duration', 0)
                this_dur = G.nodes[node].get('duration', 0)

                if rel_type == "FS":
                    candidate_lfs.append(succ_ls - lag)
                elif rel_type == "SS":
                    candidate_ls = succ_ls - lag
                    candidate_lfs.append(candidate_ls + this_dur)
                elif rel_type == "FF":
                    candidate_lfs.append(succ_lf - lag)
                elif rel_type == "SF":
                    candidate_ls = succ_lf - lag
                    candidate_lfs.append(candidate_ls + this_dur)

            late_finish[node] = min(candidate_lfs)
        late_start[node] = late_finish[node] - G.nodes[node].get('duration', 0)

    return late_start, late_finish

# --- Function to run a single forward pass ---
def run_forward_pass(topo_order, predecessors, durations):
    es, ef = {}, {}
    for act in topo_order:
        preds = predecessors[act]
        start_times = []
        if not preds:
            es[act] = 0
        else:
            for pred, link_type, lag in preds:
                dur = durations.get(pred, 0)
                if link_type == 'FS':
                    start_times.append(ef[pred] + lag)
                elif link_type == 'SS':
                    start_times.append(es[pred] + lag)
                elif link_type == 'FF':
                    ef_val = ef[pred] + lag
                    es_val = ef_val - durations.get(act, 0)
                    start_times.append(es_val)
                elif link_type == 'SF':
                    ef_val = es[pred] + lag
                    es_val = ef_val - durations.get(act, 0)
                    start_times.append(es_val)
            es[act] = max(start_times)
        ef[act] = es[act] + durations.get(act, 0)
    return es, ef

# --- Main Simulation Function ---
def run_simulation(file, iterations=100):
    activities_df = pd.read_excel(file, sheet_name='Activities_Template')
    logic_df = pd.read_excel(file, sheet_name='Logic_Template')

    all_activities = set(activities_df['Activity ID'])
    successors = {act: [] for act in all_activities}
    predecessors = {act: [] for act in all_activities}

    for _, row in logic_df.iterrows():
        from_act, to_act, link_type, lag = row['From'], row['To'], row['Link Type'], row['Lag']
        successors[from_act].append((to_act, link_type, lag))
        predecessors[to_act].append((from_act, link_type, lag))

    topo_order = compute_topological_order(predecessors, successors)
    iterations_data = []
    project_durations = []
    duration_table = {act: [] for act in all_activities}
    float_data = []

    for i in range(iterations):
        sampled_durations = {}
        for _, row in activities_df.iterrows():
            act_id = row['Activity ID']
            d = sample_duration(row['Min Duration'], row['Most Likely Duration'], row['Max Duration'], row['Distribution'])
            sampled_durations[act_id] = round(d, 2)
            duration_table[act_id].append(sampled_durations[act_id])

        es, ef = run_forward_pass(topo_order, predecessors, sampled_durations)
        project_duration = max(ef.values())

        G = nx.DiGraph()
        for act in all_activities:
            G.add_node(act, duration=sampled_durations[act])
        for from_act, links in successors.items():
            for to_act, rel_type, lag in links:
                G.add_edge(from_act, to_act, type=rel_type, lag=lag)

        ls, lf = run_backward_pass(G, es, ef, project_duration)

        for act in topo_order:
            tf = round(ls[act] - es[act], 2)
            iterations_data.append({
                'Iteration': i + 1,
                'Activity': act,
                'Sampled Duration': sampled_durations[act],
                'Early Start': es[act],
                'Early Finish': ef[act],
                'Late Start': ls[act],
                'Late Finish': lf[act],
                'Total Float': tf
            })
            float_data.append({
                'Iteration': i + 1,
                'Activity': act,
                'Total Float': tf
            })

        project_durations.append({'Iteration': i + 1, 'Project Duration': project_duration})

    durations_series = pd.Series([d['Project Duration'] for d in project_durations])
    p10 = durations_series.quantile(0.10)
    p50 = durations_series.quantile(0.50)
    p90 = durations_series.quantile(0.90)

    bare_durations = activities_df.set_index('Activity ID')['Bare Duration'].to_dict()
    es_bare, ef_bare = run_forward_pass(topo_order, predecessors, bare_durations)
    bare_project_duration = max(ef_bare.values())
    likelihood_bare = round((durations_series <= bare_project_duration).mean() * 100, 2)

    summary_data = pd.DataFrame({
        'Metric': ['P10', 'P50', 'P90', 'Bare Duration', 'Likelihood ≤ Bare Duration'],
        'Value': [p10, p50, p90, bare_project_duration, f"{likelihood_bare}%"]
    })

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    counts, bins, _ = ax1.hist(durations_series, bins=30, color='skyblue', edgecolor='black', label='Histogram')
    sns.kdeplot(durations_series, ax=ax1, color='blue', label='KDE')
    ax1.set_xlabel('Duration')
    ax1.set_ylabel('Number of Iterations')

    ax2 = ax1.twinx()
    sns.ecdfplot(durations_series, ax=ax2, color='purple', label='Cumulative')
    ax2.set_ylabel('Cumulative Probability (%)')
    ax2.set_ylim(0, 1)
    ax2.set_yticks(np.linspace(0, 1, 11))
    ax2.set_yticklabels(['{:.0f}%'.format(x*100) for x in np.linspace(0, 1, 11)])

    for val, color, label in zip([p10, p50, p90], ['green', 'orange', 'red'], ['P10', 'P50', 'P90']):
        ax1.axvline(val, color=color, linestyle='dashed', linewidth=1, label=label)
    ax1.axvline(bare_project_duration, color='black', linestyle='dotted', linewidth=1, label='Bare Duration')

    fig1.legend(loc='upper right')
    fig1.tight_layout()
    
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    df_float = pd.DataFrame(float_data)
    critical_counts = df_float[df_float['Total Float'] == 0].groupby('Activity').size().sort_values(ascending=True)
    critical_counts.plot(kind='barh', color='firebrick', ax=ax3)
    ax3.set_title('Tornado Chart: Activity Criticality by Frequency')
    ax3.set_xlabel('Number of Iterations on Critical Path')
    ax3.set_ylabel('Activity')
    fig2.tight_layout()

    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    df_main = pd.DataFrame(iterations_data)
    df_main.to_excel(writer, sheet_name='Activity Forward Pass', index=False)

    df_project = pd.DataFrame(project_durations)
    df_project.to_excel(writer, sheet_name='Project Duration', index=False)

    df_duration_matrix = pd.DataFrame(duration_table)
    df_duration_matrix.insert(0, 'Iteration', list(range(1, iterations + 1)))
    df_duration_matrix.to_excel(writer, sheet_name='Durations by Iteration', index=False)

    df_float_matrix = df_float.pivot(index='Iteration', columns='Activity', values='Total Float').reset_index()
    df_float_matrix.to_excel(writer, sheet_name='Total Floats by Iteration', index=False)

    summary_data.to_excel(writer, sheet_name='Summary', index=False)

    # Embed charts
    workbook = writer.book
    worksheet = workbook.add_worksheet('Charts')
    chart_img_buf = io.BytesIO()
    fig1.savefig(chart_img_buf, format='png')
    chart_img_buf.seek(0)
    worksheet.insert_image('B2', 'distribution_chart.png', {'image_data': chart_img_buf})

    tornado_img_buf = io.BytesIO()
    fig2.savefig(tornado_img_buf, format='png')
    tornado_img_buf.seek(0)
    worksheet.insert_image('B30', 'tornado_chart.png', {'image_data': tornado_img_buf})

    writer.close()
    excel_data = output.getvalue()

    return {
        'summary': summary_data,
        'distribution_chart': fig1,
        'tornado_chart': fig2,
        'excel_bytes': excel_data
    }
