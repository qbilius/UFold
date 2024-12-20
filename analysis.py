# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold, sklearn.metrics
import networkx as nx
import pandas
import umap

import ViennaRNA


# %% [markdown]
# RNA secondary structure visualization

def draw(
    structure: str,
    sequence: str | None = None,
    colors: list | None = None,
    node_size: int = 10,
    ax: plt.Axes | None = None,
) -> None:
    """
    Draw a structure
    """

    if sequence is None:
        sequence = [''] * len(structure)
    else:
        sequence = sequence.upper().replace('T', 'U')

    if colors is None:
        colors = ['skyblue'] * len(structure)

    assert len(structure) == len(sequence)

    hide_axis = False
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)
        hide_axis = True

    # Place nucleotides in a graph with pairings obtained from the structure
    G = nx.Graph()
    stack = []
    for i, (char, nt, color) in enumerate(zip(structure, sequence, colors, strict=True)):
        G.add_node(i, label=nt, color=color)
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            G.add_edge(j, i)

        if i > 0:
            G.add_edge(i - 1, i)

    # This layout seems to work well for gRNAs
    pos = nx.kamada_kawai_layout(G)
    # but we need to rotate the structures
    flipped_pos = {node: (1 - x, y) for node, (x, y) in pos.items()}
    rotated_pos = {node: (-y, x) for node, (x, y) in flipped_pos.items()}

    compl = str.maketrans('AUCG ', 'UAGC ')
    norm_seq = [s.translate(compl) for s in sequence]
    edge_colors = []
    for n1, n2 in G.edges():
        if abs(n1 - n2) == 1:  # subsequent bases
            edge_colors.append('lightgray')
        else:
            edge_colors.append('red' if norm_seq[n1] != norm_seq[n2].translate(compl) else 'lightgray')

    nx.draw(
        G,
        rotated_pos,
        with_labels=False,
        node_size=node_size,
        node_color=[G.nodes[n]['color'] for n in G.nodes],
        edge_color=edge_colors,
        hide_ticks=False,
        ax=ax)
    nx.draw_networkx_labels(
        G,
        rotated_pos,
        labels={n: G.nodes[n]['label'] for n in G.nodes},
        font_size=node_size / 15,
        hide_ticks=False,
        ax=ax
    )

    if hide_axis:
        ax.set_axis_off()

    return ax


# %% [markdown]
# # Get data

df = pandas.read_csv('pred_structures_all.csv')


# %% [markdown]
# # Get hairpin features

def get_features(struct):
    groups = []
    count = 0
    pairs = 0
    for node in struct:
        if node == '(':
            count += 1
            pairs += 1
        elif node == ')':
            count -= 1
            if count == 0:
                groups.append(pairs)
                pairs = 0

    if len(groups) == 0:
        return 0, 0, 0
    else:
        return len(groups), min(groups), max(groups)


df[['n_hairpins', 'min_size', 'max_size']] = df['structure'].apply(lambda x: pandas.Series(get_features(x)))


# %% [markdown]
# # Visualize hairpin features

print(f'Total sequences: {len(df)}')

for n_hairpins in np.unique(df['n_hairpins']):

    sel = df[df['n_hairpins'] == n_hairpins]
    if n_hairpins == 0:
        print(f'No hairpins: {len(sel)}')
        continue
    elif n_hairpins == 1:
        plt.subplot(2, 2, 1)
        agg = sel.groupby('min_size').size()
        agg.plot.bar(title=f'n_hairpins={n_hairpins}')
        plt.xlabel('Hairpin size')
    else:
        pv = sel.pivot_table(index='min_size', columns='max_size', aggfunc='size')
        plt.subplot(2, 2, n_hairpins)
        plt.title(f'n_hairpins={n_hairpins}')
        sns.heatmap(data=pv)

plt.tight_layout()


# %% [markdown]
# # Visualize one particular sequence

entry = df[(df['n_hairpins'] == 4) & (df['min_size'] >= 1)].iloc[0]

# struct, _ = ViennaRNA.fold(entry['sequence'])
draw(entry['structure'], sequence=entry['sequence'], node_size=80)


# %% [markdown]
# # Use UFold feature to visualize distances
# ## Get latent feature

X = np.load('latents_archaea.npy')
y = df['name'].values


# %% [markdown]
# ## Compute pairwise distances

distances = 1 - np.abs(sklearn.metrics.pairwise_distances(X, metric='cosine'))
fig, ax = plt.subplots()
cax = ax.matshow(distances)

fig.colorbar(cax)

ax.set_xticks(range(len(y)))
ax.set_yticks(range(len(y)))
ax.set_xticklabels(y, rotation=45, ha='left')
ax.set_yticklabels(y)


# %% [markdown]
# ## Visualize results with UMAP

umap_model = umap.UMAP(
    # n_neighbors=n_neighbors,
    # min_dist=min_dist,
    n_components=2,
    random_state=0
)
X = df[['n_hairpins', 'min_size', 'max_size']]
Xtr = umap_model.fit_transform(X)

scatter = plt.scatter(Xtr[:, 0], Xtr[:, 1], s=1, color='black', alpha=.5)

# for i, label in enumerate(y):
#     plt.text(Xtr[i, 0], Xtr[i, 1], label, fontsize=9, ha='left', va='bottom')

plt.axis('equal')
