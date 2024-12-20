
from pathlib import Path
import itertools

import tqdm
import fire
import numpy as np
import pandas
import torch.nn

from Network import U_Net as FCNNet
import ufold.postprocess, ufold.data_generator


PERMUTATION = list(itertools.product(np.arange(4), np.arange(4)))


def preprocess(path):
    arrays = pandas.read_csv(path)
    # remove the last repeat from each detected array as it
    # often looks different and doesn't have a spacer
    no_last_repeat = (arrays
                      .groupby(['tool', 'contig_id', 'array_id'])
                      .apply(lambda group: group.iloc[:-1],
                             include_groups=False)
                      .reset_index(drop=True)
                      )
    # unique sequences
    seqs = (no_last_repeat['repeat']
            .str.upper()
            .str.replace('T', 'U')
            )
    seqs = (seqs[seqs.str.match(r'^[ACGU]+$')]
            # .str.replace('-', '')
            # .str.replace('N', '')
            .drop_duplicates()
            .tolist()
            )
    return seqs, path.parent.name


def seqs2onehot(seqs, max_length=600):
    bases = {'A': 0, 'U': 1, 'C': 2, 'G': 3}
    one_hots = np.zeros((len(seqs), max_length, len(bases)))

    for i, seq in enumerate(seqs):
        for j, base in enumerate(seq[:max_length]):
            if base in bases:
                one_hots[i, j, bases[base]] = 1
            else:
                one_hots[i, j] = -1
    return one_hots


def get_seq(seq, one_hot, max_size=80):
    # Not sure what this does
    size = len(seq)
    exp_size = len(seq)
    if exp_size > max_size:  # TODO: What does this do?
        exp_size = (((exp_size - 1) // 16) + 1) * 16
    else:
        exp_size = max_size

    data_fcn = np.zeros((16, exp_size, exp_size))
    if exp_size >= 500:  # TODO: why is this necessary?
        one_hot = one_hot[:size]

    for n, cord in enumerate(PERMUTATION):
        i, j = cord
        data_fcn[n, :size, :size] = np.matmul(
            one_hot[:size, i].reshape(-1, 1),
            one_hot[:size, j].reshape(1, -1)
        )

    data_fcn_1 = np.zeros((1, exp_size, exp_size))
    data_fcn_1[0, :size, :size] = ufold.data_generator.creatmat(one_hot[:size])
    data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)

    return data_fcn_2, one_hot[:exp_size]


def model_eval(contact_net, seq_embeddings, seq_ori, size):
    contact_net.train()

    seq_embedding_batch = torch.Tensor(seq_embeddings).unsqueeze(0)
    seq_ori = torch.Tensor(seq_ori).unsqueeze(0)

    with torch.no_grad():
        pred_contacts, latent = contact_net(seq_embedding_batch)

    # only post-processing without learning
    u_no_train = ufold.postprocess.postprocess_new(
        pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
    predict_matrix = (u_no_train > 0.5).float()

    seq_tmp = torch.mul(
        predict_matrix.argmax(axis=1),
        predict_matrix.sum(axis=1).clamp_max(1)
    ).numpy().astype(int)

    seq_tmp[predict_matrix.sum(axis=1) == 0] = -1
    dot_list = seq2dot((seq_tmp + 1).squeeze())

    return dot_list[:size], latent.mean(axis=[0, 2, 3]).numpy()


def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file


def collect_data(root):
    root = Path(root)
    paths = root.glob('**/pred_structures.csv')
    dfs_list = []
    latents = []
    for path in tqdm.tqdm(paths):
        latent_file = path.parent / 'latents.npy'
        if latent_file.exists():
            try:
                df = pandas.read_csv(path)
                latent = np.load(latent_file)
            except:
                continue

            dfs_list.append(df)
            latents.append(latent)

    df = pandas.concat(dfs_list)
    duplicates = df['sequence'].duplicated()

    df[~duplicates].to_csv(root / f'pred_structures_{root.name}.csv', index=False)
    np.save(root / f'latents_{root.name}.npy', np.concatenate(latents)[~duplicates])


def compare_seqs(path):
    ufold.utils.seed_torch()

    contact_net = FCNNet(img_ch=17)
    contact_net.load_state_dict(torch.load('models/ufold_train_alldata.pt', map_location='cpu'))

    with open(path) as f:
        lines = f.readlines()

    seqs = {}
    name = ''
    for s in lines:
        line = s.strip()
        if s.startswith('>'):
            name = s[1:].strip()
        else:
            seqs[name] = line.upper().replace('T', 'U')

    one_hots = seqs2onehot(list(seqs.values()))

    df_list = []
    latents = []
    for (name, seq), one_hot in zip(seqs.items(), one_hots):
        seq_embeddings, seq_ori = get_seq(seq, one_hot)
        pred_struct, latent = model_eval(contact_net, seq_embeddings, seq_ori, len(seq))
        df_list.append({
            'name': name,
            'sequence': seq,
            'structure': pred_struct
        })
        latents.append(latent)

    pandas.DataFrame(df_list).to_csv('test_pred_structures.csv', index=False)
    np.save('test_latents.npy', latents)


def main(root, recompute=False):
    ufold.utils.seed_torch()

    contact_net = FCNNet(img_ch=17)
    contact_net.load_state_dict(torch.load('models/ufold_train_alldata.pt', map_location='cpu'))

    paths = Path(root).glob('**/crispr_arrays.csv')
    for path in tqdm.tqdm(paths):
        if not recompute and (path.parent / 'latents.npy').exists():
            continue

        seqs, assembly_id = preprocess(path)
        one_hots = seqs2onehot(seqs)

        df_list = []
        latents = []
        for seq, one_hot in zip(seqs, one_hots):
            seq_embeddings, seq_ori = get_seq(seq, one_hot)
            pred_struct, latent = model_eval(contact_net, seq_embeddings, seq_ori, len(seq))
            df_list.append({
                'assembly_id': assembly_id,
                'sequence': seq,
                'structure': pred_struct
            })
            latents.append(latent)

        pandas.DataFrame(df_list).to_csv(path.parent / 'pred_structures.csv', index=False)
        np.save(path.parent / 'latents.npy', latents)


if __name__ == '__main__':
    fire.Fire()
