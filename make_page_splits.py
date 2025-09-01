# make_page_splits.py
import pandas as pd, re, random, os
random.seed(12345)

df = pd.read_csv("gnhk_dataset/train_processed.csv")  # lines data
def prefix(fname):
    # e.g. eng_AS_001_0.jpg -> eng_AS_001
    parts = fname.split('_')
    if len(parts) >= 3:
        return '_'.join(parts[:3])
    return fname

df['page'] = df['image_filename'].map(prefix)
pages = df['page'].unique().tolist()
random.shuffle(pages)
n = len(pages)
n_train = int(0.8*n); n_val = int(0.1*n)
train_pages = set(pages[:n_train]); val_pages=set(pages[n_train:n_train+n_val]); test_pages=set(pages[n_train+n_val:])

os.makedirs("gnhk_dataset/splits", exist_ok=True)
for name, s in [('train', train_pages), ('val', val_pages), ('test', test_pages)]:
    df_sub = df[df['page'].isin(s)][['image_filename','text']]
    df_sub.to_csv(f"gnhk_dataset/splits/{name}_lines.csv", index=False)
    with open(f"gnhk_dataset/splits/{name}_pages.txt", 'w') as f:
        f.write('\n'.join(sorted(s)))
print("wrote splits to gnhk_dataset/splits/")
