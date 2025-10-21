# data.py  -- paste this file into /kaggle/working/data/data.py
import os
import random
from torch.utils.data import Dataset

# Try to import load_config from either data.utils or utils (works in both contexts)
try:
    from data.utils import load_config
except Exception:
    try:
        from utils import load_config
    except Exception:
        # fallback: simple loader (expects path argument)
        import yaml
        def load_config(path=None):
            if path is None:
                path = "./config.yml"
            with open(path, "r") as f:
                return yaml.safe_load(f)

def _list_image_files(folder):
    """Return sorted list of image-like files in folder."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files = sorted(files)
    return files

def _full_paths(folder, filenames):
    return [os.path.join(folder, f) for f in filenames]

class Dataset_Generation(Dataset):
    """
    Dataset generator adapted for COD10K flat folder layout:
      - Images in a flat Image/ folder
      - Masks in a flat GT_Object/ or other GT_*/ folder

    Use:
      ds = Dataset_Generation(camo_data='cod10k_train', task='cod', image_size=512, mode='train')
    """

    def __init__(self, camo_data, search_data=None, task='cod', image_size=512, mode='train', count=1, config_path=None):
        """
        camo_data: string key in config 'dataset' (e.g. 'cod10k_train')
        search_data: string key for search/reference dataset (if required)
        task: 'cod', 'ref-cod', or 'rcod'
        image_size: (unused here) kept for API compatibility
        mode: 'train' or 'test'
        count: number of samples for some pairing logic (for ref/rcod)
        config_path: optional explicit path to config.yml
        """
        self.camo_data = camo_data
        self.search_data = search_data
        self.task = task
        self.image_size = image_size
        self.mode = mode
        self.count = count
        self.config_path = config_path

        self.image_pairs = self.data_type(self.task)
        self.length = len(self.image_pairs)

    def data_type(self, task):
        if task == 'cod':
            return self.cod_data()
        elif task == 'ref-cod':
            return self.refcod_data()
        elif task == 'rcod':
            return self.rcod_data()
        else:
            raise ValueError(f"Unknown task: {task}")

    def cod_data(self):
        """
        Pair images ↔ masks from flat Image/ and GT_Object/ folders.
        Returns list of dicts: {'cod_img': ..., 'cod_mask': ...}
        """
        cfg = load_config(self.config_path) if self.config_path else load_config()
        ds_cfg = cfg.get('dataset', {})
        if self.camo_data not in ds_cfg:
            raise KeyError(f"Dataset key '{self.camo_data}' not found in config['dataset']")

        img_folder = ds_cfg[self.camo_data]['Images']
        mask_folder = ds_cfg[self.camo_data]['GT']

        if not os.path.isdir(img_folder):
            raise FileNotFoundError(f"Image folder not found: {img_folder}")
        if not os.path.isdir(mask_folder):
            raise FileNotFoundError(f"Mask folder not found: {mask_folder}")

        img_files = _list_image_files(img_folder)
        mask_files = _list_image_files(mask_folder)

        # If filenames match except extension, try to pair by basename
        pairs = []
        # build maps by basename without extension
        img_map = {os.path.splitext(f)[0]: f for f in img_files}
        mask_map = {os.path.splitext(f)[0]: f for f in mask_files}
        common_keys = sorted(set(img_map.keys()) & set(mask_map.keys()))

        if common_keys:
            for k in common_keys:
                pairs.append({
                    'cod_img': os.path.join(img_folder, img_map[k]),
                    'cod_mask': os.path.join(mask_folder, mask_map[k])
                })
        else:
            # fallback: zip same-sorted lists (only if counts equal or user accepts warning)
            if len(img_files) != len(mask_files):
                print(f"⚠️ cod_data fallback pairing: {len(img_files)} images vs {len(mask_files)} masks; using min length")
            n = min(len(img_files), len(mask_files))
            for img_file, mask_file in zip(img_files[:n], mask_files[:n]):
                pairs.append({
                    'cod_img': os.path.join(img_folder, img_file),
                    'cod_mask': os.path.join(mask_folder, mask_file)
                })

        if len(pairs) == 0:
            raise RuntimeError("No image-mask pairs found (check config paths and file naming).")

        return pairs

    def refcod_data(self):
        """
        Reference-based COD pairing:
        - We'll try to pair camo images with reference images that share the same basename prefix.
        - This function expects both camo_data and search_data to be configured in config.yml.
        Returns list of dicts: {'cod_img':..., 'cod_mask':..., 'si_img':..., 'si_mask':..., 'si_label':..., 'cod_label':...}
        This is a simple conservative implementation that won't crash on flat folders.
        """
        cfg = load_config(self.config_path) if self.config_path else load_config()
        ds_cfg = cfg.get('dataset', {})

        if self.camo_data not in ds_cfg or not self.search_data or self.search_data not in ds_cfg:
            print("⚠️ refcod_data: required dataset keys missing. Returning empty list.")
            return []

        camo_img_folder = ds_cfg[self.camo_data]['Images']
        camo_mask_folder = ds_cfg[self.camo_data]['GT']
        si_img_folder = ds_cfg[self.search_data]['Images']
        si_mask_folder = ds_cfg[self.search_data]['GT']

        camo_imgs = _list_image_files(camo_img_folder)
        camo_masks = _list_image_files(camo_mask_folder)
        si_imgs = _list_image_files(si_img_folder)
        si_masks = _list_image_files(si_mask_folder)

        # create basename maps
        camo_map = {os.path.splitext(f)[0]: f for f in camo_imgs}
        camo_mask_map = {os.path.splitext(f)[0]: f for f in camo_masks}
        si_map = {os.path.splitext(f)[0]: f for f in si_imgs}
        si_mask_map = {os.path.splitext(f)[0]: f for f in si_masks}

        # Intersection keys
        common = sorted(set(camo_map.keys()) & set(si_map.keys()))
        if not common:
            print("⚠️ refcod_data: no common basenames between camo and search dataset; returning empty list.")
            return []

        final = []
        for k in common:
            cod_img = os.path.join(camo_img_folder, camo_map[k])
            cod_mask = os.path.join(camo_mask_folder, camo_mask_map.get(k, camo_map[k].rsplit('.',1)[0]+'.png'))
            si_img = os.path.join(si_img_folder, si_map[k])
            si_mask = os.path.join(si_mask_folder, si_mask_map.get(k, si_map[k].rsplit('.',1)[0]+'.png'))

            final.append({
                'cod_img': cod_img,
                'cod_mask': cod_mask,
                'si_img': si_img,
                'si_mask': si_mask,
                'si_label': k,
                'cod_label': k
            })

        return final

    def rcod_data(self):
        """
        A conservative rcod implementation:
        - Builds category groups by splitting filename tokens and grouping by the (heuristic) category token.
        - Then samples positive and negative pairs for each camo image.
        NOTE: This function is heuristic and intended to avoid crashes; tune per dataset naming.
        """
        cfg = load_config(self.config_path) if self.config_path else load_config()
        ds_cfg = cfg.get('dataset', {})

        if self.camo_data not in ds_cfg or not self.search_data or self.search_data not in ds_cfg:
            print("⚠️ rcod_data: required dataset keys missing. Returning empty list.")
            return []

        camo_img_folder = ds_cfg[self.camo_data]['Images']
        camo_mask_folder = ds_cfg[self.camo_data]['GT']
        si_img_folder = ds_cfg[self.search_data]['Images']
        si_mask_folder = ds_cfg[self.search_data]['GT']

        camo_imgs = _list_image_files(camo_img_folder)
        camo_masks = _list_image_files(camo_mask_folder)
        si_imgs = _list_image_files(si_img_folder)
        si_masks = _list_image_files(si_mask_folder)

        # Build simple category key by taking the token after 'CAM-' if present, otherwise use first 3 tokens
        def category_key(filename):
            name = os.path.splitext(filename)[0]
            parts = name.split('-')
            for i, p in enumerate(parts):
                if p.startswith('CAM'):
                    if i+1 < len(parts):
                        return parts[i+1]
            # fallback
            return '_'.join(parts[:3])

        # group si images by category key
        si_groups = {}
        for f in si_imgs:
            k = category_key(f)
            si_groups.setdefault(k, []).append(f)

        # create camo groups similarly
        camo_groups = {}
        for f in camo_imgs:
            k = category_key(f)
            camo_groups.setdefault(k, []).append(f)

        final = []
        for cat, camo_list in camo_groups.items():
            pos_pool = si_groups.get(cat, [])
            # negative categories = other keys
            neg_keys = [kk for kk in si_groups.keys() if kk != cat]
            for camo_file in camo_list:
                cod_img = os.path.join(camo_img_folder, camo_file)
                # choose a positive sample if available, otherwise skip pos
                if pos_pool:
                    si_file = random.choice(pos_pool)
                    si_img_path = os.path.join(si_img_folder, si_file)
                    cod_mask_file = os.path.splitext(camo_file)[0] + '.png'
                    cod_mask_path = os.path.join(camo_mask_folder, cod_mask_file)
                    final.append({
                        'cod_img': cod_img,
                        'si_img': si_img_path,
                        'cod_mask': cod_mask_path,
                        'si_mask': os.path.join(si_mask_folder, os.path.splitext(si_file)[0] + '.png'),
                        'si_label': cat,
                        'cod_label': cat
                    })
                # negative: sample from other categories
                if neg_keys:
                    nk = random.choice(neg_keys)
                    neg_file = random.choice(si_groups[nk])
                    final.append({
                        'cod_img': cod_img,
                        'si_img': os.path.join(si_img_folder, neg_file),
                        'cod_mask': os.path.join(camo_mask_folder, os.path.splitext(camo_file)[0] + '.png'),
                        'si_mask': os.path.join(si_mask_folder, os.path.splitext(neg_file)[0] + '.png'),
                        'si_label': nk,
                        'cod_label': cat
                    })
        return final

    def __getitem__(self, index):
        # support tensor index
        if hasattr(index, "tolist"):
            index = int(index.tolist())
        return self.image_pairs[index]

    def __len__(self):
        return self.length
