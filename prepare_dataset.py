import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def prepare_dataset(source_root, dest_root, val_ratio=0.1):
    source_root = Path(source_root).resolve()
    dest_root = Path(dest_root).resolve()
    
    # Find all seasons
    seasons = [d for d in source_root.iterdir() if d.is_dir() and d.name.startswith('ROIs')]
    
    season_pairs = {
        'spring': [],
        'summer': [],
        'fall': [],
        'winter': []
    }
    
    print("Scanning for pairs...")
    for season_dir in seasons:
        # Determine season
        season_name = season_dir.name.lower()
        current_season = None
        if 'spring' in season_name: current_season = 'spring'
        elif 'summer' in season_name: current_season = 'summer'
        elif 'fall' in season_name: current_season = 'fall'
        elif 'winter' in season_name: current_season = 'winter'
        
        if current_season is None:
            print(f"Skipping unknown season dir: {season_dir.name}")
            continue

        # In each season dir, there are s1_X and s2_X folders
        subdirs = [d for d in season_dir.iterdir() if d.is_dir()]
        s1_dirs = {d.name.split('_')[1]: d for d in subdirs if d.name.startswith('s1_')}
        s2_dirs = {d.name.split('_')[1]: d for d in subdirs if d.name.startswith('s2_')}
        
        common_indices = set(s1_dirs.keys()) & set(s2_dirs.keys())
        
        for idx in common_indices:
            s1_dir = s1_dirs[idx]
            s2_dir = s2_dirs[idx]
            
            s1_files = sorted([f for f in s1_dir.iterdir() if f.suffix == '.png'])
            s2_files = sorted([f for f in s2_dir.iterdir() if f.suffix == '.png'])
            
            # Map by patch number pX
            s1_map = {f.name.split('_p')[-1]: f for f in s1_files}
            s2_map = {f.name.split('_p')[-1]: f for f in s2_files}
            
            common_patches = set(s1_map.keys()) & set(s2_map.keys())
            
            for p_idx in common_patches:
                season_pairs[current_season].append((s1_map[p_idx], s2_map[p_idx]))
                
    # Prepare datasets to process
    datasets_to_process = {k: v for k, v in season_pairs.items()}
    
    # Create 'all' dataset
    all_pairs = []
    for p_list in season_pairs.values():
        all_pairs.extend(p_list)
    datasets_to_process['all'] = all_pairs

    for dataset_name, pairs in datasets_to_process.items():
        if not pairs:
            print(f"Warning: No pairs found for {dataset_name}")
            continue
            
        print(f"Processing {dataset_name} dataset ({len(pairs)} pairs)...")
        
        ds_root = dest_root / dataset_name
        train_A = ds_root / 'train' / 'A'
        train_B = ds_root / 'train' / 'B'
        val_A = ds_root / 'val' / 'A'
        val_B = ds_root / 'val' / 'B'
        
        for p in [train_A, train_B, val_A, val_B]:
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
            
        # Shuffle and split
        current_pairs = pairs[:]
        random.shuffle(current_pairs)
        
        val_size = int(len(current_pairs) * val_ratio)
        if val_size == 0 and len(current_pairs) > 1:
            val_size = 1
            
        train_pairs = current_pairs[val_size:]
        val_pairs = current_pairs[:val_size]
        
        print(f"  Linking {len(train_pairs)} training pairs...")
        for s1, s2 in tqdm(train_pairs, desc=f"{dataset_name} Train"):
            new_name = s1.name.replace('_s1_', '_').replace('_s2_', '_')
            
            dst_A = train_A / new_name
            dst_B = train_B / new_name
            
            if not dst_A.exists():
                os.symlink(s1, dst_A)
            if not dst_B.exists():
                os.symlink(s2, dst_B)
            
        print(f"  Linking {len(val_pairs)} validation pairs...")
        for s1, s2 in tqdm(val_pairs, desc=f"{dataset_name} Val"):
            new_name = s1.name.replace('_s1_', '_').replace('_s2_', '_')
            
            dst_A = val_A / new_name
            dst_B = val_B / new_name
            
            if not dst_A.exists():
                os.symlink(s1, dst_A)
            if not dst_B.exists():
                os.symlink(s2, dst_B)
        
    print("Done.")

if __name__ == '__main__':
    # Adjust paths as needed
    SOURCE = '/opt/junie/sar/sen1-2_data'
    DEST = '/opt/junie/sar/Palette-Image-to-Image-Diffusion-Models/datasets/sen1-2'
    prepare_dataset(SOURCE, DEST)
