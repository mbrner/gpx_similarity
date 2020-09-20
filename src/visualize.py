from matplotlib import pyplot as plt
import numpy as np



def show_comparisons(save_path, orig, rec, n_rows=8, comps_per_row=4, figsize=(12,12)):
    n_rows = n_rows
    n_cols = comps_per_row * 2
    
    n_frames = n_rows*n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, (orig, rec) in enumerate(zip(orig, rec)):
        pos = (i % comps_per_row) * 2
        row = i // comps_per_row
        try:
            ax_orig, ax_rec = axes[row][pos], axes[row][pos+1]
        except IndexError:
            break
        if row == 0:
            ax_orig.set_title('original')
            ax_rec.set_title('Reconstructed')
        ax_orig.imshow(np.array(orig))
        ax_orig.axis('off')
        ax_rec.imshow(np.array(rec))
        ax_rec.axis('off')

    # if i < n_frames - 1:
    #     for j in range(i, n_frames):
    #         pos = (j % comps_per_row) * 2
    #         row = j // comps_per_row
    #         ax_orig, ax_rec = axes[row][pos], axes[row][pos+1]
    #         ax_rec.axis('off')
    #         ax_orig.axis('off')
    fig.savefig(save_path, dpi=300,bbox_inches='tight')
