from rfdiffusion.voxel.Voxel import VoxelGrid
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate volume and suggest length range from voxel file.")
    parser.add_argument("voxel_path", type=str, help="Path to the voxel file.")
    parser.add_argument("--cutoff", type=float, default=-0.866, help="Cutoff value for SDF. (default: -0.866)")
    return parser.parse_args()

def voxel_vol2len(voxel_path, cutoff):
    """
    Calculate the volume of a voxel grid and suggest a range of lengths based on that volume.
    
    Args:
        voxel_path (str): Path to the voxel file.
        cutoff (float): Cutoff value for SDF.
            
    Returns:
        tuple: A tuple containing the suggested minimum and maximum lengths, and the total volume.
    """
    # Initialize VoxelGrid
    count = VoxelGrid(voxel_path, resolution=2, cutoff=cutoff, shell=False, N_center=np.array([0, 0, 0]), target_to_pdb=False, centered=False).target_xyzs.shape[0]
    
    # Calculate total volume
    total_vol = count * 8 # 8 A^3 per voxel (w/ resolution 2 A), faster than 1 A^3 (w/ resolution 1 A) with little loss of accuracy
    
    # Suggest range based on total volume, Hyemi's formula. Drop decimal part
    min_n = int((total_vol + 5110) / 167)
    max_n = int((total_vol + 5060) / 152)
    
    return min_n, max_n, total_vol

if __name__ == "__main__":
    args = parse_args()
    voxel_path = args.voxel_path
    cutoff = args.cutoff
    
    # Calculate volume and suggest length range
    min_n, max_n, total_vol = voxel_vol2len(voxel_path, cutoff)
    
    print(f"Suggested RFdiffusion 'contigmap.contigs=[{min_n}-{max_n}]'")
    print(f"Total volume: {total_vol}")