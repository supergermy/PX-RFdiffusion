import torch
import numpy as np 
import json

class VoxelGrid:
    def __init__(self, voxel_path, resolution=1, cutoff=0, shell=True, N_center=np.array([0,0,0]), target_to_pdb=True, centered=False):
        """Initialize VoxelGrid from JSON file using memory-efficient techniques"""
        with open(voxel_path, "r") as f:
            data = json.load(f)
        
        self.grid_size = data["gridSize"]
        self.voxel_size = data["voxelSize"]
        voxel_origin = np.array([
            data["origin"]["x"],
            data["origin"]["y"], 
            data["origin"]["z"],
        ], dtype=np.float32)
        self.centered_voxel_origin = voxel_origin - N_center.squeeze() # now, rfdiff and voxel file share the world coord
        
        self.resolution = resolution
        self.cutoff = cutoff
        self.sdf = np.array(data["sdfValues"], dtype=np.float32)
        
        self.target_xyzs = self.get_target_xyz_vectorized(shell=shell)
        prefix = 'shell' if shell else 'core'
        
        self.centered = centered
        
        if centered:
            # center the targets so their center-of-mass is at the origin
            print('Centering the targets to the voxel com')
            self.com = self.target_xyzs.mean(dim=0, keepdim=True)
            self.target_xyzs = self.target_xyzs - self.com
            prefix += '_centered'
        
        out_path = voxel_path.replace('.voxel',f'.{prefix}_c{cutoff}.pdb')
        if target_to_pdb:
            self._to_pdb(self.target_xyzs, out_path)
    
    def get_target_xyz_vectorized(self, shell=True):
        """Vectorized version of get_target_xyz using NumPy."""

        # 1. Create grid indices using the specified resolution
        indices = np.arange(0, self.grid_size, self.resolution)
        
        # Use np.meshgrid to generate index grids
        X_idx, Y_idx, Z_idx = np.meshgrid(indices, indices, indices, indexing='ij')

        # 2. Calculate the flattened 1D indices for SDF lookup
        # Ensure indices stay within bounds of self.sdf if grid_size is not perfectly divisible
        X_idx_flat = X_idx.flatten()
        Y_idx_flat = Y_idx.flatten()
        Z_idx_flat = Z_idx.flatten()
        flat_indices = X_idx_flat + Y_idx_flat * self.grid_size + Z_idx_flat * self.grid_size * self.grid_size

        # Handle potential out-of-bounds due to resolution steps near the edge
        valid_flat_indices_mask = flat_indices < len(self.sdf)
        flat_indices = flat_indices[valid_flat_indices_mask]
        X_idx_flat = X_idx_flat[valid_flat_indices_mask]
        Y_idx_flat = Y_idx_flat[valid_flat_indices_mask]
        Z_idx_flat = Z_idx_flat[valid_flat_indices_mask]


        # 3. Look up SDF values using the calculated indices
        sdf_values = self.sdf[flat_indices]

        # 4. Apply filtering conditions using boolean masking
        upper_bound = self.cutoff + 0.866
        mask = (sdf_values < upper_bound)
        if shell:
            mask &= (sdf_values >= self.cutoff)

        # 5. Select indices that satisfy the conditions
        filtered_X_idx = X_idx_flat[mask]
        filtered_Y_idx = Y_idx_flat[mask]
        filtered_Z_idx = Z_idx_flat[mask]

        # 6. Calculate world coordinates for the filtered indices
        xxx = self.centered_voxel_origin[0] + filtered_X_idx * self.voxel_size
        yyy = self.centered_voxel_origin[1] + filtered_Y_idx * self.voxel_size
        zzz = self.centered_voxel_origin[2] + filtered_Z_idx * self.voxel_size

        # 7. Stack coordinates and convert to PyTorch tensor
        xyzs_np = np.stack([xxx, yyy, zzz], axis=-1).astype(np.float32)
        result_tensor = torch.from_numpy(xyzs_np) # More efficient than torch.tensor()

        return result_tensor # [Ns, 3]
    
    def is_outside(self, point):
        """Check if a given point is inside the shell."""
        point = point.detach().numpy()
        if self.centered:
            # If the grid was centered, shift back by the saved center-of-mass
            point = point + self.com.squeeze().cpu().numpy()
        # Compute voxel indices
        voxel_index = ((np.array(point) - self.centered_voxel_origin) / self.voxel_size).astype(int)
        
        x, y, z = voxel_index
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size and 0 <= z < self.grid_size:
            idx = x + y * self.grid_size + z * self.grid_size * self.grid_size
            sdf_value = self.sdf[idx]
            
            return sdf_value > self.cutoff+0.866
        else:
            return True
    
    def is_inside(self, point):
        return not self.is_outside(point)
    
    def _to_pdb(self, xyz, out_path):
        with open(out_path, 'w') as f:
            for idx, (xxx, yyy, zzz) in enumerate(xyz):
                # Compute voxel indices from coordinate
                voxel_index = ((np.array([xxx, yyy, zzz]) - self.centered_voxel_origin) / self.voxel_size).astype(int)
                x_idx, y_idx, z_idx = voxel_index
                # Compute the flattened index for the voxel grid
                flat_idx = x_idx + y_idx * self.grid_size + z_idx * (self.grid_size ** 2)
                sdf_value = self.sdf[flat_idx]
                f.write(
                    "%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"
                    % (
                        "ATOM",
                        idx,
                        'Fe',
                        'X',
                        'A',
                        idx,
                        xxx,
                        yyy,
                        zzz,
                        1.0,
                        sdf_value,
                    )
                )
        print(out_path)
                # f.write(f'ATOM      1  CA  ALA A   1    {xxx:8.3f}{yyy:8.3f}{zzz:8.3f}  1.00  0.00           C\n')
        return
    
    def random_sample(self, n_samples=1):
        """Randomly sample points from the voxel grid."""
        xT = torch.zeros((n_samples, 14, 3), dtype=torch.float32)
        indices = np.random.choice(len(self.target_xyzs), size=n_samples, replace=False)
        ca = self.target_xyzs[indices]
        n = ca + torch.tensor([-0.5272, 1.3593, 0.000])
        c = ca + torch.tensor([1.5233, 0.000, 0.000])
        
        xT[:, 0, :] = ca
        xT[:, 1, :] = n
        xT[:, 2, :] = c
        
        return xT