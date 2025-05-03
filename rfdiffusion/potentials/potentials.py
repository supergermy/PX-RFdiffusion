import torch
from rfdiffusion.util import generate_Cbeta
from rfdiffusion.inference import utils as iu
from rfdiffusion.voxel.Voxel import VoxelGrid
import numpy as np
from scipy.sparse.csgraph import shortest_path

class Potential:
    '''
        Interface class that defines the functions a potential must implement
    '''

    def compute(self, xyz):
        '''
            Given the current structure of the model prediction, return the current
            potential as a PyTorch tensor with a single entry

            Args:
                xyz (torch.tensor, size: [L,27,3]: The current coordinates of the sample
            
            Returns:
                potential (torch.tensor, size: [1]): A potential whose value will be MAXIMIZED
                                                     by taking a step along it's gradient
        '''
        raise NotImplementedError('Potential compute function was not overwritten')

class monomer_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging monomer compactness

        Written by DJ and refactored into a class by NRB
    '''

    def __init__(self, weight=1, min_dist=15):

        self.weight   = weight
        self.min_dist = min_dist

    def compute(self, xyz):
        Ca = xyz[:,1] # [L,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]

        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,L,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [L,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration

class binder_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging binder compactness

        Author: NRB
    '''

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, xyz):
        
        # Only look at binder residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]

        centroid = torch.mean(Ca, dim=0, keepdim=True) # [1,3]
        # centroid += torch.tensor([[105.42, 120.42, 16.95]]) # this should keep moving the centroid to the direction
        # centroid = torch.tensor([[105.42, 120.42, 16.95]]) # directly moving the centroid

        # cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), centroid[None,...].contiguous(), p=2) # [1,Lb,1,3]

        dgram = torch.maximum(self.min_dist * torch.ones_like(dgram.squeeze(0)), dgram.squeeze(0)) # [Lb,1,3]

        rad_of_gyration = torch.sqrt( torch.sum(torch.square(dgram)) / Ca.shape[0] ) # [1]

        return -1 * self.weight * rad_of_gyration


class dimer_ROG(Potential):
    '''
        Radius of Gyration potential for encouraging compactness of both monomers when designing dimers

        Author: PV
    '''

    def __init__(self, binderlen, weight=1, min_dist=15):

        self.binderlen = binderlen
        self.min_dist  = min_dist
        self.weight    = weight

    def compute(self, xyz):

        # Only look at monomer 1 residues
        Ca_m1 = xyz[:self.binderlen,1] # [Lb,3]
        
        # Only look at monomer 2 residues
        Ca_m2 = xyz[self.binderlen:,1] # [Lb,3]

        centroid_m1 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]
        centroid_m2 = torch.mean(Ca_m1, dim=0, keepdim=True) # [1,3]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 1
        dgram_m1 = torch.cdist(Ca_m1[None,...].contiguous(), centroid_m1[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m1 = torch.maximum(self.min_dist * torch.ones_like(dgram_m1.squeeze(0)), dgram_m1.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m1 = torch.sqrt( torch.sum(torch.square(dgram_m1)) / Ca_m1.shape[0] ) # [1]

        # cdist needs a batch dimension - NRB
        #This calculates RoG for Monomer 2
        dgram_m2 = torch.cdist(Ca_m2[None,...].contiguous(), centroid_m2[None,...].contiguous(), p=2) # [1,Lb,1,3]
        dgram_m2 = torch.maximum(self.min_dist * torch.ones_like(dgram_m2.squeeze(0)), dgram_m2.squeeze(0)) # [Lb,1,3]
        rad_of_gyration_m2 = torch.sqrt( torch.sum(torch.square(dgram_m2)) / Ca_m2.shape[0] ) # [1]

        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return -1 * self.weight * (rad_of_gyration_m1 + rad_of_gyration_m2)/2

class binder_ncontacts(Potential):
    '''
        Differentiable way to maximise number of contacts within a protein
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

    '''

    def __init__(self, binderlen, weight=1, r_0=8, d_0=4):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, xyz):

        # Only look at binder Ca residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]
        
        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        binder_ncontacts = (1 - numerator) / (1 - denominator)
        
        print("BINDER CONTACTS:", binder_ncontacts.sum())
        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * binder_ncontacts.sum()

class interface_ncontacts(Potential):

    '''
        Differentiable way to maximise number of contacts between binder and target
        
        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html

        Author: PV
    '''


    def __init__(self, binderlen, weight=1, r_0=8, d_0=6):

        self.binderlen = binderlen
        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0

    def compute(self, xyz):

        # Extract binder Ca residues
        Ca_b = xyz[:self.binderlen,1] # [Lb,3]

        # Extract target Ca residues
        Ca_t = xyz[self.binderlen:,1] # [Lt,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca_b[None,...].contiguous(), Ca_t[None,...].contiguous(), p=2) # [1,Lb,Lt]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        interface_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        interface_ncontacts = interface_ncontacts.sum()

        print("INTERFACE CONTACTS:", interface_ncontacts.sum())

        return self.weight * interface_ncontacts


class monomer_contacts(Potential):
    '''
        Differentiable way to maximise number of contacts within a protein

        Motivation is given here: https://www.plumed.org/doc-v2.7/user-doc/html/_c_o_o_r_d_i_n_a_t_i_o_n.html
        Author: PV

        NOTE: This function sometimes produces NaN's -- added check in reverse diffusion for nan grads
    '''

    def __init__(self, weight=1, r_0=8, d_0=2, eps=1e-6):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0
        self.eps       = eps

    def compute(self, xyz):

        Ca = xyz[:,1] # [L,3]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), Ca[None,...].contiguous(), p=2) # [1,Lb,Lb]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)

        ncontacts = (1 - numerator) / ((1 - denominator))


        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * ncontacts.sum()


class olig_contacts(Potential):
    """
    Applies PV's num contacts potential within/between chains in symmetric oligomers 

    Author: DJ 
    """

    def __init__(self, 
                 contact_matrix, 
                 weight_intra=1, 
                 weight_inter=1,
                 r_0=8, d_0=2):
        """
        Parameters:
            chain_lengths (list, required): List of chain lengths, length is (Nchains)

            contact_matrix (torch.tensor/np.array, required): 
                square matrix of shape (Nchains,Nchains) whose (i,j) enry represents 
                attractive (1), repulsive (-1), or non-existent (0) contact potentials 
                between chains in the complex

            weight (int/float, optional): Scaling/weighting factor
        """
        self.contact_matrix = contact_matrix
        self.weight_intra = weight_intra 
        self.weight_inter = weight_inter 
        self.r_0 = r_0
        self.d_0 = d_0

        # check contact matrix only contains valid entries 
        assert all([i in [-1,0,1] for i in contact_matrix.flatten()]), 'Contact matrix must contain only 0, 1, or -1 in entries'
        # assert the matrix is square and symmetric 
        shape = contact_matrix.shape 
        assert len(shape) == 2 
        assert shape[0] == shape[1]
        for i in range(shape[0]):
            for j in range(shape[1]):
                assert contact_matrix[i,j] == contact_matrix[j,i]
        self.nchain=shape[0]

         
    def _get_idx(self,i,L):
        """
        Returns the zero-indexed indices of the residues in chain i
        """
        assert L%self.nchain == 0
        Lchain = L//self.nchain
        return i*Lchain + torch.arange(Lchain)


    def compute(self, xyz):
        """
        Iterate through the contact matrix, compute contact potentials between chains that need it,
        and negate contacts for any 
        """
        L = xyz.shape[0]

        all_contacts = 0
        start = 0
        for i in range(self.nchain):
            for j in range(self.nchain):
                # only compute for upper triangle, disregard zeros in contact matrix 
                if (i <= j) and (self.contact_matrix[i,j] != 0):

                    # get the indices for these two chains 
                    idx_i = self._get_idx(i,L)
                    idx_j = self._get_idx(j,L)

                    Ca_i = xyz[idx_i,1]  # slice out crds for this chain 
                    Ca_j = xyz[idx_j,1]  # slice out crds for that chain 
                    dgram           = torch.cdist(Ca_i[None,...].contiguous(), Ca_j[None,...].contiguous(), p=2) # [1,Lb,Lb]

                    divide_by_r_0   = (dgram - self.d_0) / self.r_0
                    numerator       = torch.pow(divide_by_r_0,6)
                    denominator     = torch.pow(divide_by_r_0,12)
                    ncontacts       = (1 - numerator) / (1 - denominator)

                    # weight, don't double count intra 
                    scalar = (i==j)*self.weight_intra/2 + (i!=j)*self.weight_inter

                    #                 contacts              attr/repuls          relative weights 
                    all_contacts += ncontacts.sum() * self.contact_matrix[i,j] * scalar 

        return all_contacts 
                    
def get_damped_lj(r_min, r_lin,p1=6,p2=12):
    
    y_at_r_lin = lj(r_lin, r_min, p1, p2)
    ydot_at_r_lin = lj_grad(r_lin, r_min,p1,p2)
    
    def inner(dgram):
        return (dgram < r_lin) * (ydot_at_r_lin * (dgram - r_lin) + y_at_r_lin) + (dgram >= r_lin) * lj(dgram, r_min, p1, p2)
    return inner

def lj(dgram, r_min,p1=6, p2=12):
    return 4 * ((r_min / (2**(1/p1) * dgram))**p2 - (r_min / (2**(1/p1) * dgram))**p1)

def lj_grad(dgram, r_min,p1=6,p2=12):
    return -p2 * r_min**p1*(r_min**p1-dgram**p1) / (dgram**(p2+1))

def mask_expand(mask, n=1):
    mask_out = mask.clone()
    assert mask.ndim == 1
    for i in torch.where(mask)[0]:
        for j in range(i-n, i+n+1):
            if j >= 0 and j < len(mask):
                mask_out[j] = True
    return mask_out

def contact_energy(dgram, d_0, r_0):
    divide_by_r_0 = (dgram - d_0) / r_0
    numerator = torch.pow(divide_by_r_0,6)
    denominator = torch.pow(divide_by_r_0,12)
    
    ncontacts = (1 - numerator) / ((1 - denominator)).float()
    return - ncontacts

def poly_repulse(dgram, r, slope, p=1):
    a = slope / (p * r**(p-1))

    return (dgram < r) * a * torch.abs(r - dgram)**p * slope

#def only_top_n(dgram


class substrate_contacts(Potential):
    '''
    Implicitly models a ligand with an attractive-repulsive potential.
    '''

    def __init__(self, weight=1, r_0=8, d_0=2, s=1, eps=1e-6, rep_r_0=5, rep_s=2, rep_r_min=1):

        self.r_0       = r_0
        self.weight    = weight
        self.d_0       = d_0
        self.eps       = eps
        
        # motif frame coordinates
        # NOTE: these probably need to be set after sample_init() call, because the motif sequence position in design must be known
        self.motif_frame = None # [4,3] xyz coordinates from 4 atoms of input motif
        self.motif_mapping = None # list of tuples giving positions of above atoms in design [(resi, atom_idx)]
        self.motif_substrate_atoms = None # xyz coordinates of substrate from input motif
        r_min = 2
        self.energies = []
        self.energies.append(lambda dgram: s * contact_energy(torch.min(dgram, dim=-1)[0], d_0, r_0))
        if rep_r_min:
            self.energies.append(lambda dgram: poly_repulse(torch.min(dgram, dim=-1)[0], rep_r_0, rep_s, p=1.5))
        else:
            self.energies.append(lambda dgram: poly_repulse(dgram, rep_r_0, rep_s, p=1.5))


    def compute(self, xyz):
        
        # First, get random set of atoms
        # This operates on self.xyz_motif, which is assigned to this class in the model runner (for horrible plumbing reasons)
        self._grab_motif_residues(self.xyz_motif)
        
        # for checking affine transformation is corect
        first_distance = torch.sqrt(torch.sqrt(torch.sum(torch.square(self.motif_substrate_atoms[0] - self.motif_frame[0]), dim=-1))) 

        # grab the coordinates of the corresponding atoms in the new frame using mapping
        res = torch.tensor([k[0] for k in self.motif_mapping])
        atoms = torch.tensor([k[1] for k in self.motif_mapping])
        new_frame = xyz[self.diffusion_mask][res,atoms,:]
        # calculate affine transformation matrix and translation vector b/w new frame and motif frame
        A, t = self._recover_affine(self.motif_frame, new_frame)
        # apply affine transformation to substrate atoms
        substrate_atoms = torch.mm(A, self.motif_substrate_atoms.transpose(0,1)).transpose(0,1) + t
        second_distance = torch.sqrt(torch.sqrt(torch.sum(torch.square(new_frame[0] - substrate_atoms[0]), dim=-1)))
        assert abs(first_distance - second_distance) < 0.01, "Alignment seems to be bad" 
        diffusion_mask = mask_expand(self.diffusion_mask, 1)
        Ca = xyz[~diffusion_mask, 1]

        #cdist needs a batch dimension - NRB
        dgram = torch.cdist(Ca[None,...].contiguous(), substrate_atoms.float()[None], p=2)[0] # [Lb,Lb]

        all_energies = []
        for i, energy_fn in enumerate(self.energies):
            energy = energy_fn(dgram)
            all_energies.append(energy.sum())
        return - self.weight * sum(all_energies)

        #Potential value is the average of both radii of gyration (is avg. the best way to do this?)
        return self.weight * ncontacts.sum()

    def _recover_affine(self,frame1, frame2):
        """
        Uses Simplex Affine Matrix (SAM) formula to recover affine transform between two sets of 4 xyz coordinates
        See: https://www.researchgate.net/publication/332410209_Beginner%27s_guide_to_mapping_simplexes_affinely

        Args: 
        frame1 - 4 coordinates from starting frame [4,3]
        frame2 - 4 coordinates from ending frame [4,3]
        
        Outputs:
        A - affine transformation matrix from frame1->frame2
        t - affine translation vector from frame1->frame2
        """

        l = len(frame1)
        # construct SAM denominator matrix
        B = torch.vstack([frame1.T, torch.ones(l)])
        D = 1.0 / torch.linalg.det(B) # SAM denominator

        M = torch.zeros((3,4), dtype=torch.float64)
        for i, R in enumerate(frame2.T):
            for j in range(l):
                num = torch.vstack([R, B])
                # make SAM numerator matrix
                num = torch.cat((num[:j+1],num[j+2:])) # make numerator matrix
                # calculate SAM entry
                M[i][j] = (-1)**j * D * torch.linalg.det(num)

        A, t = torch.hsplit(M, [l-1])
        t = t.transpose(0,1)
        return A, t

    def _grab_motif_residues(self, xyz) -> None:
        """
        Grabs 4 atoms in the motif.
        Currently random subset of Ca atoms if the motif is >= 4 residues, or else 4 random atoms from a single residue
        """
        idx = torch.arange(self.diffusion_mask.shape[0])
        idx = idx[self.diffusion_mask].float()
        if torch.sum(self.diffusion_mask) >= 4:
            rand_idx = torch.multinomial(idx, 4).long()
            # get Ca atoms
            self.motif_frame = xyz[rand_idx, 1]
            self.motif_mapping = [(i,1) for i in rand_idx]
        else:
            rand_idx = torch.multinomial(idx, 1).long()
            self.motif_frame = xyz[rand_idx[0],:4]
            self.motif_mapping = [(rand_idx, i) for i in range(4)]


class binder_shell_ncontacts(Potential):
    def __init__(
        self,
        binderlen,
        weight,
        voxel_path,
        target_pdb_path,
        resolution=1,
        d_0=1,
        r_0=10, 
        cutoff=0,
    ):
        target_struct = iu.parse_pdb(target_pdb_path, parse_hetatom=True) # [Lr,14,3]
        # Zero-center positions
        N_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True) # [1,1,3]
        self.binderlen = binderlen
        self.weight=weight
        self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), N_center=N_center, cutoff=cutoff)
        self.shell = self.voxel.target_xyzs
        self.r_0 = r_0
        self.d_0 = d_0

    def compute(self, xyz):
        # Only look at binder Ca residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]
        inside = torch.tensor([self.voxel.is_inside(ca) for ca in Ca], dtype=torch.bool)
        print(f"INSIDE: {inside.sum()} / {inside.shape[0]}")
        # print(Ca.shape, self.shell.shape)
        dgram = torch.cdist(Ca[None,...].contiguous(), self.shell[None,...].contiguous(), p=2) # [1,Lb,Ns]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        shell_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        shell_ncontacts = shell_ncontacts.sum()
        
        print("SHELL CONTACTS:", shell_ncontacts.sum())
        
        return self.weight * shell_ncontacts

# class binder_core_ncontacts(Potential):
#     def __init__(
#         self,
#         binderlen,
#         weight,
#         voxel_path,
#         target_pdb_path,
#         resolution=1,
#         d_0=6,
#         r_0=8, 
#         cutoff=0,
#     ):
#         target_struct = iu.parse_pdb(target_pdb_path, parse_hetatom=True) # [Lr,14,3]
#         # Zero-center positions
#         N_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True) # [1,1,3]
#         self.binderlen = binderlen
#         self.weight=weight
#         self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), N_center=N_center, cutoff=cutoff, shell=False)
#         self.core = self.voxel.target_xyzs
#         self.r_0 = r_0
#         self.d_0 = d_0

#     def compute(self, xyz):
#         # Only look at binder Ca residues
#         Ca = xyz[:self.binderlen,1] # [Lb,3]
#         outside = torch.tensor([self.voxel.is_outside(ca) for ca in Ca], dtype=torch.bool)
#         print(f"Outside: {outside.sum()} / {outside.shape[0]}")
#         # print(Ca.shape, self.shell.shape)
#         dgram = torch.cdist(Ca[None,...].contiguous(), self.core[None,...].contiguous(), p=2) # [1,Lb,Ns]
#         divide_by_r_0 = (dgram - self.d_0) / self.r_0
#         numerator = torch.pow(divide_by_r_0,6)
#         denominator = torch.pow(divide_by_r_0,12)
#         core_ncontacts = (1 - numerator) / (1 - denominator)
#         #Potential is the sum of values in the tensor
#         core_ncontacts = core_ncontacts.sum()
        
#         print("CORE CONTACTS:", core_ncontacts.sum())
        
#         return self.weight * core_ncontacts

class monomer_shell_ncontacts(Potential):
    def __init__(
        self,
        weight,
        voxel_path,
        resolution=1,
        d_0=1,
        r_0=10, 
        cutoff=0,
        centered=True,
    ):
        self.weight=weight
        self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), cutoff=cutoff, centered=bool(centered))
        self.shell = self.voxel.target_xyzs
        self.r_0 = r_0
        self.d_0 = d_0

    def compute(self, xyz):
        # Only look at binder Ca residues
        Ca = xyz[:,1] # [Lb,3]
        inside = torch.tensor([self.voxel.is_inside(ca) for ca in Ca], dtype=torch.bool)
        print(f"INSIDE: {inside.sum()} / {inside.shape[0]}")
        # print(Ca.shape, self.shell.shape)
        dgram = torch.cdist(Ca[None,...].contiguous(), self.shell[None,...].contiguous(), p=2) # [1,Lb,Ns]
        divide_by_r_0 = (dgram - self.d_0) / self.r_0
        numerator = torch.pow(divide_by_r_0,6)
        denominator = torch.pow(divide_by_r_0,12)
        shell_ncontacts = (1 - numerator) / (1 - denominator)
        #Potential is the sum of values in the tensor
        shell_ncontacts = shell_ncontacts.sum()
        
        print("SHELL CONTACTS:", shell_ncontacts.sum())
        
        return self.weight * shell_ncontacts

# class monomer_nearest_shell_ncontacts(Potential):
#     def __init__(
#         self,
#         weight,
#         voxel_path,
#         resolution=1,
#         d_0=6,
#         r_0=8, 
#         cutoff=0,
#     ):
#         self.weight=weight
#         self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), cutoff=cutoff)
#         self.shell = self.voxel.target_xyzs
#         self.r_0 = r_0
#         self.d_0 = d_0

#     def compute(self, xyz):
#         # Only look at binder Ca residues
#         Ca = xyz[:,1] # [Lb,3]
#         inside = torch.tensor([self.voxel.is_inside(ca) for ca in Ca], dtype=torch.bool)
#         print(f"INSIDE: {inside.sum()} / {inside.shape[0]}")
        
#         # print(Ca.shape, self.shell.shape)
#         dgram = torch.cdist(Ca[None,...].contiguous(), self.shell[None,...].contiguous(), p=2) # [1,Lb,Ns]
        
#         # Get the minimum distance to the shell
#         min_dgram, _ = torch.min(dgram, dim=-1)
        
#         divide_by_r_0 = (min_dgram - self.d_0) / self.r_0
#         numerator = torch.pow(divide_by_r_0,6)
#         denominator = torch.pow(divide_by_r_0,12)
        
#         shell_ncontacts = (1 - numerator) / (1 - denominator)
        
#         #Potential is the sum of values in the tensor
#         shell_ncontacts = shell_ncontacts.sum()
        
#         print("NEAREST SHELL CONTACTS:", shell_ncontacts.sum())
        
#         return self.weight * shell_ncontacts

# class binder_nearest_shell_ncontacts(Potential):
#     def __init__(
#         self,
#         binderlen,
#         weight,
#         voxel_path,
#         target_pdb_path,
#         resolution=1,
#         d_0=6,
#         r_0=8, 
#         cutoff=0,
#     ):
#         target_struct = iu.parse_pdb(target_pdb_path, parse_hetatom=True) # [Lr,14,3]
#         # Zero-center positions
#         N_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True) # [1,1,3]
#         self.binderlen = binderlen
#         self.weight=weight
#         self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), N_center=N_center, cutoff=cutoff)
#         self.shell = self.voxel.target_xyzs
#         self.r_0 = r_0
#         self.d_0 = d_0

#     def compute(self, xyz):
#         # Only look at binder Ca residues
#         Ca = xyz[:self.binderlen,1] # [Lb,3]
        
#         inside = torch.tensor([self.voxel.is_inside(ca) for ca in Ca], dtype=torch.bool)
#         print(f"INSIDE: {inside.sum()} / {inside.shape[0]}")
        
#         dgram = torch.cdist(Ca[None,...].contiguous(), self.shell[None,...].contiguous(), p=2) # [1,Lb,Ns]
#         # Get the minimum distance to the shell
#         min_dgram, _ = torch.min(dgram, dim=-1)
#         divide_by_r_0 = (min_dgram - self.d_0) / self.r_0
#         numerator = torch.pow(divide_by_r_0,6)
#         denominator = torch.pow(divide_by_r_0,12)
#         shell_ncontacts = (1 - numerator) / (1 - denominator)
#         #Potential is the sum of values in the tensor
#         shell_ncontacts = shell_ncontacts.sum()
        
#         print("NEAREST SHELL CONTACTS:", shell_ncontacts.sum())
        
#         return self.weight * shell_ncontacts

# class shell_nearest_binder_ncontacts(Potential):
#     def __init__(
#         self,
#         binderlen,
#         weight,
#         voxel_path,
#         target_pdb_path,
#         resolution=1,
#         d_0=6,
#         r_0=8, 
#         cutoff=0,
#     ):
#         target_struct = iu.parse_pdb(target_pdb_path, parse_hetatom=True) # [Lr,14,3]
#         # Zero-center positions
#         N_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True) # [1,1,3]
#         self.binderlen = binderlen
#         self.weight=weight
#         self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), N_center=N_center, cutoff=cutoff)
#         self.shell = self.voxel.target_xyzs
#         self.r_0 = r_0
#         self.d_0 = d_0

#     def compute(self, xyz):
#         # Only look at binder Ca residues
#         Ca = xyz[:self.binderlen,1] # [Lb,3]
        
#         # inside = torch.tensor([self.voxel.is_inside(ca) for ca in Ca], dtype=torch.bool)
#         # print(f"INSIDE: {inside.sum()} / {inside.shape[0]}")
        
#         dgram = torch.cdist(Ca[None,...].contiguous(), self.shell[None,...].contiguous(), p=2) # [1,Lb,Ns]
        
#         # Get the minimum distance to the binder
#         min_dgram, _ = torch.min(dgram, dim=1)
        
#         divide_by_r_0 = (min_dgram - self.d_0) / self.r_0
#         numerator = torch.pow(divide_by_r_0,6)
#         denominator = torch.pow(divide_by_r_0,12)
        
#         shell_ncontacts = (1 - numerator) / (1 - denominator)
        
#         #Potential is the sum of values in the tensor
#         shell_ncontacts = shell_ncontacts.sum()
        
#         print("NEAREST SHELL CONTACTS:", shell_ncontacts.sum())
        
#         return self.weight * shell_ncontacts

class shell_nearest_monomer_distance(Potential):
    def __init__(
        self, 
        weight,
        voxel_path,
        resolution=1,
        cutoff=0,
        centered=True,
    ):
        self.weight = weight
        self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), cutoff=cutoff, centered=bool(centered))
        self.shell = self.voxel.target_xyzs
        
    def compute(self, xyz):
        # Only look at monomer Ca residues
        Ca = xyz[:,1] # [Lb,3]
        
        # shell -> monomer min dists
        dgram = torch.cdist(Ca[None,...].contiguous(), self.shell[None,...].contiguous(), p=2) # [1,Lb,Ns]
        # Get the minimum distance to the monomer
        min_dgram, _ = torch.min(dgram, dim=1)
        
        #Potential is the sum of values in the tensor
        min_dgram = min_dgram.sum()
        print("NEAREST SHELL DISTANCE:", -  min_dgram)
        
        return - self.weight * min_dgram

class shell_nearest_binder_distance(Potential):
    def __init__(
        self, 
        binderlen,
        weight,
        voxel_path,
        target_pdb_path,
        resolution=1,
        cutoff=0,
    ):
        target_struct = iu.parse_pdb(target_pdb_path, parse_hetatom=True) # [Lr,14,3]
        # Zero-center positions
        N_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True) # [1,1,3]
        self.binderlen = int(binderlen)
        self.weight = weight
        self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), N_center=N_center, cutoff=cutoff)
        self.shell = self.voxel.target_xyzs
        
    def compute(self, xyz):
        # Only look at binder Ca residues
        Ca = xyz[:self.binderlen,1] # [Lb,3]
        
        # shell -> binder min dists
        dgram = torch.cdist(Ca[None,...].contiguous(), self.shell[None,...].contiguous(), p=2) # [1,Lb,Ns]
        # Get the minimum distance to the binder
        min_dgram, _ = torch.min(dgram, dim=1)
        
        #Potential is the sum of values in the tensor
        min_dgram = min_dgram.sum()
        print("NEAREST SHELL DISTANCE:", -  min_dgram)
        
        return - self.weight * min_dgram

# class binder_shape(Potential):
#     def __init__(
#         self,
#         binderlen,
#         weight,
#         target_X,
#     ):
#         target_X = VoxelGrid(target_X, step=1, threshold=0).target_X
#         if torch.is_tensor(target_X):
#             target_X = target_X.cpu().data.numpy()
            
#         self.binderlen = int(binderlen)
#         self.weight = weight
#         self._map_gw_coupling_ideal_glob(target_X, binderlen)
        
#         target_X = torch.Tensor(target_X)
#         self.target_X = target_X[None, ...].clone().detach()
    
#     def optimize_couplings_sinkhorn(self, C, scale=1.0, iterations=10):
#         log_T = -C * scale

#         # Initialize normalizers
#         B, I, J = log_T.shape
#         log_u = torch.zeros((B, I), device=log_T.device)
#         log_v = torch.zeros((B, J), device=log_T.device)
#         log_a = log_u - np.log(I)
#         log_b = log_v - np.log(J)

#         # Iterate normalizers
#         for j in range(iterations):
#             log_u = log_a - torch.logsumexp(log_T + log_v.unsqueeze(1), 2)
#             log_v = log_b - torch.logsumexp(log_T + log_u.unsqueeze(2), 1)
#         log_T = log_T + log_v.unsqueeze(1) + log_u.unsqueeze(2)
#         T = torch.exp(log_T)
#         return T
    
#     def optimize_couplings_gw(
#         self, D_a, D_b, scale=200.0, iterations_outer=30,
#     ):
#         # Gromov-Wasserstein Distance
#         N_a = D_a.shape[1]
#         N_b = D_b.shape[1]
#         p_a = torch.ones_like(D_a[:, :, 0]) / N_a
#         p_b = torch.ones_like(D_b[:, :, 0]) / N_b
#         C_ab = (
#             torch.einsum("bij,bj->bi", D_a ** 2, p_a)[:, :, None]
#             + torch.einsum("bij,bj->bi", D_b ** 2, p_b)[:, None, :]
#         )
#         T_gw = torch.einsum("bi,bj->bij", p_a, p_b)
#         for i in range(iterations_outer):
#             cost = C_ab - 2.0 * torch.einsum("bik,bkl,blj->bij", D_a, T_gw, D_b)
#             T_gw = self.optimize_couplings_sinkhorn(cost, scale)

#         # Compute cost
#         cost = C_ab - 2.0 * torch.einsum("bik,bkl,blj->bij", D_a, T_gw, D_b)
#         D_gw = (T_gw * cost).sum([-1, -2]).abs().sqrt()
#         return T_gw, D_gw
    
#     def _map_gw_coupling_ideal_glob(self, target_X, binderlen):
#         target_X = torch.Tensor(target_X).float().unsqueeze(0).to('cuda')
        
#         # chain_ix = torch.arange(4 * binderlen, device='cuda') / 4.0
#         chain_ix = torch.arange(binderlen, device='cuda')
#         distance_1D = (chain_ix[None, :, None] - chain_ix[None, None, :]).abs()
#         D_model = 7.21 * distance_1D**0.322
#         D_model = D_model / D_model.mean([1, 2], keepdim=True)
        
#         D_target = self._distance_knn(target_X)
#         D_target = D_target / D_target.mean([1, 2], keepdim=True)
        
#         print(f"D_model.shape={D_model.shape}, D_target.shape={D_target.shape}")
        
#         T_gw, D_gw = self.optimize_couplings_gw(D_model, D_target)
#         self.T_gw = T_gw.clone().detach().cpu()
#         return
    
#     def _distance_knn(self, X):
#         X_np = X.cpu().data.numpy()
#         D = np.sqrt(
#             ((X_np[:, :, np.newaxis, :] - X_np[:, np.newaxis, :, :]) ** 2).sum(-1)
#         )

#         # Distance cutoff
#         D_cutoff = np.mean(np.sort(D[0, :, :], axis=-1)[:, 12])
#         D[D > D_cutoff] = 10.0 * np.max(D)
#         D = shortest_path(D[0, :, :])[np.newaxis, :, :]
#         D = torch.Tensor(D).float().to(X.device)
#         return D
    
#     def _distance(self, X_i, X_j):
#         # print(f"X_i.shape={X_i.shape}, X_j.shape={X_j.shape}")
#         dX = X_i.unsqueeze(2) - X_j.unsqueeze(1)
#         D = torch.sqrt((dX**2).sum(-1) + 1e-6)
#         return D
    
    
#     def compute(self, xyz):
#         target_X = self.target_X
#         binderlen = self.binderlen
#         binder_X = xyz[None, :binderlen, 1] # Ca
#         # print(f'target_X.shape={target_X.shape}, binder_X.shape={binder_X.shape}')
#         # print(f'target_X.device={target_X.device}, binder_X.shape={binder_X.device}')
        
#         min_rg = 2.0 * binderlen**0.333
#         max_rg = 1.5 * 2.0 * binderlen**0.4
        
#         def _center(_X):
#             _X = _X - _X.mean(1, keepdim=True)
#             return _X
        
#         def _rg(_X):
#             _X = _center(_X)
#             rsq = _X.square().sum(2, keepdim=True)
#             rg = rsq.mean(1, keepdim=True).sqrt()
#             return rg
        
#         # will disabling this move the centroid correctly
#         target_X = _center(target_X)
#         binder_X = _center(binder_X)
        
#         D_inter = self._distance(target_X, binder_X)
        
#         T_w = self.optimize_couplings_sinkhorn(D_inter)
#         # print(f'T_w.shape={T_w.shape}, T_gw.shape={self.T_gw.shape}')
#         # print(f'T_w.device={T_w.device}, T_gw.shape={self.T_gw.device}')
#         T_w = T_w + self.T_gw.permute(0,2,1) * 0.4
#         T_w = T_w / T_w.sum([-1, -2], keepdims=True)
#         D_w = (T_w * D_inter).sum([-1, -2])
        
#         print(f"GW distance={D_w}")
#         return - self.weight * D_w

class monomer_shape(Potential):
    def __init__(
        self,
        monomerlen,
        weight,
        voxel_path,
        resolution=1,
        cutoff=0,
    ):
        self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), cutoff=cutoff, shell=False)
        self.core = self.voxel.target_xyzs
            
        self.monomerlen = int(monomerlen)
        self.weight = weight
        self._map_gw_coupling_ideal_glob(self.core, self.monomerlen)
        
        self.core = torch.Tensor(self.core)
        self.core = self.core[None, ...].clone().detach()
    
    def optimize_couplings_sinkhorn(self, C, scale=1.0, iterations=10):
        log_T = -C * scale

        # Initialize normalizers
        B, I, J = log_T.shape
        log_u = torch.zeros((B, I), device=log_T.device)
        log_v = torch.zeros((B, J), device=log_T.device)
        log_a = log_u - np.log(I)
        log_b = log_v - np.log(J)

        # Iterate normalizers
        for j in range(iterations):
            log_u = log_a - torch.logsumexp(log_T + log_v.unsqueeze(1), 2)
            log_v = log_b - torch.logsumexp(log_T + log_u.unsqueeze(2), 1)
        log_T = log_T + log_v.unsqueeze(1) + log_u.unsqueeze(2)
        T = torch.exp(log_T)
        return T
    
    def optimize_couplings_gw(
        self, D_a, D_b, scale=200.0, iterations_outer=30,
    ):
        # Gromov-Wasserstein Distance
        N_a = D_a.shape[1]
        N_b = D_b.shape[1]
        p_a = torch.ones_like(D_a[:, :, 0]) / N_a
        p_b = torch.ones_like(D_b[:, :, 0]) / N_b
        C_ab = (
            torch.einsum("bij,bj->bi", D_a ** 2, p_a)[:, :, None]
            + torch.einsum("bij,bj->bi", D_b ** 2, p_b)[:, None, :]
        )
        T_gw = torch.einsum("bi,bj->bij", p_a, p_b)
        for i in range(iterations_outer):
            cost = C_ab - 2.0 * torch.einsum("bik,bkl,blj->bij", D_a, T_gw, D_b)
            T_gw = self.optimize_couplings_sinkhorn(cost, scale)

        # Compute cost
        cost = C_ab - 2.0 * torch.einsum("bik,bkl,blj->bij", D_a, T_gw, D_b)
        D_gw = (T_gw * cost).sum([-1, -2]).abs().sqrt()
        return T_gw, D_gw
    
    def _map_gw_coupling_ideal_glob(self, target_X, monomerlen):
        target_X = torch.Tensor(target_X).float().unsqueeze(0).to('cuda')
        
        # chain_ix = torch.arange(4 * monomerlen, device='cuda') / 4.0
        chain_ix = torch.arange(monomerlen, device='cuda')
        distance_1D = (chain_ix[None, :, None] - chain_ix[None, None, :]).abs()
        D_model = 7.21 * distance_1D**0.322
        D_model = D_model / D_model.mean([1, 2], keepdim=True)
        
        D_target = self._distance_knn(target_X)
        D_target = D_target / D_target.mean([1, 2], keepdim=True)
        
        print(f"D_model.shape={D_model.shape}, D_target.shape={D_target.shape}")
        
        T_gw, D_gw = self.optimize_couplings_gw(D_model, D_target)
        self.T_gw = T_gw.clone().detach().cpu()
        return
    
    def _distance_knn(self, X):
        X_np = X.cpu().data.numpy()
        D = np.sqrt(
            ((X_np[:, :, np.newaxis, :] - X_np[:, np.newaxis, :, :]) ** 2).sum(-1)
        )

        # Distance cutoff
        D_cutoff = np.mean(np.sort(D[0, :, :], axis=-1)[:, 12])
        D[D > D_cutoff] = 10.0 * np.max(D)
        D = shortest_path(D[0, :, :])[np.newaxis, :, :]
        D = torch.Tensor(D).float().to(X.device)
        return D
    
    def _distance(self, X_i, X_j):
        # print(f"X_i.shape={X_i.shape}, X_j.shape={X_j.shape}")
        dX = X_i.unsqueeze(2) - X_j.unsqueeze(1)
        D = torch.sqrt((dX**2).sum(-1) + 1e-6)
        return D
    
    
    def compute(self, xyz):
        target_X = self.core
        monomer_X = xyz[None,:,1] # Ca
        
        def _center(_X):
            _X = _X - _X.mean(1, keepdim=True)
            return _X
        
        target_X = _center(target_X)
        monomer_X = _center(monomer_X)
        
        D_inter = self._distance(target_X, monomer_X)
        
        T_w = self.optimize_couplings_sinkhorn(D_inter)
        T_w = T_w + self.T_gw.permute(0,2,1) * 0.4
        T_w = T_w / T_w.sum([-1, -2], keepdims=True)
        D_w = (T_w * D_inter).sum([-1, -2]).squeeze()
        
        print(f"GW DISTANCE: {D_w}")
        return - self.weight * D_w

# class monomer_chamfer_distance(Potential):
#     """
#     두 점 집합 간의 Chamfer Distance를 계산하는 Potential 클래스.

#     낮은 Chamfer Distance 값은 두 점 집합이 더 유사함을 의미하며,
#     이는 일반적으로 더 낮은 (더 좋은) 포텐셜 에너지에 해당합니다.
#     """
#     def __init__(
#         self, 
#         weight,
#         voxel_path,
#         resolution=1,
#         cutoff=0,
#     ):
#         """
#         binder_chamfer_distance

#         Args:
#             weight (float): 계산된 Chamfer Distance에 곱해질 가중치.
#         """
#         self.weight = weight
#         self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), cutoff=cutoff)
#         self.shell = self.voxel.target_xyzs

#     def _calculate_cd(self, p_cloud, shell_pts):
#         """
#         두 3D 포인트 클라우드 간의 Chamfer Distance를 계산하는 내부 헬퍼 함수.
#         (이전 chamfer_distance_pytorch 함수의 핵심 로직)

#         Args:
#             p_cloud (torch.Tensor): 첫 번째 포인트 클라우드 (N, 3 형태).
#             shell_pts (torch.Tensor): 두 번째 포인트 클라우드 (M, 3 형태).

#         Returns:
#             torch.Tensor: 계산된 Chamfer Distance 값 (scalar).
#         """
#         # 각 점 쌍 간의 제곱 유클리드 거리 계산
#         diff = p_cloud.unsqueeze(1) - shell_pts.unsqueeze(0) # (N, M, 3)
#         dist_matrix_sq = torch.sum(diff ** 2, dim=2) # (N, M)

#         # p_cloud -> shell_pts 최소 거리 계산
#         dists_p_to_s, _ = torch.min(dist_matrix_sq, dim=1)

#         # shell_pts -> p_cloud 최소 거리 계산
#         dists_s_to_p, _ = torch.min(dist_matrix_sq, dim=0)

#         # 각 방향의 평균 거리 계산 (빈 입력 방지)
#         term1 = torch.mean(dists_p_to_s) if dists_p_to_s.numel() > 0 else torch.tensor(0.0, device=p_cloud.device)
#         term2 = torch.mean(dists_s_to_p) if dists_s_to_p.numel() > 0 else torch.tensor(0.0, device=p_cloud.device)


#         # Chamfer Distance 계산
#         chamfer_dist = term1 + term2

#         # 개별 거리 항 반환 안 함 (필요 시 수정)
#         return -chamfer_dist


#     def compute(self, xyz):
#         """
#         입력 점들(input_points)과 목표 점들(target_points) 사이의
#         Chamfer Distance를 계산하여 포텐셜 값을 반환합니다.

#         Args:
#             xyz (torch.Tensor or array-like): Chamfer Distance를 계산할
#                 입력 점들의 집합. [N, 3] 형태의 텐서 또는 텐서로 변환 가능한 배열.

#         Returns:
#             torch.Tensor: 계산된 포텐셜 값 (weight * chamfer_distance). 스칼라 텐서.
#         """
#         # Only look at Ca residues
#         Ca = xyz[:,1] # [Lb,3]

#         # 내부 CD 계산 함수 호출
#         chamfer_dist = self._calculate_cd(Ca, self.shell)
#         print("CHAMFER DISTANCE:", chamfer_dist)

#         # 최종 포텐셜 계산
#         potential = self.weight * chamfer_dist
        

#         return potential

# class binder_chamfer_distance(Potential):
#     """
#     두 점 집합 간의 Chamfer Distance를 계산하는 Potential 클래스.

#     낮은 Chamfer Distance 값은 두 점 집합이 더 유사함을 의미하며,
#     이는 일반적으로 더 낮은 (더 좋은) 포텐셜 에너지에 해당합니다.
#     """
#     def __init__(
#         self, 
#         binderlen,
#         weight,
#         voxel_path,
#         target_pdb_path,
#         resolution=1,
#         cutoff=0,
#     ):
#         """
#         binder_chamfer_distance

#         Args:
#             weight (float): 계산된 Chamfer Distance에 곱해질 가중치.
#         """
#         target_struct = iu.parse_pdb(target_pdb_path, parse_hetatom=True) # [Lr,14,3]
#         # Zero-center positions
#         N_center = target_struct["xyz"][:, :1, :].mean(axis=0, keepdims=True) # [1,1,3]
#         self.binderlen = binderlen
#         self.weight = weight
#         self.voxel = VoxelGrid(voxel_path=voxel_path, resolution=int(resolution), N_center=N_center, cutoff=cutoff)
#         self.shell = self.voxel.target_xyzs

#     def _calculate_cd(self, p_cloud, shell_pts):
#         """
#         두 3D 포인트 클라우드 간의 Chamfer Distance를 계산하는 내부 헬퍼 함수.
#         (이전 chamfer_distance_pytorch 함수의 핵심 로직)

#         Args:
#             p_cloud (torch.Tensor): 첫 번째 포인트 클라우드 (N, 3 형태).
#             shell_pts (torch.Tensor): 두 번째 포인트 클라우드 (M, 3 형태).

#         Returns:
#             torch.Tensor: 계산된 Chamfer Distance 값 (scalar).
#         """
#         # 각 점 쌍 간의 제곱 유클리드 거리 계산
#         diff = p_cloud.unsqueeze(1) - shell_pts.unsqueeze(0) # (N, M, 3)
#         dist_matrix_sq = torch.sum(diff ** 2, dim=2) # (N, M)

#         # p_cloud -> shell_pts 최소 거리 계산
#         dists_p_to_s, _ = torch.min(dist_matrix_sq, dim=1)

#         # shell_pts -> p_cloud 최소 거리 계산
#         dists_s_to_p, _ = torch.min(dist_matrix_sq, dim=0)

#         # 각 방향의 평균 거리 계산 (빈 입력 방지)
#         term1 = torch.mean(dists_p_to_s) if dists_p_to_s.numel() > 0 else torch.tensor(0.0, device=p_cloud.device)
#         term2 = torch.mean(dists_s_to_p) if dists_s_to_p.numel() > 0 else torch.tensor(0.0, device=p_cloud.device)


#         # Chamfer Distance 계산
#         chamfer_dist = term1 + term2

#         # 개별 거리 항 반환 안 함 (필요 시 수정)
#         return -chamfer_dist


#     def compute(self, xyz):
#         """
#         입력 점들(input_points)과 목표 점들(target_points) 사이의
#         Chamfer Distance를 계산하여 포텐셜 값을 반환합니다.

#         Args:
#             xyz (torch.Tensor or array-like): Chamfer Distance를 계산할
#                 입력 점들의 집합. [N, 3] 형태의 텐서 또는 텐서로 변환 가능한 배열.

#         Returns:
#             torch.Tensor: 계산된 포텐셜 값 (weight * chamfer_distance). 스칼라 텐서.
#         """
#         # Only look at binder Ca residues
#         Ca = xyz[:self.binderlen,1] # [Lb,3]

#         # 내부 CD 계산 함수 호출
#         chamfer_dist = self._calculate_cd(Ca, self.shell)
#         print("CHAMFER DISTANCE:", chamfer_dist)

#         # 최종 포텐셜 계산
#         potential = self.weight * chamfer_dist
        

#         return potential

# Dictionary of types of potentials indexed by name of potential. Used by PotentialManager.
# If you implement a new potential you must add it to this dictionary for it to be used by
# the PotentialManager
implemented_potentials = { 'monomer_ROG':          monomer_ROG,
                           'binder_ROG':           binder_ROG,
                           'dimer_ROG':            dimer_ROG,
                           'binder_ncontacts':     binder_ncontacts,
                           'interface_ncontacts':  interface_ncontacts,
                           'monomer_contacts':     monomer_contacts,
                           'olig_contacts':        olig_contacts,
                           'substrate_contacts':   substrate_contacts,
                        #    'binder_shell':   binder_shell,
                           'monomer_shell_ncontacts': monomer_shell_ncontacts,
                        #    'monomer_nearest_shell_ncontacts': monomer_nearest_shell_ncontacts,
                        #    'binder_nearest_shell_ncontacts': binder_nearest_shell_ncontacts,
                        #    'shell_nearest_binder_ncontacts': shell_nearest_binder_ncontacts,
                           'binder_shell_ncontacts': binder_shell_ncontacts,
                        #    'binder_core_ncontacts': binder_core_ncontacts,
                           'shell_nearest_monomer_distance': shell_nearest_monomer_distance,
                           'shell_nearest_binder_distance': shell_nearest_binder_distance,
                        #    'binder_shape':   binder_shape,
                           'monomer_shape':   monomer_shape,
                        #    'monomer_chamfer_distance': monomer_chamfer_distance,
                        #    'binder_chamfer_distance': binder_chamfer_distance,
                           }

require_binderlen      = { 'binder_ROG',
                           'dimer_ROG',
                           'binder_ncontacts',
                           'interface_ncontacts',
                        #    'binder_shape',
                        #    'binder_shell',
                           'shell_nearest_binder_ncontacts',
                           'binder_shell_ncontacts',
                           'shell_nearest_binder_distance'
                        #    'binder_nearest_shell_ncontacts',
                        #    'binder_core_ncontacts',
                        #    'binder_chamfer_distance',
                           }

require_voxel_path      = { 'monomer_shell_ncontacts',
                            'binder_shell_ncontacts',
                            'shell_nearest_monomer_distance',
                            'shell_nearest_binder_distance',
                            'monomer_shape',
}

require_target_pdb_path = { 'binder_shell_ncontacts',
                            'shell_nearest_binder_distance',
}