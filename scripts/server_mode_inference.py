#!/usr/bin/env python
import re
import os, time, pickle, sys
import torch
from omegaconf import OmegaConf
import hydra
import logging
from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.inference import utils as iu
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
import glob
from hydra.core.global_hydra import GlobalHydra

def make_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(version_base=None, config_path="../config/inference", config_name="oneshot")
def main(conf: HydraConfig) -> None:
    log = logging.getLogger(__name__)
    if conf.inference.deterministic:
        make_deterministic()

    # Check for GPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        log.info(f"Found GPU with device_name {device_name}. Will run RFdiffusion on {device_name}")
    else:
        log.info("////////////////////////////////////////////////")
        log.info("///// NO GPU DETECTED! Falling back to CPU /////")
        log.info("////////////////////////////////////////////////")

    # Initialize sampler
    sampler = iu.sampler_selector(conf)

    while True:
        # Get new input PDB path
        print("\nEnter new input PDB path (or 'exit' to quit):")
        new_input = input().strip()
        
        if new_input.lower() == 'exit':
            print("Exiting program...")
            break
            
        if os.path.exists(new_input):
            # Update the input PDB path
            sampler.inf_conf.input_pdb = new_input
            sampler.inf_conf.output_prefix = f"{new_input.split('/')[-1].replace('.pdb','_fixed')}"
            log.info(f"Processing new input PDB: {new_input}")

            # Main design loop
            i_des = 0
            if conf.inference.deterministic:
                make_deterministic(i_des)

            start_time = time.time()
            out_prefix = f"{os.getcwd()}/{sampler.inf_conf.output_prefix}"
            log.info(f"Making design {out_prefix}.pdb")
            
            x_init, seq_init = sampler.sample_init()
            denoised_xyz_stack = []
            px0_xyz_stack = []
            seq_stack = []
            plddt_stack = []

            x_t = torch.clone(x_init)
            seq_t = torch.clone(seq_init)
            # Loop over number of reverse diffusion time steps.
            for t in range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1):
                px0, x_t, seq_t, plddt = sampler.sample_step(
                    t=t, x_t=x_t, seq_init=seq_t, final_step=sampler.inf_conf.final_step
                )
                px0_xyz_stack.append(px0)
                denoised_xyz_stack.append(x_t)
                seq_stack.append(seq_t)
                plddt_stack.append(plddt[0])  # remove singleton leading dimension

            # Flip order for better visualization in pymol
            denoised_xyz_stack = torch.stack(denoised_xyz_stack)
            denoised_xyz_stack = torch.flip(
                denoised_xyz_stack,
                [
                    0,
                ],
            )
            px0_xyz_stack = torch.stack(px0_xyz_stack)
            px0_xyz_stack = torch.flip(
                px0_xyz_stack,
                [
                    0,
                ],
            )

            # For logging -- don't flip
            plddt_stack = torch.stack(plddt_stack)

            # Save outputs
            os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
            final_seq = seq_stack[-1]

            # Output glycines, except for motif region
            final_seq = torch.where(
                torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1)
            )  # 7 is glycine

            bfacts = torch.ones_like(final_seq.squeeze())
            # make bfact=0 for diffused coordinates
            bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
            # pX0 last step
            out = f"{out_prefix}.pdb"

            # Now don't output sidechains
            writepdb(
                out,
                denoised_xyz_stack[0, :, :4],
                final_seq,
                sampler.binderlen,
                chain_idx=sampler.chain_idx,
                bfacts=bfacts,
            )

            # run metadata
            # trb = dict(
            #     config=OmegaConf.to_container(sampler._conf, resolve=True),
            #     plddt=plddt_stack.cpu().numpy(),
            #     device=torch.cuda.get_device_name(torch.cuda.current_device())
            #     if torch.cuda.is_available()
            #     else "CPU",
            #     time=time.time() - start_time,
            # )
            # if hasattr(sampler, "contig_map"):
            #     for key, value in sampler.contig_map.get_mappings().items():
            #         trb[key] = value
            # with open(f"{out_prefix}.trb", "wb") as f_out:
            #     pickle.dump(trb, f_out)

            if sampler.inf_conf.write_trajectory:
                # trajectory pdbs
                traj_prefix = (
                    os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix)
                )
                os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)

                out = f"{traj_prefix}_Xt-1_traj.pdb"
                writepdb_multi(
                    out,
                    denoised_xyz_stack,
                    bfacts,
                    final_seq.squeeze(),
                    use_hydrogens=False,
                    backbone_only=False,
                    chain_ids=sampler.chain_idx,
                )

                out = f"{traj_prefix}_pX0_traj.pdb"
                writepdb_multi(
                    out,
                    px0_xyz_stack,
                    bfacts,
                    final_seq.squeeze(),
                    use_hydrogens=False,
                    backbone_only=False,
                    chain_ids=sampler.chain_idx,
                )

            log.info(f"Finished design in {(time.time()-start_time)/60:.2f} minutes")


        else:
            log.error(f"File not found: {new_input}")

if __name__ == "__main__":
    main()