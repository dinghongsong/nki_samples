import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa
import numpy as np
from scipy.special import softmax
import torch
from torch_xla.core import xla_model as xm 
import time 

@nki.jit
def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref, use_causal_mask=False, mix_precision=True):
    pe_in_dt = nl.float32
    seqlen, d_head = q_ref.shape
    out_ref = nl.ndarray(q_ref.shape, dtype=q_ref.dtype, buffer=nl.shared_hbm) #defaults to sbuf
    softmax_scale = 0.125

    q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

    ########### load sbuf from hbm
    q_trans = nl.ndarray((d_head, q_seq_n_tiles, q_seq_tile_size), dtype=pe_in_dt)
    for i in nl.affine_range(q_seq_n_tiles):
        q_trans[nl.arange(d_head)[:, None], i, nl.arange(q_seq_tile_size)[None, :]] = nl.load_transpose2d(
            q_ref[i * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None], nl.arange(d_head)[None, :]], dtype=pe_in_dt
        ) * softmax_scale
    
    k_trans = nl.ndarray((d_head, k_seq_n_tiles, k_seq_tile_size), dtype=pe_in_dt)
    for i in nl.affine_range(k_seq_n_tiles):
        k_trans[nl.arange(d_head)[:, None], i, nl.arange(k_seq_tile_size)[None, :]] = nl.load_transpose2d(
            k_ref[i * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None], nl.arange(d_head)[None, :]]
        )
 

    v_local = nl.ndarray((v_seq_tile_size, v_seq_n_tiles, d_head), dtype=pe_in_dt)
    for i in nl.affine_range(v_seq_n_tiles):
        v_local[nl.arange(v_seq_tile_size)[:, None], i, nl.arange(d_head)[None, :]] = nl.load(
            v_ref[i * v_seq_tile_size + nl.arange(v_seq_tile_size)[:, None], nl.arange(d_head)[None, :]], dtype=pe_in_dt
        )
    
    ############
    for qi in nl.affine_range(q_seq_n_tiles):
        qi_res = nl.ndarray((q_seq_tile_size, seqlen), dtype=pe_in_dt)
        neg_max = nl.ndarray((q_seq_tile_size, k_seq_n_tiles), dtype=pe_in_dt)
        
        for ki in nl.affine_range(k_seq_n_tiles):
            qk_psum = nisa.nc_matmul(q_trans[nl.arange(d_head)[:, None], qi, nl.arange(q_seq_tile_size)[None, :]], 
                                     k_trans[nl.arange(d_head)[:, None], ki, nl.arange(k_seq_tile_size)[None, :]])
            if use_causal_mask:
                qi_res[nl.arange(q_seq_tile_size)[:, None], ki * k_seq_tile_size + nl.arange(k_seq_tile_size)[None, :]] = nisa.affine_select(
                    pred=(qi * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None] >= ki * k_seq_tile_size + nl.arange(k_seq_tile_size)[None, :]),
                    on_true_tile = qk_psum, on_false_value=-9984.0
                )
            else:
                qi_res[nl.arange(q_seq_tile_size)[:, None], ki * k_seq_tile_size + nl.arange(k_seq_tile_size)[None, :]] = nl.copy(qk_psum, dtype=pe_in_dt)
            neg_max[:, ki] = nisa.tensor_reduce(
                op=nl.max, data=qk_psum, axis=[1], dtype=pe_in_dt,negate=True
            )

        neg_max_final = nisa.tensor_reduce(op=nl.min, data=neg_max, axis=[1], dtype=pe_in_dt)
        exp_res = nisa.activation(op=nl.exp, data=qi_res, bias=neg_max_final, dtype=pe_in_dt)
        sum_res = nisa.tensor_reduce(op=nl.add, data=exp_res, axis=[1], dtype=pe_in_dt)
        

        ############# (qk^T)v
        res_psum = nl.zeros((q_seq_tile_size, d_head), dtype=pe_in_dt, buffer=nl.psum)
        for i in nl.affine_range(k_seq_n_tiles):
            tem = nisa.nc_transpose(exp_res[nl.arange(q_seq_tile_size)[:, None], i * k_seq_tile_size + nl.arange(k_seq_tile_size)[None, :]], dtype=pe_in_dt)
            res_psum += nisa.nc_matmul(tem, v_local[nl.arange(v_seq_tile_size)[:, None], i, nl.arange(d_head)[None, :]])
        res_sbuf = nl.copy(res_psum, dtype=pe_in_dt)
        res_sbuf = res_sbuf * (1.0 / sum_res)
        nl.store(out_ref[qi * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None], nl.arange(d_head)[None, :]], res_sbuf)

    return out_ref
        



def cpu_golden_attn(q, k, v):
    softmax_scale = 0.125
    q_scaled = q * softmax_scale
    raw_score = torch.matmul(q_scaled, k.transpose(1, 0))
    norm_score = torch.nn.functional.softmax(raw_score, dim=-1)
    return torch.matmul(norm_score, v)        

if __name__=="__main__":
    device = xm.xla_device()
    q_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)
    k_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)
    v_tensor = torch.rand((4096, 64), dtype=torch.float32).to(device=device)

    start_time = time.perf_counter()
    output_nki = fused_self_attn_for_SD_small_head_size(q_tensor, k_tensor, v_tensor)
    end_time = time.perf_counter()
    print("nki time: ", end_time - start_time)

    start_time = time.perf_counter()
    output_torch = cpu_golden_attn(q_tensor, k_tensor, v_tensor)
    end_time = time.perf_counter()
    print("pytorch time: ", end_time - start_time)

    allclose = torch.allclose(output_nki, output_torch, atol=1e-5, rtol=1e-3)
    if allclose:
        print("NKI and Torch match")
    else:
        print("NKI and Torch differ")

    
    

            


