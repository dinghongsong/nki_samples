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
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.float16 if mix_precision else np.float32
    seqlen, d_head = q_ref.shape  # 4096, 64
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)
    softmax_scale = 0.125
    
    q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

    d_head_tile_size = d_head

    #####################################
    # 1. transpose v
    trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt) #defaults to sbuf
    ip_v = nl.arange(v_seq_tile_size)[:, None]
    if_v = nl.arange(d_head_tile_size)[None, :]
    for i in nl.affine_range(v_seq_n_tiles):
        trans_v[ip_v, i, if_v] = nl.load(v_ref[i * v_seq_tile_size + ip_v, if_v], dtype=pe_in_dt)
    
    # 2.
    q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
    ip_q = nl.arange(d_head_tile_size)[:, None]
    if_q = nl.arange(q_seq_tile_size)[None, :]
    for i in nl.affine_range(q_seq_n_tiles):
        q_local[i, ip_q, if_q] = nl.load_transpose2d(
            q_ref[i * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None], nl.arange(d_head_tile_size)[None, :]], 
            dtype=pe_in_dt) * softmax_scale
    
    # 3.
    k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
    ip_k = nl.arange(d_head_tile_size)[:, None]
    if_k = nl.arange(k_seq_tile_size)[None, :]
    for i in nl.affine_range(k_seq_n_tiles):
        k_local[i, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None], nl.arange(d_head_tile_size)[None, :]],
            dtype=pe_in_dt)
        

    #############################
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

        neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
        ip_max = nl.arange(q_seq_tile_size)[:, None]
        if_max = nl.arange(k_seq_n_tiles)[None, :]

        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=nl.float32, buffer=nl.psum)
            ip_qk = nl.arange(q_seq_tile_size)[:, None]
            if_qk = nl.arange(k_seq_tile_size)[None, :]
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(q_local[i_q_seq_tile, ip_q, if_q],
                                                   k_local[i_k_seq_tile, ip_k, if_k])
 
            if use_causal_mask:
                qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
                    pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk), # row >= collomn
                    on_true_tile=qk_psum[ip_qk, if_qk], 
                    on_false_value=-9984.0,
                    dtype=kernel_dtype
                )
            else:
                qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk], dtype=kernel_dtype)
            

            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(  #try nki.language.maximum ?
                np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk], 
                axis=(1,), dtype=kernel_dtype, negate=True)
        
        neg_max_res_final = nisa.tensor_reduce(
            np.min, data=neg_max_res[ip_max, if_max],
            axis=(1,), dtype=kernel_dtype)
       
        #################### 128 x 6096 统一softmax
        ip_softmax = nl.arange(q_seq_tile_size)[:, None]
        if_softmax = nl.arange(seqlen)[None, :]

        ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
        if_sum_res = nl.arange(d_head_tile_size)[None, :]

        softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
        sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

        exp_res = nisa.activation(np.exp, # note: partiiton <= 128, free no limit
                                  data=qk_res_buf[ip_softmax, if_softmax],
                                  bias=neg_max_res_final, scale=1.0)
        sum_res = nisa.tensor_reduce(np.add, # free no limit
                                     data=exp_res[ip_softmax, if_softmax],
                                     axis=(1,), dtype=kernel_dtype)
                                     
        softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt) #sbuf
        sum_reciprocal_broadcase = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
        sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcase, dtype=kernel_dtype)

        #################################### 算出 exp后再分开
        trans_softmax_res = nl.ndarray(
            (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )

        attn_res_psum = nl.zeros((par_dim(q_seq_tile_size), d_head_tile_size),
                                 dtype=nl.float32, buffer=nl.psum)
        
        ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
        if_scores_t = nl.arange(q_seq_tile_size)[None, :]
        
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ip_scores = nl.arange(q_seq_tile_size)[:, None]
            if_scores = nl.arange(k_seq_tile_size)[None, :]
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
                softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores]
            )
            
        
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ip_v_t = nl.arange(k_seq_tile_size)[:, None]
            if_v_t = nl.arange(d_head_tile_size)[None, :]

            attn_res_psum += nisa.nc_matmul(
                trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                trans_v[ip_v_t, i_k_seq_tile, if_v_t])

        attn_res_sbuf = nl.copy(attn_res_psum, dtype=kernel_dtype)

        attn_res_div = attn_res_sbuf * sum_divisor[ip_sum_res, if_sum_res]


        nl.store(
                 out_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(128)[:, None], nl.arange(64)[None, :]],
        value=attn_res_div)

    
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

    
    

            





