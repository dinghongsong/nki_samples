import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from neuronxcc.nki.language import par_dim
import neuronxcc.nki.isa as nisa
import numpy as np
import argparse
from scipy.special import softmax


@nki.jit
def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref, use_causal_mask=False,
                                           mixed_precision=True):
  """
  Fused self attention kernel for small head dimension Stable Diffusion workload, 
  simplified for this tutorial. 
  
  Computes softmax(QK^T)V. Decoder model can optionally include a causal mask 
  application. Does not include QKV projection, output projection, dropout, 
  residual connection, etc.

  This kernel is designed to be used for Stable Diffusion models where the 
  d_head is smaller or equal to 128. Assertion is thrown if `d_head` does
  not satisfy the requirement.

  IO tensor layouts:
   - q_ptr: shape   (seq_q, d_head)
   - k_ptr: shape   (seq_k, d_head)
   - v_ptr: shape   (seq_v, d_head)
   - out_ptr: shape (seq_q, d_head)
   - We use seq_q and seq_k and seq_v just for clarity, this kernel requires 
   seq_q == seq_k == seq_v

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - If mixed_precision is True, then all Tensor Engine operation will be performed in
   bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
   will be in the same type as the inputs.
  """
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_precision else nl.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype

  # Shape checking
  seqlen, d_head = q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert tuple(q_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(k_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (seqlen,d_head), \
  f'Input shape mismatch! Expected: {(seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
  out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 0.125

  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  # No tiling on d_head dimension since the dimension of d_head fits in SB
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  ###################################
  # Step 1. transpose(tensor_v)
  ###################################
  # Buffer for v matrix transposed
  # Pre-fetch and keep it in SBUF throughout different softmax tiles
#   for i in nl.static_range(k_seq_n_tiles):
#     print("i: ", i)
  trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)  #defaults to sbuf.
  ip_v = nl.arange(v_seq_tile_size)[:, None]
  if_v = nl.arange(d_head_tile_size)[None, :]
  for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):
    trans_v[ip_v, i_v_seq_tile, if_v] = nl.load(
      v_ref[i_v_seq_tile * v_seq_tile_size + ip_v, if_v],
      dtype=pe_in_dt)

  q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
  ip_q = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    q_local[i_q_seq_tile, ip_q, if_q] = nl.load_transpose2d(
      q_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]
      ],
      dtype=pe_in_dt) * softmax_scale

  k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
      k_ref[i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]],
      dtype=pe_in_dt)
###########################################################################
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype) # [128, 4096]

    neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype) # [128, 32]
    ip_max = nl.arange(q_seq_tile_size)[:, None]
    if_max = nl.arange(k_seq_n_tiles)[None, :]

    # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

      # Since the K^T tile is the RHS, the q_seq_len dimension will be P in the result
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),  # [128, 128]
                         dtype=nl.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      ##############################################################
      # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
      ##############################################################
    #   qk_psum[ip_qk, if_qk]+ = nisa.nc_matmul(moving=k_local[i_k_seq_tile, ip_k, if_k],  # ip_k ip_q is partition
    #                                           stationary=q_local[i_q_seq_tile, ip_q, if_q])
      qk_psum[ip_qk, if_qk] += nisa.nc_matmul(q_local[i_q_seq_tile, ip_q, if_q],
                                             k_local[i_k_seq_tile, ip_k, if_k]  # ip_k ip_q is partition
                                              )
      ###################################
      # Step 3. Apply optional causal mask
      ###################################
      if use_causal_mask:
        # Magic number -9984.0 to replace -inf similar to what neuronx-cc uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk), # 行数大于列数
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
                                                                              dtype=kernel_dtype)

      ###################################
      # Step 4. Softmax
      ###################################
      neg_max_res[ip_qk, i_k_seq_tile] = nisa.tensor_reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final = nisa.tensor_reduce(
      np.min, data=neg_max_res[ip_max, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)  # [128, 4096]
    sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype) # [128, 64]

    # Simply use a large tile of seq_len in size since this is a "blocking" instruction
    # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
    # exp_res = nisa.activation(np.exp,
    #                           data=qk_res_buf[ip_softmax, if_softmax],
    #                           bias=neg_max_res_final, scale=1.0)
    exp_res = nisa.activation(np.exp,
                          data=qk_res_buf[ip_softmax, if_softmax],
                          bias=neg_max_res_final, scale=1.0)

    sum_res = nisa.tensor_reduce(np.add, 
                                 data=exp_res[ip_softmax, if_softmax], axis=(1,),
                          dtype=kernel_dtype)
    softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)
    # softmax_res = exp_res #work
    # sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
    # sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)
    sum_divisor[ip_sum_res, if_sum_res]  = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
####################################
    # Buffer for transposed softmax results (FP32 in PSUM)
    trans_softmax_res = nl.ndarray(  #(128, 32, 128)
      (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
      dtype=pe_in_dt)

    # Result psum buffer has the hidden dim as P
    attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),  #(64, 128)
                             dtype=nl.float32, buffer=nl.psum)
    

    ###################
    attn_res_psum2 = nl.zeros((par_dim(q_seq_tile_size), d_head_tile_size),  #(64, 128)
                             dtype=nl.float32, buffer=nl.psum)
    ####################

    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    # Loop over matmul_1 contraction
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ###################################
      # Step 5. transpose(softmax_res)
      ###################################
      ip_scores = nl.arange(q_seq_tile_size)[:, None]
      if_scores = nl.arange(k_seq_tile_size)[None, :]

      trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
        softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

    ip_out = nl.arange(d_head_tile_size)[:, None]
    if_out = nl.arange(q_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ######################################################################
      # Step 6. matmul_1(stationary=trans_v, moving=trans_softmax_res, contract=seqlen_v=seqlen_k)
      ######################################################################
      ip_v_t = nl.arange(k_seq_tile_size)[:, None]
      if_v_t = nl.arange(d_head_tile_size)[None, :]
      attn_res_psum[ip_out, if_out] += \
        nisa.nc_matmul(moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                       stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t])
      
      attn_res_psum2[nl.arange(q_seq_tile_size)[:, None], nl.arange(d_head_tile_size)[None,:]] += \
        nisa.nc_matmul(trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                       trans_v[ip_v_t, i_k_seq_tile, if_v_t])



    attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)
    attn_res_sbuf2 = nl.copy(attn_res_psum2, dtype=kernel_dtype)

    attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor[ip_sum_res, if_sum_res])
    attn_res_div2 = attn_res_sbuf2 * sum_divisor[ip_sum_res, if_sum_res]
    # nl.store(
    #   out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
    #   value=attn_res_div)

    nl.store(
      out_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(128)[:, None], nl.arange(64)[None, :]],
      value=attn_res_div2)


  return out_ref



import torch
from torch_xla.core import xla_model as xm

# from sd_attention_nki_kernels import fused_self_attn_for_SD_small_head_size


if __name__ == "__main__":

  device = xm.xla_device()
  import time
  def cpu_golden_attn(q, k, v):
      softmax_scale = 0.125
      q_scaled = q * softmax_scale
      raw_score = torch.matmul(q_scaled, k.transpose(1, 0))
      
      norm_score = torch.nn.functional.softmax(raw_score, dim=-1)

      return torch.matmul(norm_score, v)

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

  allclose = torch.allclose(output_torch, output_nki, atol=1e-5, rtol=1e-3)

  if allclose:
    print("NKI and Torch match")
  else:
    print("NKI and Torch differ")

  assert allclose