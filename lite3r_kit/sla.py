"""Sparse Linear Attention (SLA) module.

Drop-in replacement for the standard scaled-dot-product Attention used by both
VGGT (vggt/layers/attention.py) and DA3 (dinov2/layers/attention.py).

Computes:
    A_lin  = phi(Q) @ ( phi(K)^T @ V ) / ( phi(Q) @ ( phi(K)^T @ 1 ) )
    A_topk = softmax( top_k_mask( Q @ K^T / sqrt(d) ) ) @ V
    out    = A_lin + lambda_l * A_topk

Reference: arXiv:2509.24006

The constructor signature matches both VGGT and DA3 Attention so the class is
drop-in interchangeable. Forward signature also matches both.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _phi_elu1(x: Tensor) -> Tensor:
    return F.elu(x) + 1.0


class SLAAttention(nn.Module):
    """Sparse Linear Attention.

    The constructor signature is a superset of VGGT/DA3 Attention so it can be
    swapped in via `attn_class=SLAAttention`. Extra SLA-specific params
    (`keep_ratio`, `lambda_init`) are read via class attributes set by
    `set_sla_defaults` or via kwargs.
    """

    DEFAULT_KEEP_RATIO: float = 0.2
    DEFAULT_LAMBDA_INIT: float = 0.5

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer=nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        keep_ratio: float | None = None,
        lambda_init: float | None = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        kr = keep_ratio if keep_ratio is not None else self.DEFAULT_KEEP_RATIO
        li = lambda_init if lambda_init is not None else self.DEFAULT_LAMBDA_INIT
        self.keep_ratio = float(kr)
        # Kept for backward-compat with old ckpts; no longer used in forward.
        self.lam = nn.Parameter(torch.tensor(float(li)))

        # Learned-residual linear-attention projection, following the SLA
        # paper's `O = O^s + Proj(O^l)` form. Init = small-std Gaussian
        # (σ=1e-3, mean=0).
        # Why not eye*0.5 (the original default): at t=0 it adds 0.5·a_lin to
        # a_topk and flips abs_rel from 0.014 (pretrained) to 0.28 BEFORE any
        # training (see /tmp/diag_sla_init.py). The pretrained-dense quality
        # is destroyed before training starts, and downstream optimisation
        # cannot recover it.
        # Why σ=1e-3 and not exactly zero: ||W||_F ≈ sqrt(d²)·σ ≈ 0.06 for
        # head_dim=64 — about 60× smaller than eye*0.5, perturbing pretrained
        # by ~1% (verified ≤0.06 abs_rel at init). Small-std init makes the
        # residual branch *non-zero by design*, defending against the
        # "dormant parameter" critique while still letting QAT scale it up to
        # compensate quantisation noise. Strict zero would also work
        # mathematically (∂L/∂W ≠ 0 at W=0) but invites reviewer pushback.
        self.proj_lin = nn.Linear(self.head_dim, self.head_dim, bias=False)
        nn.init.normal_(self.proj_lin.weight, mean=0.0, std=1e-3)

        # Optional: dual-projection ablation. With LITE3R_DUAL_PROJ=1, also
        # add a learnable `proj_topk` matrix on the sparse branch:
        #   out = proj_topk(a_topk) + proj_lin(a_lin)
        # proj_topk init = identity + small Gaussian, so at t=0 the sparse
        # branch passes through unchanged (preserving pretrained behaviour).
        # Hypothesis: giving QAT a per-channel knob on the dominant a_topk
        # output lets it absorb quantisation noise WITHOUT having to drag the
        # qkv weights themselves away from pretrained.
        import os as _os
        self.use_proj_topk = _os.environ.get("LITE3R_DUAL_PROJ", "0") == "1"
        if self.use_proj_topk:
            self.proj_topk = nn.Linear(self.head_dim, self.head_dim, bias=False)
            with torch.no_grad():
                self.proj_topk.weight.copy_(torch.eye(self.head_dim))
                self.proj_topk.weight.add_(
                    torch.randn_like(self.proj_topk.weight) * 1e-3
                )

        # Faithful 3-way block-wise SLA (per arXiv:2509.24006 paper).
        # When LITE3R_SLA_3WAY=1, classify Q×K block pairs into critical/
        # marginal/negligible via a compressed pooled attention map, then run
        # sparse softmax on critical blocks, masked-aggregate linear attention
        # on marginal blocks, and ignore negligible. Default OFF (uses our
        # simpler 2-way: top-k softmax + full-N linear attention).
        self.use_sla_3way = _os.environ.get("LITE3R_SLA_3WAY", "0") == "1"
        self.sla_block_size = int(_os.environ.get("LITE3R_SLA_BLOCK", "64"))
        self.sla_kh = float(_os.environ.get("LITE3R_SLA_KH", "0.05"))
        self.sla_kl = float(_os.environ.get("LITE3R_SLA_KL", "0.10"))

        self._last_output_for_kd: Tensor | None = None

    def _linear_branch(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q,k,v: [B, H, N, d]
        phi_q = _phi_elu1(q)
        phi_k = _phi_elu1(k)
        # KV: [B, H, d, d]
        kv = torch.einsum("bhnd,bhne->bhde", phi_k, v)
        num = torch.einsum("bhnd,bhde->bhne", phi_q, kv)
        # 1: [B, H, d, 1]
        k_sum = phi_k.sum(dim=-2, keepdim=True).transpose(-1, -2)
        den = torch.einsum("bhnd,bhdo->bhno", phi_q, k_sum)
        return num / (den + 1e-6)

    def _topk_branch(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q,k,v: [B, H, N, d]
        N = q.shape[-2]
        kk = max(1, int(self.keep_ratio * N))
        # full N x N affinity (O(N^2) — acceptable for our training sizes)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B,H,N,N]
        if kk < N:
            topk_vals, topk_idx = scores.topk(kk, dim=-1)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            scores = mask
        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return torch.matmul(attn, v)

    def _topk_branch_block_sparse(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Block-level top-k sparse attention (inference-only fast path).

        Replaces token-level top-k (which still computes full N×N affinity)
        with block-level: pool Q/K into blocks of size `bs`, score block-pairs
        (n_blk×n_blk, ~500 dot-prods for N=1369/bs=64), per query block keep
        kb=⌈keep_ratio·n_blk⌉ key blocks, gather the selected key/value tokens
        and run dense SDPA on the reduced (bs, kb·bs) per query block.

        FLOPs on attention: O(N²) → O(N·kb·bs) (~3-4× saving at keep_ratio=0.3,
        bs=64, N=1369). Train-deploy mismatch: training used token-level top-k;
        accuracy drop is the main thing to verify empirically.
        """
        import os as _os
        bs = int(_os.environ.get("LITE3R_BLOCK_SIZE", "64"))
        B, H, N, d = q.shape
        pad = (bs - N % bs) % bs
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        N_p = N + pad
        n_blk = N_p // bs

        q_grouped = q.reshape(B, H, n_blk, bs, d)
        k_grouped = k.reshape(B, H, n_blk, bs, d)
        v_grouped = v.reshape(B, H, n_blk, bs, d)
        q_pool = q_grouped.mean(dim=-2)              # [B,H,n_blk,d]
        k_pool = k_grouped.mean(dim=-2)              # [B,H,n_blk,d]

        # block-level affinity, no N²
        block_scores = torch.matmul(q_pool, k_pool.transpose(-1, -2)) * self.scale

        kb = max(1, int(round(self.keep_ratio * n_blk)))
        if kb >= n_blk:
            return F.scaled_dot_product_attention(q, k, v, scale=self.scale)[:, :, :N, :]

        # per query block, top-kb key blocks
        topk_blk_idx = block_scores.topk(kb, dim=-1).indices  # [B,H,n_blk,kb]

        # advanced indexing to gather selected key/value blocks:
        # broadcast (bb, hh, topk_blk_idx) over (B, H, n_blk, kb) and pick from
        # k_grouped along its 3rd dim (the n_blk_KEY axis).
        bb = torch.arange(B, device=q.device).view(B, 1, 1, 1)
        hh = torch.arange(H, device=q.device).view(1, H, 1, 1)
        k_red = k_grouped[bb, hh, topk_blk_idx]      # [B,H,n_blk,kb,bs,d]
        v_red = v_grouped[bb, hh, topk_blk_idx]
        k_red = k_red.reshape(B, H, n_blk, kb * bs, d)
        v_red = v_red.reshape(B, H, n_blk, kb * bs, d)

        # SDPA batched over n_blk: query L=bs, key L=kb*bs
        out = F.scaled_dot_product_attention(
            q_grouped, k_red, v_red, scale=self.scale
        )                                             # [B,H,n_blk,bs,d]
        out = out.reshape(B, H, N_p, d)
        if pad:
            out = out[:, :, :N, :]
        return out

    def _topk_branch_sage(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Dense INT8 attention via SageAttention (thu-ml/SageAttention).

        Replaces the entire top-k masked attention with SageAttention's
        INT8 fused-kernel dense attention. The top-k mask is DROPPED
        because Sage does not accept arbitrary attn_mask. Net effect at
        deploy:
          - Computational pattern shifts from "fp16 dense q@k^T + topk mask
            + softmax + @v" to "INT8 fused softmax-attention".
          - Top-k mask had no FLOPs benefit (full N² computed before mask),
            so removing it costs little; INT8 kernel is the win.
          - proj_lin was trained for top-k attention output; running with
            dense INT8 attention introduces a train-deploy mismatch. Quality
            is acceptable because (a) dense → strictly more attention info,
            (b) top-k preserves ~30% of attention mass — the dropped mask
            mostly affected attention to "near-zero score" tokens anyway.
        """
        from sageattention import sageattn
        # smooth_k=True is Sage's recommended default: subtracts mean of K
        # along seq-dim before INT8 quantization, reducing K-outlier impact
        # on attention scores. Empirically gives small precision gain.
        # Controllable via env LITE3R_SAGE_SMOOTH_K=0 to disable.
        import os as _os
        smooth_k = _os.environ.get("LITE3R_SAGE_SMOOTH_K", "1") == "1"
        return sageattn(q, k, v, tensor_layout="HND", is_causal=False,
                        sm_scale=self.scale, smooth_k=smooth_k)

    def _topk_branch_thuml(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Sparse top-k attention via thu-ml/SLA official Triton kernel.

        Replaces our N×N + mask `_topk_branch` with the IO-aware Triton
        kernel from arXiv:2509.24006. Block-level top-k (BLKQ=BLKK=64) — same
        granularity as our `_topk_branch_block_sparse` but the kernel skips
        the dense N² scores entirely (loads only the selected k-blocks per
        q-block via the `lut` indirection table).

        Quality risk: block-level granularity matches our G-series experiments
        (which gave abs_rel 0.115). Mitigation: pre-trained ckpt was trained
        with token-level dense top-k → block-level deploy may have train/eval
        mismatch. Empirically verify on BlendedMVS first.

        Requires: thu-ml/SLA installed. Inference-only here (kernel supports
        backward but we deploy a frozen ckpt).
        """
        from sparse_linear_attention.kernel import _attention as _SLAAttnFn
        from sparse_linear_attention.utils import get_block_map
        BLKQ, BLKK = 64, 64
        # thu-ml kernel requires bf16/fp16 contiguous
        orig_dtype = q.dtype
        q_ = q.contiguous().to(torch.bfloat16)
        k_ = k.contiguous().to(torch.bfloat16)
        v_ = v.contiguous().to(torch.bfloat16)
        sparse_map, lut, real_topk = get_block_map(
            q_, k_, topk_ratio=float(self.keep_ratio),
            BLKQ=BLKQ, BLKK=BLKK,
        )
        out = _SLAAttnFn.apply(
            q_, k_, v_, sparse_map, lut, real_topk,
            BLKQ, BLKK, None,  # qk_scale=None defaults to head_dim**-0.5
        )
        return out.to(orig_dtype)

    def _topk_branch_sdpa(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Inference-only fused-SDPA path.

        Trades one extra q@k^T (to obtain top-k indices) for fused
        softmax+matmul via PyTorch's mem-efficient / Flash kernel. Net win
        depends on whether the SDPA fusion saves more memory bandwidth than
        the extra matmul costs; at N≈1369, d=64 typically yields ~5-10%.
        """
        N = q.shape[-2]
        kk = max(1, int(self.keep_ratio * N))
        if kk >= N:
            return F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # Use unscaled scores for topk selection (scale doesn't change argmax).
        scores = torch.matmul(q, k.transpose(-1, -2))
        topk_idx = scores.topk(kk, dim=-1).indices
        bool_mask = torch.zeros_like(scores, dtype=torch.bool)
        bool_mask.scatter_(-1, topk_idx, True)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=bool_mask, scale=self.scale
        )

    def _sla_3way_branches(self, q: Tensor, k: Tensor, v: Tensor):
        """Faithful 3-way block-wise SLA per arXiv:2509.24006.

        1. Mean-pool Q, K within blocks of size `sla_block_size`.
        2. Compute compressed attention map Pc = softmax(Q_pool · K_pool^T).
        3. Per query-block row: top kh% blocks → critical (Mc=1),
           bottom kl% blocks → negligible (Mc=-1), middle → marginal (Mc=0).
        4. Sparse softmax on critical key-blocks (per query-block).
        5. Masked-aggregate linear attention over marginal blocks.

        Returns (a_topk, a_lin), each [B, H, N, d].
        """
        B, H, N, d = q.shape
        bs = self.sla_block_size
        # Pad N to multiple of bs (pad keys & values; queries also padded so
        # output reshape works; padded queries discarded at end).
        pad = (bs - N % bs) % bs
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        N_p = N + pad
        n_blk = N_p // bs

        # Compressed-attention block classification
        q_pool = q.reshape(B, H, n_blk, bs, d).mean(dim=3)  # [B, H, n_blk, d]
        k_pool = k.reshape(B, H, n_blk, bs, d).mean(dim=3)  # [B, H, n_blk, d]
        Pc = (q_pool @ k_pool.transpose(-2, -1)) * self.scale
        Pc = Pc.softmax(dim=-1)  # [B, H, n_blk, n_blk]

        k_crit = max(1, int(self.sla_kh * n_blk))
        k_neg = max(1, int(self.sla_kl * n_blk))

        crit_vals, crit_idx = Pc.topk(k_crit, dim=-1)        # [B, H, n_blk, k_crit]
        neg_vals,  neg_idx  = (-Pc).topk(k_neg, dim=-1)      # [B, H, n_blk, k_neg]
        crit_blk_mask = torch.zeros_like(Pc, dtype=torch.bool).scatter_(-1, crit_idx, True)
        neg_blk_mask  = torch.zeros_like(Pc, dtype=torch.bool).scatter_(-1, neg_idx,  True)
        marg_blk_mask = ~(crit_blk_mask | neg_blk_mask)      # [B, H, n_blk, n_blk]

        # ---- Sparse branch: softmax over critical key-blocks ---------------
        # token indices for each query-block i: gather k_crit blocks * bs tokens
        bs_range = torch.arange(bs, device=q.device)
        crit_tok_idx = (crit_idx.unsqueeze(-1) * bs + bs_range).reshape(
            B, H, n_blk, k_crit * bs
        )
        idx_exp = crit_tok_idx.unsqueeze(-1).expand(-1, -1, -1, -1, d)  # [B,H,n_blk,k_crit*bs,d]
        K_crit = torch.gather(k.unsqueeze(2).expand(-1, -1, n_blk, -1, -1), 3, idx_exp)
        V_crit = torch.gather(v.unsqueeze(2).expand(-1, -1, n_blk, -1, -1), 3, idx_exp)

        Q_blk = q.reshape(B, H, n_blk, bs, d)
        scores_c = (Q_blk @ K_crit.transpose(-2, -1)) * self.scale  # [B,H,n_blk,bs,k_crit*bs]
        attn_c = scores_c.softmax(dim=-1)
        attn_c = self.attn_drop(attn_c)
        a_topk_blk = attn_c @ V_crit                                  # [B,H,n_blk,bs,d]
        a_topk = a_topk_blk.reshape(B, H, N_p, d)

        # ---- Linear branch: per-query-block KV aggregate over marginals ----
        # phi(K)^T V per block: [B, H, n_blk, d, d]
        phi_q = _phi_elu1(q)
        phi_k = _phi_elu1(k)
        phi_k_blk = phi_k.reshape(B, H, n_blk, bs, d)
        v_blk = v.reshape(B, H, n_blk, bs, d)
        KV_per_blk = torch.einsum("bhjte, bhjtf -> bhjef", phi_k_blk, v_blk)  # [B,H,n_blk,d,d]
        # Per-query-block aggregate over marginal key-blocks
        marg_f = marg_blk_mask.to(phi_q.dtype)
        KV_marg = torch.einsum("bhij, bhjef -> bhief", marg_f, KV_per_blk)    # [B,H,n_blk,d,d]
        # Normalise: phi(K)·1 summed over marginal positions per query-block
        phi_k_blk_sum = phi_k_blk.sum(dim=3)                                  # [B,H,n_blk,d]
        K_sum_marg = torch.einsum("bhij, bhjd -> bhid", marg_f, phi_k_blk_sum)  # [B,H,n_blk,d]

        phi_q_blk = phi_q.reshape(B, H, n_blk, bs, d)                         # [B,H,n_blk,bs,d]
        num = torch.einsum("bhite, bhief -> bhitf", phi_q_blk, KV_marg)       # [B,H,n_blk,bs,d]
        # denom = phi_q · K_sum_marg  → [B,H,n_blk,bs,1]
        den = torch.einsum("bhite, bhie -> bhit", phi_q_blk, K_sum_marg).unsqueeze(-1)
        a_lin_blk = num / (den + 1e-6)
        a_lin = a_lin_blk.reshape(B, H, N_p, d)

        if pad:
            a_topk = a_topk[:, :, :N, :]
            a_lin  = a_lin[:, :, :N, :]
        return a_topk, a_lin

    def forward(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        if attn_mask is not None:
            # SLA does not currently support arbitrary attn_mask; fall back to
            # standard scaled-dot-product so the model still runs end-to-end.
            return self._fallback_sdpa(x, pos=pos, attn_mask=attn_mask)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.use_sla_3way:
            # Faithful 3-way (block-wise) per arXiv:2509.24006
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                a_topk, a_lin = checkpoint(self._sla_3way_branches, q, k, v,
                                           use_reentrant=False)
            else:
                a_topk, a_lin = self._sla_3way_branches(q, k, v)
        else:
            a_lin = self._linear_branch(q, k, v)        # [B, H, N, d]
            # Top-K branch allocates an O(B*H*N*N) scores tensor. With KD
            # (teacher+student double forward) this OOMs on a 40GB A100 at N=1373.
            # Gradient checkpointing trades a re-forward for activation memory:
            # only q/k/v are saved across the boundary; scores/mask/attn are
            # re-computed during backward.
            # Topk branch dispatch. Env flag mapping:
            #   LITE3R_BLOCK_SPARSE=1 : block-level top-k (both train and eval)
            #   LITE3R_SAGE_ATTN=1   : eval uses SageAttention INT8 dense;
            #                          train uses fp16 SDPA dense (Sage has
            #                          no backward — fp16 SDPA is the
            #                          differentiable proxy with the same
            #                          mathematical op, only precision diff)
            #   LITE3R_TOPK_SDPA=1   : inference-only fused-SDPA top-k
            import os as _os
            use_block_sparse = _os.environ.get("LITE3R_BLOCK_SPARSE", "0") == "1"
            use_sage = _os.environ.get("LITE3R_SAGE_ATTN", "0") == "1"
            use_thuml = _os.environ.get("LITE3R_TOPK_THUML", "0") == "1"
            if self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                if use_sage:
                    # Sage has no backward; use dense fp16 SDPA as proxy.
                    # Note: this drops the top-k mask during training so
                    # proj_lin learns to compensate dense attention output
                    # (matching deploy-time Sage which is also dense).
                    def _sdpa_dense(q_, k_, v_):
                        return F.scaled_dot_product_attention(
                            q_, k_, v_, scale=self.scale)
                    branch_fn = _sdpa_dense
                elif use_block_sparse:
                    branch_fn = self._topk_branch_block_sparse
                else:
                    branch_fn = self._topk_branch
                a_topk = checkpoint(branch_fn, q, k, v, use_reentrant=False)
            else:
                if use_thuml:
                    a_topk = self._topk_branch_thuml(q, k, v)
                elif use_block_sparse:
                    a_topk = self._topk_branch_block_sparse(q, k, v)
                elif use_sage:
                    a_topk = self._topk_branch_sage(q, k, v)
                elif _os.environ.get("LITE3R_TOPK_SDPA", "0") == "1":
                    a_topk = self._topk_branch_sdpa(q, k, v)
                else:
                    a_topk = self._topk_branch(q, k, v)
        # New mixing: a_topk is the dominant signal (it inherits pretrained
        # softmax behaviour via shared qkv), and a_lin contributes via a
        # learnable projection. Replaces the old `a_lin + lam*a_topk` which
        # collapsed because both branches were too correlated for any scalar
        # lam to find a useful mix.
        if self.use_proj_topk:
            out = self.proj_topk(a_topk) + self.proj_lin(a_lin)
        else:
            out = a_topk + self.proj_lin(a_lin)         # [B, H, N, d]
        out = out.transpose(1, 2).reshape(B, N, C)   # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)
        # save attention output for distillation hooks
        self._last_output_for_kd = out
        return out

    def _fallback_sdpa(self, x: Tensor, pos=None, attn_mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask[:, None].expand(-1, self.num_heads, -1, -1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            attn_mask=attn_mask,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        self._last_output_for_kd = out
        return out


def replace_attention_with_sla(
    module: nn.Module,
    keep_ratio: float = 0.2,
    lambda_init: float = 0.5,
    target_class_names: tuple[str, ...] = ("Attention", "MemEffAttention"),
) -> int:
    """Recursively replace any submodule whose class name is in
    `target_class_names` with an SLAAttention that copies over qkv/proj weights.

    Returns the number of modules replaced.
    """
    n_replaced = 0
    for name, child in list(module.named_children()):
        cls_name = child.__class__.__name__
        if cls_name in target_class_names:
            new = _build_sla_from_existing(child, keep_ratio, lambda_init)
            setattr(module, name, new)
            n_replaced += 1
        else:
            n_replaced += replace_attention_with_sla(
                child, keep_ratio, lambda_init, target_class_names
            )
    return n_replaced


def _build_sla_from_existing(old: nn.Module, keep_ratio: float, lambda_init: float) -> SLAAttention:
    qkv: nn.Linear = old.qkv  # type: ignore[assignment]
    proj: nn.Linear = old.proj  # type: ignore[assignment]
    dim = qkv.in_features
    num_heads = old.num_heads
    qkv_bias = qkv.bias is not None
    proj_bias = proj.bias is not None
    qk_norm = not isinstance(old.q_norm, nn.Identity)
    rope = getattr(old, "rope", None)
    sla = SLAAttention(
        dim=dim,
        num_heads=num_heads,
        qkv_bias=qkv_bias,
        proj_bias=proj_bias,
        qk_norm=qk_norm,
        rope=rope,
        keep_ratio=keep_ratio,
        lambda_init=lambda_init,
    )
    with torch.no_grad():
        sla.qkv.weight.copy_(qkv.weight)
        if qkv_bias:
            sla.qkv.bias.copy_(qkv.bias)
        sla.proj.weight.copy_(proj.weight)
        if proj_bias:
            sla.proj.bias.copy_(proj.bias)
        if qk_norm and hasattr(old.q_norm, "weight"):
            sla.q_norm.weight.copy_(old.q_norm.weight)
            sla.k_norm.weight.copy_(old.k_norm.weight)
            if hasattr(old.q_norm, "bias"):
                sla.q_norm.bias.copy_(old.q_norm.bias)
                sla.k_norm.bias.copy_(old.k_norm.bias)
    return sla
