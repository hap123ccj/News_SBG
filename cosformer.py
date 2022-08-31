import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn

class CosformerAttention(nn.Module):
	def __init__(
		self,
		embed_dim,
		num_heads,
		kdim=None,
		vdim=None,
		dropout_rate=0.0,
		causal=False,
		has_outproj=True,
		act_fun="elu",
	):
		super().__init__()
		self.embed_dim = embed_dim
		self.kdim = kdim if kdim is not None else embed_dim
		self.vdim = vdim if kdim is not None else embed_dim
		self.num_heads = num_heads
		self.has_outproj = has_outproj
		self.act_fun = self.get_act_fun(act_fun)
		# q, k, v projection
		self.k_proj = nn.Linear(self.kdim, embed_dim)
		self.v_proj = nn.Linear(self.vdim, embed_dim)
		self.q_proj = nn.Linear(embed_dim, embed_dim)
		# outprojection
		self.out_proj = nn.Linear(embed_dim, embed_dim)
		# dropout rate
		self.dropout_rate = dropout_rate
		# causal
		self.causal = causal

		assert (self.embed_dim % self.num_heads == 0), 

	def get_index(self, seq_len):
		index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

		return nn.Parameter(index, requires_grad=False)

	def get_act_fun(self, act_fun):
		if act_fun == "elu":
			return 1 + F.elu

	def forward(
		self,
		query: Tensor,
		key: Optional[Tensor] = None,
		value: Optional[Tensor] = None,
		attn_mask: Optional[Tensor] = None,
		eps: Optional[float] = 1e-6,
	):
		
		if key == None:
			key = query
		if value == None:
			value = query
		
		num_heads = self.num_heads
		tgt_len, bsz, embed_dim = query.size()
		src_len = key.size(0)
		head_dim = embed_dim // num_heads

		
		q = self.q_proj(query)
	
		k = self.k_proj(key)
		
		v = self.v_proj(value)

	
		q = self.act_fun(q)
		k = self.act_fun(k)

		
		
		q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
		
		k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
		
		v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
		

		m = max(src_len, tgt_len)
		
		weight_index = self.get_index(m).to(q)
		
		q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
		
		k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

		if self.causal:
			
			kv_ = torch.einsum("nld,nlm->nldm", k_, v)
		
			kv_cum = torch.cumsum(kv_, dim=1)
			
			qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
		
			k_cum = torch.cumsum(k_, dim=1)
			
			denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
			
			attn_output = qkv / denom.unsqueeze(-1)
			
			attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
		else:
			
			kv_ = torch.einsum('nld,nlm->ndm', k_, v)
		
			z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), eps)
		
			attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)
			
			attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
		
		if self.has_outproj:
			attn_output = self.out_proj(attn_output)

		return attn_output

	def left_product(
		self,
		query: Tensor,
		key: Optional[Tensor] = None,
		value: Optional[Tensor] = None,
		attn_mask: Optional[Tensor] = None,
		eps: Optional[float] = 1e-6,
	):
		
		
		if key == None:
			key = query
		if value == None:
			value = query
		
		num_heads = self.num_heads
		tgt_len, bsz, embed_dim = query.size()
		src_len = key.size(0)
		head_dim = embed_dim // num_heads

		
	
		q = self.q_proj(query)
		
		k = self.k_proj(key)
		
		v = self.v_proj(value)

		
		q = self.act_fun(q)
		k = self.act_fun(k)


		q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

		k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

		v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
		
	
		m = max(src_len, tgt_len)
		
		weight_index = self.get_index(m).to(q)
		
		q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
		
		k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

		
		weights = torch.bmm(q_, k_.transpose(1, 2))
		
		if self.causal:
			weights = weights.masked_fill(attn_mask==float("-inf"), 0)
		
		denom = torch.clamp_min(weights.sum(dim=-1, keepdim=True), eps)
		
		attn_weights = weights / denom
		
		attn_output = torch.bmm(attn_weights, v)
		
		attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, -1)
		
		if self.has_outproj:
			attn_output = self.out_proj(attn_output)

		return attn_output

def test(batch=2, tgt_len=10, src_len=20, embed_dim=128, num_heads=8, N=100, causal=False):
	model = CosformerAttention(embed_dim=embed_dim, num_heads=num_heads, causal=causal)
	diff = 0
	if causal:
		mask = (torch.triu(torch.ones(tgt_len, tgt_len)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf'))
	else:
		mask = None
	for i in range(N):
		query = torch.rand(tgt_len, batch, embed_dim)
		key = torch.rand(src_len, batch, embed_dim)
		value = torch.rand(src_len, batch, embed_dim)
		left_res = model.left_product(query, key, value, mask)
		right_res = model(query, key, value)
		diff += torch.norm(left_res - right_res)
	diff /= N

	if causal:
		print("Test result for causal model:")
	else:
		print("Test result for bidirectional model:")
	print(f" Is {diff}")

def main():
	test(tgt_len=10, src_len=20, causal=False)
	test(tgt_len=10, src_len=10, causal=True)

if __name__ == "__main__":
	main()