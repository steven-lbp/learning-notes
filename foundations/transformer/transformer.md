# Transformer
## Basic
Scaled dot-product attention（拿每个q对每个k做attention）:
$$
Attention(Q, K, V) = softmax(\frac{Q K^{T}}{\sqrt{d_{k}}}) V
$$
```
energy = torch.einsum("nqhd,nkhd->nhqk", [query, key])
# query shape: (N, query_len, heads, heads_dim)
# keys shape: (N, key_len, heads, heads_dim)
# energy shape: (N, heads, query_len, key_len)
attention  = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)

out = torch.einsum("nhql,nlhd->nqhd", [attention, value])
out = out.reshape(N, query_len, self.heads * self.heads_dim)
# attention shape: (N, heads, query_len, key_len)
# value shape: (N, value_len, heads, heads_dim)
# energy shape:(N, query_len, heads, heads_dim)
```

### Model Inference
Two part: Prefill and Decode
- Prefill

The model performs a one-time parallel computation on all Prompt tokens and ultimately generates the first output token.

- Decode

Generate a token each time until an EOS (end-to-end) token is generated, producing the final response

## Question
### Why Does Transformer Use Multi-Head Attention?
MHA is a crucial component of the Transformer architecture. It enhances the model's ability to **capture different types of relationships** between words, enabling better contextual understanding and improving performance in various NLP tasks.

In fact, it was the original author of the paper who found that this effect was good.(Not the more the better, in the paper 8 or 16 heads are better than 4 or 32 heads)

### Why is there a scaling factor $\frac{1}{\sqrt{d_{k}}}$
The dot product ($A = QK^{T}$) of two random vectors of dimension $d_{k}$ typically has a variance proportional to $d_{k}$, assuming the elements ($Q, K$) are independent and have a mean of 0 and variance of 1.

By scaling the dot product by $\frac{1}{\sqrt{d_{k}}}$, we normalize its variance to approximately 1, preventing values in the softmax from becoming too extreme. This keeps the softmax outputs well-distributed, allowing for better gradient flow and more stable training.

### KV Cache
Transformers process input tokens sequentially.
- Without a KV cache, the model recomputes attention for all previous tokens at every step.
- With a KV cache, the model stores the computed key (K) and value (V) tensors for each token, so they don’t need to be recomputed.

Instead of reprocessing past tokens, the model retrieves stored key-value pairs and only computes attention for the new token. This speeds up inference significantly, especially for long sequences.

eg. 

input = "The largest city of China is"

expected output = "The largest city of China is Shang Hai"

1. Before generating "Shang", kKVcache multiplies the input 6 tokens by the two parameter matrices W_K and W_V, that is, 6 kvs are cached. 

2. At this time, the token "Shang" is obtained through self-attention + sampling scheme (greedy, beam, top k, top p, etc.)

3. After generating "Shang", the token that passes through the transformer again is only the token "Shang", not the entire sentence "The largest city of China is Shang". 

4. At this time, the kvcache corresponding to the token "Shang" and the kvcache corresponding to the previous 6 tokens are combined to form 7 kvcaches. The token "Shang" and the previous 6 tokens can finally generate the token "Hai"