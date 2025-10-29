{
  // Sets
  RR: "\\mathbb{R}",
  CC: "\\mathbb{C}",
  NN: "\\mathbb{N}",
  ZZ: "\\mathbb{Z}",
  
  // Probability/Statistics
  EE: "\\mathbb{E}",
  PP: "\\mathbb{P}",
  Var: "\\operatorname{Var}",
  Cov: "\\operatorname{Cov}",
  KL: "\\operatorname{KL}",
  
  // Transformers/ML
  softmax: "\\operatorname{softmax}",
  attn: "\\operatorname{Attention}",
  layernorm: "\\operatorname{LayerNorm}",
  selfattn: "\\operatorname{SelfAttention}",
  crossattn: "\\operatorname{CrossAttention}",
  MHA: "\\operatorname{MultiHead}",
  FFN: "\\operatorname{FFN}",
  
  // Vectors/Matrices (bold)
  vv: ["\\mathbf{#1}", 1],
  mm: ["\\mathbf{#1}", 1],
  
  // Common transformer symbols
  QQ: "\\mathbf{Q}",
  KK: "\\mathbf{K}",
  VV: "\\mathbf{V}",
  WQ: "\\mathbf{W}_Q",
  WK: "\\mathbf{W}_K",
  WV: "\\mathbf{W}_V",
  WQK: "\\mathbf{W}_{QK}",
  WOV: "\\mathbf{W}_{OV}",
  
  // Dimensions
  //dmodel: "d_{\\text{model}}",
  dmodel: "d",
  dff: "d_{\\text{ff}}",
  dk: "d_k",
  dv: "d_v",
  Nseq: "N_{\\text{seq}}",
  Nvocab: "N_{\\text{vocab}}",
  // Physics
  Ham: "\\mathcal{H}",
  Lag: "\\mathcal{L}",
  ket: ["|#1\\rangle", 1],
  bra: ["\\langle#1|", 1],
  braket: ["\\langle#1|#2\\rangle", 2],
  
  // Operators
  tr: "\\operatorname{tr}",
  diag: "\\operatorname{diag}",
  
  // Calculus
  dd: ["\\frac{d#1}{d#2}", 2],
  pd: ["\\frac{\\partial#1}{\\partial#2}", 2],
  pdd: ["\\frac{\\partial^2#1}{\\partial#2^2}", 2],
  
  // Common functions
  sigmoid: "\\sigma",
  relu: "\\operatorname{ReLU}",
  gelu: "\\operatorname{GELU}"
}