
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
from typing import Optional, Tuple
import sys


class PatchEmbedding(nn.Module):
    """
    Conv2d-based patch projection used in ViT-style models.

    Inputs:
        in_channels:   Input channels (e.g., 3 for RGB)
        patch_size:    Patch size (kernel=stride=patch_size)
        embedding_dim: Output embedding size per patch
        img:           Optional (C,H,W) example to precreate pos-embed length
        use_cls_token: Prepend a learnable [CLS] token if True
    """
    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 embedding_dim: int,
                 img: Optional[torch.Tensor] = None,
                 use_cls_token: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.use_cls_token = use_cls_token

        self.conv2dproj = nn.Conv2d(in_channels,
                                    embedding_dim,
                                    kernel_size=patch_size,
                                    stride=patch_size,
                                    padding=0,
                                    bias=True)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim)) if use_cls_token else None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02)

        # Precreate positional embedding if img is given; else lazy-init in forward
        if img is not None:
            assert img.ndim == 3, "img must be (C,H,W)"
            _, H, W = img.shape
            assert H % patch_size == 0 and W % patch_size == 0, \
                f"img_size must be divisible by patch_size (H={H}, W={W}, P={patch_size})"
            n_patches = (H // patch_size) * (W // patch_size)
            n_tokens = n_patches + (1 if use_cls_token else 0)
            self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, embedding_dim))
            nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02)
        else:
            self.pos_embed = None  # will be created on first forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()  # ensure float
        B, C, H, W = x.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0, f"H={H}, W={W} must be divisible by patch_size={p}"

        # (B, C, H, W) -> (B, E, H/P, W/P) -> (B, N, E)
        x = self.conv2dproj(x).flatten(2).transpose(1, 2)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,E)
            x = torch.cat([cls, x], dim=1)          # (B,N+1,E)

        # Positional embeddings (lazy init to the current token length)
        if self.pos_embed is None:
            self.pos_embed = nn.Parameter(torch.randn(1, x.shape[1], x.shape[2], device=x.device, dtype=x.dtype))
            nn.init.trunc_normal_(self.pos_embed, mean=0.0, std=0.02)
        elif self.pos_embed.shape[1] != x.shape[1]:
            raise ValueError(
                f"pos_embed length {self.pos_embed.shape[1]} != tokens {x.shape[1]} — "
                f"provide matching img_size or implement interpolation."
            )

        return x + self.pos_embed  # (B, N(+1), E)


class MultiHeadAttentionBlock(nn.Module):
    """
    Pre-norm multi-head self-attention block.
    Input/Output shape: (B, S, E) if batch_first=True. B=batch_size, S=seq_len, E=embedding_dim.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 dropout_p: float = 0.0,
                 batch_first: bool = True,
                 eps: float = 1e-5):
        super().__init__()
        assert embedding_dim % num_heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"

        self.ln = nn.LayerNorm(embedding_dim, eps=eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=batch_first,
            bias=True
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self,
                x: torch.Tensor,
                *,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                average_attn_weights: bool = True) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.ln(x)
        x, attn = self.attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=need_weights,
                            average_attn_weights=average_attn_weights)
        x = residual + self.dropout(x)
        return (x, attn) if need_weights else x


class MLPBlock(nn.Module):
    """Pre-norm MLP block with residual."""
    def __init__(self,
                 embedding_dim: int,
                 mlp_size: int | None = None,
                 mlp_ratio: float = 4.0,
                 dropout_p: float = 0.1,
                 eps: float = 1e-5,
                 activation: type[nn.Module] = nn.GELU):
        super().__init__()
        hidden = mlp_size if mlp_size is not None else int(mlp_ratio * embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            activation(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, embedding_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.ln(x))


class TransformerEncoderBlock(nn.Module):
    """One pre-norm Transformer encoder block."""
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 mlp_size: int | None = None,
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.msa = MultiHeadAttentionBlock(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout_p=attn_dropout,
            batch_first=True
        )
        self.mlp = MLPBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            dropout_p=mlp_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa(x)          # residual inside
        x = self.mlp(x)          # residual inside
        return x

    def to_device(self, device: str):
        self.to(device)


class ViTModel(nn.Module):
    """
    Minimal ViT-style classifier: PatchEmbed → N×Encoder → LN → CLS → Linear.
    """
    def __init__(self,
                 in_channels: int,
                 patch_size: int,
                 embedding_dim: int,
                 num_heads: int,
                 num_layers: int,
                 *,
                 img: Optional[torch.Tensor] = None,
                 num_classes: int = 1000,
                 mlp_dropout: float = 0.1,
                 attn_dropout: float = 0.0):
        super().__init__()

        self.patchify = PatchEmbedding(in_channels=in_channels,
                                       patch_size=patch_size,
                                       embedding_dim=embedding_dim,
                                       img=img,
                                       use_cls_token=True)

        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dropout=mlp_dropout,
                attn_dropout=attn_dropout
            ) for _ in range(num_layers)
        ])

        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(in_features=embedding_dim,
                              out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x)          # (B, 1+N, E)

        # Offload encoder layers
        for layer in self.encoder:
            x = layer(x)
        x = self.pre_head_ln(x)       # (B, 1+N, E)
        cls = x[:, 0]                 # take [CLS] → (B, E)
        logits = self.head(cls)       # (B, C)
        return logits

#-----------------------CLI Entry Point------------------------------
def _build_argparser()-> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
      description="Create a ViT Model",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  p.add_argument(
      "--in_channels",
      "--in-channels",
      dest="in_channels",
      type=int,
      default=3,
      help="Number of input channels"
  )
  p.add_argument(
      "--patch_size",
      "--patch-size",
      dest="patch_size",
      type=int,
      default=16,
      help="Patch size"
  )

  p.add_argument(
      "--embedding_dim",
      "--embedding-dim",
      dest="embedding_dim",
      type=int,
      default=768,
      help="Embedding dimension"
  )

  p.add_argument(
      "--num_heads",
      "--num-heads",
      dest="num_heads",
      type=int,
      default=12,
      help="Number of attention heads"
  )

  p.add_argument(
      "--num_layers",
      "--num-layers",
      dest="num_layers",
      type=int,
      default=12,
      help="Number of encoder layers"
  )

  p.add_argument(
      "--img_size",
      "--img-size",
      dest="img_size",
      type=int,
      default=224,
      help="Image size"
  )

  p.add_argument(
      "--num_classes",
      "--num-classes",
      dest="num_classes",
      type=int,
      default=1000,
      help="Number of output classes"
  )

  p.add_argument(
      "--mlp_dropout",
      "--mlp-dropout",
      dest="mlp_dropout",
      type=float,
      default=0.1,
      help="Dropout after MLP"
  )

  p.add_argument(
      "--attn_dropout",
      "--attn-dropout",
      dest="attn_dropout",
      type=float,
      default=0.0,
      help="Dropout after MSA"
  )

  p.add_argument(
    "--class_names",
    "--class-names",
    dest="class_names",
    type=str,
    default=None,
    help="Path to class names file. TXT: one name per line. JSON: list or {id:name} dict."
)

  p.add_argument(
      "--topk",
      type=int,
      default=5,
      help="How many top predictions to print."
  )

  p.add_argument(
      "--image-path",
      "--image-path",
      dest="image_path",
      type=str,
      default=None,
      help="Path to image to run through the model."
  )

  return p

#------------------Helper Function to get the actual class names---------
def _load_class_names(path: str) -> list[str]:
    """
    Loads class names from a TXT (one per line) or JSON file.
    JSON may be either a list ["cat","dog",...] or a dict {"0":"cat","1":"dog",...}.
    """
    import json, os

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Class names file not found: {path}")

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            # Map numeric-ish keys to a list in index order
            items = sorted(((int(k), v) for k, v in obj.items()), key=lambda kv: kv[0])
            return [str(v) for _, v in items]
        raise ValueError("Unsupported JSON schema for class names: must be list or {id:name} dict.")
    else:
        # Treat as TXT with one class per line; ignore blanks and comments
        with open(path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        return names


#-----------------------Main Block-----------------------------------
def main(argv:Optional[list[str]]=None)-> int:
  args = _build_argparser().parse_args(argv)

  try:
    model = ViTModel(
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        img=None,
        num_classes=args.num_classes,
        mlp_dropout=args.mlp_dropout,
        attn_dropout=args.attn_dropout
        )
    print(f"{model.__class__.__name__} built successfully")

    # --- Optional: run inference on an image ---
    if args.image_path:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()

        # Load optional class names
        class_names = None
        if args.class_names:
            try:
                class_names = _load_class_names(args.class_names)
                print(f"Loaded {len(class_names)} class names from {args.class_names}")
            except Exception as e:
                print(f"Warning: failed to load class names ({e}). Falling back to class_id numbers.")

        if class_names and len(class_names) != args.num_classes:
            print(f"Warning: num_classes={args.num_classes} but loaded {len(class_names)} class names.")

        # Load and preprocess the image
        img = Image.open(args.image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        x = preprocess(img).unsqueeze(0).to(device)

        # Run the model
        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            k = max(1, min(args.topk, probs.shape[-1]))
            topk = torch.topk(probs, k=k)

        print("\n📊 Top predictions:")
        for i, (p, c) in enumerate(zip(topk.values[0], topk.indices[0])):
            c_int = int(c.item())
            if class_names and 0 <= c_int < len(class_names):
                label = class_names[c_int]
                print(f"{i+1:>2}. {label:<25} prob={p.item():.4f} (id={c_int})")
            else:
                print(f"{i+1:>2}. class_id={c_int:<5} prob={p.item():.4f}")

    return 0
  except SystemExit as e:
    raise e
  except Exception as e:
    print(f"Error: {e}")
    return 1

if __name__ == "__main__":
  sys.exit(main())

  
