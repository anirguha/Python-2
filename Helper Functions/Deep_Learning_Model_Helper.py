
"""Multiclass Classification - """


# Import Libraries
from typing import Optional, List, Tuple, Sequence, Union, Mapping, Any

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler






import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,              # change to DEBUG for more details
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

device

"""# Import dataset from scikit-learn dataset **make-blobs**"""



from sklearn.datasets import make_blobs


"""# Build a Multi-class Classifciation Model"""

from typing import Iterable, Optional, Sequence, Union, List, Callable

ModuleLike = Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

def _broadcast(value, length: int):
    """Turn a scalar/bool/module into a length-sized list; validate list length."""
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, nn.Module)):
        value = list(value)
        if len(value) != length:
            raise ValueError(f"Expected list of length {length}, got {len(value)}")
        return value
    return [value for _ in range(length)]

class MultiClassClassifier(nn.Module):
    """
    input_features: int               Number of input features
    output_features: int              Number of output features
    hidden_layers: Sequence[int]      Hidden layer widths, e.g. [128, 64, 32]
    activation_funcs: Module or list  Activation per hidden layer (broadcast if single), e.g. nn.ReLU()
    dropout_probs: float or list      Dropout p per hidden layer (broadcast if single), or 0.0 for none
    batch_norm: bool or list[bool]    Enable BN per hidden layer (broadcast if single), or False/None for none
    out_activation: Optional Module   Optional output activation (e.g. nn.Sigmoid(), nn.Softmax(dim=1))
    """
    def __init__(
        self,
        input_features: int,
        output_features: int,
        hidden_layers: Sequence[int],
        *,
        activation_funcs: Optional[Union[ModuleLike, Sequence[ModuleLike]]] = nn.ReLU(),
        dropout_probs: Optional[Union[float, Sequence[float]]] = 0.0,
        batch_norm: Optional[Union[bool, Sequence[bool]]] = None,
        out_activation: Optional[ModuleLike] = None,
    ):
        super().__init__()

        if not hidden_layers:
            raise ValueError("At least one hidden layer is required.")

        L = len(hidden_layers)

        # Broadcast configs
        acts = _broadcast(activation_funcs if activation_funcs is not None else nn.Identity(), L)
        drops = _broadcast(dropout_probs if dropout_probs is not None else 0.0, L)
        if batch_norm is None:
            bns = [False] * L
        else:
            bns = _broadcast(batch_norm, L)

        # Validate values
        for p in drops:
            if not (0.0 <= float(p) < 1.0):
                raise ValueError(f"Dropout p must be in [0,1), got {p}")

        # Build per-layer blocks
        linears: List[nn.Linear] = []
        bnl: List[nn.BatchNorm1d] = []
        actl: List[nn.Module] = []
        dropl: List[nn.Dropout] = []

        in_f = input_features
        for i, out_f in enumerate(hidden_layers):
            linears.append(nn.Linear(in_f, out_f))
            if bns[i]:
                bnl.append(nn.BatchNorm1d(out_f))
            else:
                bnl.append(nn.Identity())  # keeps indexing simple
            actl.append(acts[i] if isinstance(acts[i], nn.Module) else nn.Identity())
            dropl.append(nn.Dropout(float(drops[i])) if float(drops[i]) > 0 else nn.Identity())
            in_f = out_f

        self.hidden_linears = nn.ModuleList(linears)
        self.hidden_bns     = nn.ModuleList(bnl)
        self.hidden_acts    = nn.ModuleList(actl)
        self.hidden_drops   = nn.ModuleList(dropl)

        self.out = nn.Linear(hidden_layers[-1], output_features)
        self.out_activation = out_activation if isinstance(out_activation, nn.Module) else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin, bn, act, drop in zip(self.hidden_linears, self.hidden_bns, self.hidden_acts, self.hidden_drops):
            x = lin(x)
            x = bn(x)
            x = act(x)
            x = drop(x)

        x = self.out(x)
        if self.out_activation is not None:
            x = self.out_activation(x)
        return x

# Function to initialize all paaremeters and gradients
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

    m.zero_grad(set_to_none=True)

# function to create a loss module
def build_loss(
    lossfunc: Union[str, nn.Module, type, None],
    *,
    device: torch.device
) -> nn.Module:
    """Return an instantiated loss module from a string / class / instance."""
    if isinstance(lossfunc, nn.Module):
        return lossfunc


    if lossfunc is None:
        return nn.CrossEntropyLoss()

    if isinstance(lossfunc, type) and issubclass(lossfunc, nn.modules.loss._Loss):
        return lossfunc()

    if isinstance(lossfunc, str):
        name = lossfunc.strip().lower()
        aliases = {
            "ce": "CrossEntropyLoss",
            "cross_entropy": "CrossEntropyLoss",
            "crossentropy": "CrossEntropyLoss",
            "bcewithlogits": "BCEWithLogitsLoss",
            "bce_logits": "BCEWithLogitsLoss",
            "bce": "BCELoss",
            "mse": "MSELoss",
            "l1": "L1Loss",
            "smoothl1": "SmoothL1Loss",
            "huber": "HuberLoss",
            "nll": "NLLLoss",
            "kldiv": "KLDivLoss",
        }

        cls_name = aliases.get(name)
        cls = getattr(nn, cls_name, None) if cls_name else getattr(nn, lossfunc, None)

        # Fallback: try TitleCase + 'Loss' heuristic
        if cls is None:
            guess = "".join(part.capitalize() for part in name.split("_"))
            if not guess.endswith("Loss"): guess += "Loss"
            cls = getattr(nn, guess, None)

        if cls is None or not issubclass(cls, nn.modules.loss._Loss):
            raise ValueError(f"Unknown loss function: {lossfunc!r}")

        return cls()

    raise TypeError("lossfunc must be a string, a loss class, a loss instance, or None.")

# Function to create optimizer
def create_optimizer(
    model: nn.Module,
    optimizer: Union[str, type],
    lr: float = 1e-3,
):

  if isinstance(optimizer, str):
          name = optimizer.strip()
          # small, case-insensitive map for common opts
          opt_map = {
              "sgd": optim.SGD,
              "adam": optim.Adam,
              "adamw": optim.AdamW,
              "rmsprop": optim.RMSprop,
              "adagrad": optim.Adagrad,
          }
          OptClass = opt_map.get(name.lower())
          if OptClass is None:
              # fall back to getattr with exact name if user passed e.g. "AdamW"
              try:
                  OptClass = getattr(optim, name)
              except AttributeError as e:
                  raise ValueError(f"Unknown optimizer '{optimizer}'.") from e
  elif isinstance(optimizer, type):
          OptClass = optimizer
  else:
          raise TypeError("optimizer must be a string name or an optimizer class.")

  optimizer = OptClass(model.parameters(), lr=lr)

  return optimizer

# Function to create the model


def createModel(
    input_features: int,
    output_features: int,
    hidden_layers: Sequence[int],
    *,
    activation_funcs: Union[nn.Module, Sequence[nn.Module]] = nn.ReLU(),
    dropout_probs: Optional[Union[float, Sequence[float]]] = None,
    batch_norm: Optional[Union[bool, Sequence[bool]]] = None,
    out_activation: Optional[nn.Module] = None,
    optimizer: Union[str, type] = "SGD",
    learning_rate: float = 1e-2,
    lossfunc: Union[str, nn.Module, type, None] = None,
    device: Optional[Union[str, torch.device]] = None
):

    model = MultiClassClassifier(input_features=input_features,
                                 output_features=output_features,
                                 hidden_layers=hidden_layers,
                                 activation_funcs=activation_funcs,
                                 dropout_probs=dropout_probs,
                                 batch_norm=batch_norm,
                                 out_activation=out_activation)

    model.apply(init_weights)

    model = model.to(device)

    # choose optimizer
    optimizer = create_optimizer(
                                model=model,
                                optimizer=optimizer,
                                lr=learning_rate
                                )

    criterion = build_loss(lossfunc, device=device)

    return model, optimizer, criterion

"""# Define an Accuracy Function"""

def accuracy_fn(y_true, y_pred):
  assert y_true.ndim == y_pred.ndim == 1 and len(y_true) == len(y_pred), 'The pred and true values must be 1D and of same length'
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

"""# Train the model"""

import sys

def trainModel(numepochs,
              input_features: int,
              output_features: int,
              hidden_layers: Sequence[int],
              *,
              activation_funcs: Union[nn.Module, Sequence[nn.Module]] = nn.ReLU(),
              dropout_probs: Optional[Union[float, Sequence[float]]] = None,
              batch_norm: Optional[Union[bool, Sequence[bool]]] = None,
              out_activation: Optional[nn.Module] = None,
              optimizer: Union[str, type] = "SGD",
              learning_rate: float = 1e-2,
              lossfunc: Union[str, nn.Module, type, None] = None,
              device: Optional[Union[str, torch.device]] = None,
          ):

  model, optimizer, criterion = createModel(
    input_features=input_features,
    output_features=output_features,
    hidden_layers=hidden_layers,
    activation_funcs=activation_funcs,
    dropout_probs=dropout_probs,
    batch_norm=batch_norm,
    out_activation=out_activation,
    optimizer=optimizer,
    learning_rate=learning_rate,
    lossfunc=lossfunc,
    device=device
    )

  trainloss, testloss, trainacc, testacc = [], [], [], []


  for epochi in range(numepochs):
    model.train()
    y_preds = model(X_train).squeeze()

    train_loss = criterion(y_preds, y_train)
    trainloss.append(train_loss.detach())

    train_acc = accuracy_fn(y_train,
                            torch.argmax(torch.softmax(y_preds,dim=1),dim=1))

    trainacc.append(train_acc)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


    model.eval()
    with torch.inference_mode():
      test_preds = model(X_test).squeeze()

    test_loss = criterion(test_preds, y_test)
    testloss.append(test_loss.detach())

    test_acc = accuracy_fn(y_test, torch.argmax(torch.softmax(test_preds,dim=1),dim=1))
    testacc.append(test_acc)

    # if epochi % 100 == 0:
    #   sys.stdout.write(f'\nEpoch: {epochi} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f} | Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}')

  return trainloss, testloss, trainacc, testacc, model

  ## Import CNN Model

  # Helper functions

def _as_list(value, n: int):
    """Normalize a scalar/tuple/sequence into a list of length n.
       - If value is a scalar or tuple, replicate it n times.
       - If value is a sequence of length 1, replicate it n times.
       - If value is a sequence of length n, use as is.
    """

    # tuples are valid for kernel/stride/padding/dilation
    if isinstance(value, (int, float, tuple)):
        return [value for _ in range(n)]
    if isinstance(value, Sequence):
        if len(value) == 1 and n > 1:
            return [value[0] for _ in range(n)]
        if len(value) == n:
            return list(value)
    raise ValueError(f"Could not broadcast parameter to length {n}: {value}")

# create a class for the model

def makeTheNet(out_channels: Sequence[int] = [6,6],         #conv1, conv2
                kernel_size: int | Tuple[int, int]=3,
                stride: int | Tuple[int, int]=1,
                padding: Optional[int | Tuple[int, int]]=[1,0],
                pool_size: Optional[Tuple[int, int]]=(2,2),
                dilation: int | Tuple[int, int]=1,
                groups: int =1,
                bias: bool =True,
                padding_mode: str='zeros',
                cv_dropout: float = 0.0,
                cv_batchnorm: bool = False,

                # Defintions for the FC layer
                uLinear: Optional[Sequence[int]] = [50],
                ln_batchnorm: bool = False,
                out_size: int = 26,
                ln_dropout: float = 0.0,

                # Regularization
                optimizer = 'Adam',
                weight_decay: float = 0.0,
                lr: float = 1e-3,

                device=None,
                dtype=None,
                printtoggle: bool = False):

  class cnnNet(nn.Module):
    def __init__(self,
                out_channels: Sequence[int] = out_channels,     #conv1, conv2
                kernel_size: int | Tuple[int, int]=kernel_size,
                stride: int | Tuple[int, int]=stride,
                padding: Optional[int | Tuple[int, int]]=padding,
                pool_size: Optional[Tuple[int, int]]=pool_size,
                dilation: int | Tuple[int, int]=dilation,
                groups: int =groups,
                bias: bool = bias,
                padding_mode: str=padding_mode,
                cv_dropout: float = cv_dropout,
                cv_batchnorm: bool = cv_batchnorm,

                # Defintions for the FC layer
                uLinear: Optional[Sequence[int]] = uLinear,
                ln_batchnorm: bool = ln_batchnorm,
                out_size: int = out_size,
                ln_dropout: float = ln_dropout,
                device=device,
                dtype=dtype,
                printtoggle: bool = printtoggle
                ):
      super().__init__()

      for i, oc in enumerate(out_channels):
          assert int(oc) > 0, f"out_channels[{i}] must be > 0, got {oc}"


      # print toggle
      self.printtoggle = printtoggle

      self.pool = nn.MaxPool2d(pool_size)

      self.cv_batchnorm = cv_batchnorm
      self.ln_batchnorm = ln_batchnorm


      nBlocks = len(out_channels)               # number of conv blocks

      # Normalize per-block parameters
      k_list = _as_list(kernel_size, nBlocks)
      s_list = _as_list(stride,      nBlocks)
      p_list = _as_list(padding,     nBlocks)
      d_list = _as_list(dilation,    nBlocks)
      pool_list = _as_list(pool_size, nBlocks)

      #--------Build Convoluation Blocks----------
      convs = []
      cbns = []                 # Batch normalization for each layer
      pools = []
      cdo = []                 # Dropout per convolution block (optional)


      # ---- Convolutional stack ----
      for i in range(nBlocks):
        convs.append(nn.LazyConv2d(
        out_channels=out_channels[i],
        kernel_size=k_list[i],
        stride=s_list[i],
        padding=p_list[i],
        dilation=d_list[i],
        groups=groups,
        bias=bias,
        padding_mode=padding_mode
    ))

        if self.cv_batchnorm:
          cbns.append(nn.BatchNorm2d(out_channels[i]))
        else:
          cbns.append(nn.Identity())

        pools.append(nn.MaxPool2d(kernel_size=pool_list[i]))

        if cv_dropout > 0:
          cdo.append(nn.Dropout2d(cv_dropout))
        else:
          cdo.append(nn.Identity())

      self.convs = nn.ModuleList(convs)
      self.cbns  = nn.ModuleList(cbns)
      self.pools = nn.ModuleList(pools)
      self.cdo   = nn.ModuleList(cdo)

      # Safety checks (helps catch the “zip returns nothing” bug)
      assert len(self.convs) > 0, "No conv blocks built"
      assert len(self.convs) == len(self.pools) == len(self.cbns) == len(self.cdo), (len(self.convs), len(self.pools), len(self.cbns), len(self.cdo))

      # --- Linear stack/ Multi-Layer Perceptorn (MLP) (variable depth) ---
      sizes = list(uLinear) if uLinear is not None else [50]
      fcs, fbns, fdo = [], [], []

      for h in sizes:
          fcs.append(nn.LazyLinear(h))
          if self.ln_batchnorm:
            fbns.append(nn.BatchNorm1d(h))
          else:
            fbns.append(nn.Identity())

          if ln_dropout > 0:
              fdo.append(nn.Dropout(ln_dropout))
          else:
              fdo.append(nn.Identity())


      self.fcs  = nn.ModuleList(fcs)
      self.fbns = nn.ModuleList(fbns)
      self.fdo  = nn.ModuleList(fdo)
      self.output_layer = nn.LazyLinear(out_size)

    def forward(self,x):

      if self.printtoggle: print(f'Input: {list(x.shape)}')

      # Convolutional stack: Conv -> (BN) -> Max_Pool -> LeakyReLU -> (Dropout2d)
      for i in range(len(self.convs)):

        x = self.convs[i](x)
        x = self.cbns[i](x)
        x = self.pools[i](x)
        x = F.leaky_relu(x, inplace=True)
        x = self.cdo[i](x)

        if self.printtoggle: print(f'After block {i+1}: {list(x.shape)}')
        if self.printtoggle: print(f'Block {i+1} padding {self.convs[i].padding}')

      # reshape for linear layer
      x = torch.flatten(x,start_dim=1)
      if self.printtoggle: print(f'Vectorized: {list(x.shape)}')

      ## FC stack
      for i in range(len(self.fcs)):

        x = self.fcs[i](x)
        x = self.fbns[i](x)
        x = F.leaky_relu(x, inplace=True)
        x = self.fdo[i](x)

        if self.printtoggle: print(f'After FC {i+1}: {list(x.shape)}')


      x = self.output_layer(x)

      return x

  #-------------------End of model factory------------------------------

  # create the model instance
  net = cnnNet()

  # Move to device/dtype once (constructor kwargs are version-fragile)
  if device is not None or dtype is not None:
      net = net.to(device=device, dtype=dtype)

  # loss function
  lossfun = nn.CrossEntropyLoss()

  # optimizer
  optimizer = getattr(torch.optim,optimizer)(net.parameters(),lr=lr,weight_decay=weight_decay)

  return net,lossfun,optimizer


### Function to train the CNN model
def function2trainTheModel(
    numepochs: int,
    test_dataset: DataLoader,
    *,
    out_channels: list[int] = [6, 6],
    kernel_size: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | list[tuple[int, int]] = [1, 0, 0],
    pool_size: tuple[int, int] = (2, 2),
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    cv_dropout: float = 0.0,
    cv_batchnorm: bool = False,
    uLinear: list[int] = [50],
    ln_batchnorm: bool = False,
    out_size: int = 26,
    ln_dropout: float = 0.0,
    optimizer: str = "Adam",          # name of optimizer (Adam, AdamW, SGD, ...)
    weight_decay: float = 0.0,
    lr: float = 1e-3,
    device: torch.device | None = None,
    dtype=None,
    printtoggle: bool = False,
):
    # ---------- Device selection (don’t rely on outer globals) ----------
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print("Using device:", device)

    opt_name = optimizer
    net, lossfun, opt = makeTheNet(
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pool_size=pool_size,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        cv_dropout=cv_dropout,
        cv_batchnorm=cv_batchnorm,
        uLinear=uLinear,
        ln_batchnorm=ln_batchnorm,
        out_size=out_size,
        ln_dropout=ln_dropout,
        optimizer=opt_name,
        weight_decay=weight_decay,
        lr=lr,
        device=device,   # safely pass device through
        dtype=dtype,
        printtoggle=printtoggle,
    )

    net.to(device)
    lossfun = lossfun.to(device)

    # ---------- AMP (new API); enable only on CUDA ----------
    use_cuda = (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=use_cuda)
    autocast_ctx = torch.autocast(device_type="cuda",dtype=torch.float16, enabled=use_cuda)

    # ---------- Metrics (avoid per-batch .item() → sync once/epoch) ----------
    trainLoss = torch.zeros(numepochs)
    testLoss  = torch.zeros(numepochs)
    trainErr  = torch.zeros(numepochs)
    testErr   = torch.zeros(numepochs)

    net.train()
    # ---------- Epoch loop ----------
    # for epochi in tqdm(range(numepochs), desc="Epoch", total=numepochs):
    for epochi in range(numepochs):
        # ---------- Batch loop ----------
        batch_loss_sum = torch.zeros((), device=device)
        err_count = torch.zeros((), device=device, dtype=torch.long)
        n_seen = 0

        # for X, y in tqdm(train_loader, desc="Batch", total=len(train_loader.dataset.data)/batch_size,leave=False):
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast_ctx:
                yHat = net(X)
                loss = lossfun(yHat, y)

            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            # accumulate on device (no CPU sync)
            with torch.no_grad():
                batch_loss_sum += loss.detach() * y.size(0)
                err_count += (yHat.argmax(dim=1) != y).sum()
                n_seen += y.size(0)

        # end train epoch: sync once
        trainLoss[epochi] = (batch_loss_sum / max(n_seen, 1)).detach().cpu()
        trainErr[epochi]  = (100.0 * err_count.float() / max(n_seen, 1)).detach().cpu()

        # ---------- Evaluation on full test loader (not a single batch) ----------
        net.eval()
        test_loss_sum = torch.zeros((), device=device)
        test_err_count = torch.zeros((), device=device, dtype=torch.long)
        test_seen = 0

        with torch.no_grad():
            for X, y in test_dataset:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                yHat = net(X)
                test_loss_sum += lossfun(yHat, y) * y.size(0)
                test_err_count += (yHat.argmax(dim=1) != y).sum()
                test_seen += y.size(0)

        testLoss[epochi] = (test_loss_sum / max(test_seen, 1)).detach().cpu()
        testErr[epochi]  = (100.0 * test_err_count.float() / max(test_seen, 1)).detach().cpu()

    # done
    return trainLoss, testLoss, trainErr, testErr, net





