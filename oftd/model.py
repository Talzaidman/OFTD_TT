
import math
import time
import torch
import numpy as np
from torch import nn
from torch import optim
try:
    from thop import profile
except ImportError:
    def profile(*args, **kwargs):
        return float("nan"), float("nan")
from utils import *
from affine import *
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                is_first=False, omega_0=1.5): 
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Online_CP_single_net_affine(nn.Module): 
    def __init__(self,n_1,n_2,R=100,mid_channel=256,omega_0=1.5):
        super(Online_CP_single_net_affine, self).__init__()
        self.A = nn.Parameter(torch.Tensor(R,n_1,1))
        self.B = nn.Parameter(torch.Tensor(R,1,n_2))

        self.C_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_0),
                                SineLayer(mid_channel, mid_channel, is_first=True, omega_0 = omega_0),
                                nn.Linear(mid_channel, R))
        
        self.x_net = nn.Sequential(SineLayer(R, R, omega_0=0.2, is_first=True),
                                       nn.Linear(R, 1, bias = False))
        
        self.y_net = nn.Sequential(SineLayer(R, R, omega_0=0.2, is_first=True),
                                       nn.Linear(R, 1, bias = False))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.A.size(0))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)
                                    
    def forward(self,C_input):
        x = torch.matmul(self.A, self.B).permute(1,2,0)
        C = self.C_net(C_input).permute(1,0)
        x1 = self.x_net(C.permute(1,0))
        y = self.y_net(C.permute(1,0))
        return x @ C, x1.squeeze(-1), y.squeeze(-1)
  

class Online_CP_single_net(nn.Module): 
    def __init__(self,n_1,n_2,R=100,mid_channel=256,omega_0=1.5):
        super(Online_CP_single_net, self).__init__()
        self.A = nn.Parameter(torch.Tensor(R,n_1,1))
        self.B = nn.Parameter(torch.Tensor(R,1,n_2))

        self.C_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_0),
                                SineLayer(mid_channel, mid_channel, is_first=True, omega_0 = omega_0),
                                nn.Linear(mid_channel, R))
        


        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.A.size(0))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)
                                    
    def forward(self,C_input):
        x = torch.matmul(self.A, self.B).permute(1,2,0)
        C = self.C_net(C_input).permute(1,0)
        return x @ C
    

class Online_FTD_net(nn.Module):
    """Theory model from optimization_project.pdf:
    X[i,j,k] = A[i]^T * B[j] * C[k] with B[j] in R^{R1 x R2}.
    """
    def __init__(
        self,
        R=100,
        R1=None,
        R2=None,
        mid_channel=256,
        omega_A=1.5,
        omega_B=1.5,
        omega_C=1.5,
        w_init=0.05,
    ):
        super(Online_FTD_net, self).__init__()
        self.R1 = R if R1 is None else R1
        self.R2 = R if R2 is None else R2

        self.A_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega_A),
            nn.Linear(mid_channel, self.R1),
        )
        self.B_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega_B),
            nn.Linear(mid_channel, self.R1 * self.R2),
        )
        self.C_net = nn.Sequential(
            SineLayer(1, mid_channel, is_first=True, omega_0=omega_C),
            SineLayer(mid_channel, mid_channel, is_first=True, omega_0=omega_C),
            nn.Linear(mid_channel, self.R2),
        )
        self._init_inr_weights(w_init)

    def _init_inr_weights(self, w_init):
        # matching the PDF assumption “weights i.i.d. N(0,w^2)”
        for module in [self.A_net, self.B_net, self.C_net]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=w_init)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, A_input, B_input, C_input):
        A = self.A_net(A_input)  # (i, R1)
        B = self.B_net(B_input).view(-1, self.R1, self.R2)  # (j, R1, R2)
        C = self.C_net(C_input)  # (k, R2)
        return torch.einsum("ir,jrs,ks->ijk", A, B, C)


def make_ftd_optimizer(
    model,
    lr=1e-3,
    weight_decay=1e-8,
    lr_a_mult=1.0,
    lr_b_mult=1.0,
    lr_c_mult=1.0,
):
    """Adam optimizer with separate TT-factor learning-rate controls."""
    grouped_param_ids = set()
    param_groups = []
    for name, module, mult in [
        ("A_net", getattr(model, "A_net", None), lr_a_mult),
        ("B_net", getattr(model, "B_net", None), lr_b_mult),
        ("C_net", getattr(model, "C_net", None), lr_c_mult),
    ]:
        if module is None:
            continue
        params = [p for p in module.parameters() if p.requires_grad]
        if params:
            grouped_param_ids.update(id(p) for p in params)
            param_groups.append(
                {
                    "params": params,
                    "lr": lr * mult,
                    "weight_decay": weight_decay,
                    "name": name,
                }
            )

    rest = [
        p
        for p in model.parameters()
        if p.requires_grad and id(p) not in grouped_param_ids
    ]
    if rest:
        param_groups.append(
            {
                "params": rest,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "other",
            }
        )

    return optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)


    
class Online_CP_multi_net(nn.Module): 
    def __init__(self,R1=100,R2=100,R3=100,mid_channel=256,omega_A=1.5,omega_B=1.5,omega_C=1.5):
        super(Online_CP_multi_net, self).__init__()
        if not (R1 == R2 == R3):
            raise ValueError("Online_CP_multi_net expects equal CP rank across modes (R1 == R2 == R3).")
        self.rank = R1

        self.A_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_A),
                                nn.Linear(mid_channel, R1))
        
        self.B_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_B),
                                nn.Linear(mid_channel, R2))

        self.C_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_C),
                                SineLayer(mid_channel, mid_channel, is_first=True, omega_0 = omega_C),
                                nn.Linear(mid_channel, R3))
        
    def forward(self, A_input, B_input, C_input):
        A = self.A_net(A_input)
        B = self.B_net(B_input)
        C = self.C_net(C_input)
        # Direct CP reconstruction: X[i,j,k] = sum_r A[i,r] * B[j,r] * C[k,r]
        return torch.einsum("ir,jr,kr->ijk", A, B, C)


def online_update_single_affine(aa, model, X_train, X_test, mask_train, 
                 mask_test, t, delta_t, divide, flops_all = None, every_iter = 100):    

    start = time.time()
    rotate_theta = torch.zeros((103), device=0, requires_grad=True)
    Scale_factor = torch.ones((103), device=0, requires_grad=True)
    x = torch.ones((103), device=0, requires_grad=True)
    y = torch.ones((103), device=0, requires_grad=True)
    params = []
    params += [rotate_theta]
    params += [Scale_factor]

    optimizier = optim.Adam([{'params':model.parameters(), 'lr':0.001, 'weight_decay': 10e-8},
                                        {'params':params, 'lr':0.002*0.001}])   

    t = t + delta_t
    if t > X_train.shape[2]:
        t = X_train.shape[2]

    X_t = X_train[:, :, :t]
    X_t_test = X_test[:, :, :t]

    mask_t_train = mask_train[:, :, :t]
    mask_t_test = mask_test[:, :, :t]
    
    C_input = torch.from_numpy(np.arange(t)).reshape(t, 1).type(dtype).to(device)
    loss_best = 1e10

    new_data_ind = np.arange(t-delta_t, t)
    
    for iter in range(every_iter):
        indexes = sample(aa, divide = divide, t = t, )
        ind = np.concatenate([indexes, new_data_ind], axis=0)

        mask_train_here = mask_t_train[:,:,ind]
        X_t_here = X_t[:,:,ind]
        optimizier.zero_grad()
        C_input_here = torch.from_numpy(ind).unsqueeze(-1).type(dtype).to(device)

        X_Out_real,x,y = model(C_input_here)

        rotate_theta1 = rotate_theta[ind]
        Scale_factor1 = Scale_factor[ind]

        x1= x
        y1 = y

        X_Out_real = X_Out_real.permute(2,0,1)
        X_Out_real = affine_B1(X_Out_real.unsqueeze(0), x1, y1,
                        rotate_theta1, Scale_factor1, 32).squeeze(0)
        X_Out_real = X_Out_real.permute(1,2,0)
        if iter == 0:
            flops_update, params = profile(model, inputs=(C_input_here,), verbose=False)
            flops_all.append(flops_update)
        loss = ((X_Out_real*mask_train_here - X_t_here*mask_train_here)**2).sum()
        if loss.item() < loss_best:
            loss_best = loss.item()
            best_params = model.state_dict()

        loss.backward() 
        optimizier.step()


    model.load_state_dict(best_params)

    X_Out_real,x,y = model(C_input)
    X_Out_real = X_Out_real.permute(2,0,1)
    c = C_input.detach().cpu().numpy().squeeze()
    x2 = x[c]
    y2 = y[c]
    rotate_theta2 = rotate_theta[c]
    Scale_factor2 = Scale_factor[c]
    X_Out_real = affine_B1(X_Out_real.unsqueeze(0), x2, y2,
                    rotate_theta2, Scale_factor2, 32).squeeze(0)
    X_Out_real = X_Out_real.permute(1,2,0)

    if torch.eq(mask_t_test[:, :, -1], 0).all():
        mask_t_test[:, :, -1] = (mask_t_test[:, :, -2] + mask_test[:, :, t]) / 2
        X_t_test[:, :, -1] = (X_t_test[:, :, -2] + X_test[:, :, t]) / 2
    
    nre = calcu_nre(X_t[:,:,:], X_Out_real[:,:,:], mask_t_train[:,:,:]).item()
    nre_test = calcu_nre(X_t_test[:,:,:], X_Out_real[:,:,:], mask_t_test[:,:,: ]).item()
    end = time.time()
    time_cost = end - start

    return model, t, time_cost, nre, nre_test


def online_update_single(aa, model, X_train, X_test, mask_train, 
                 mask_test, t, delta_t, divide, flops_all = None, every_iter = 100):    

    start = time.time()
    params = []
    params += [x for x in model.parameters()]
    optimizier = optim.Adam(params, lr=0.001, weight_decay=10e-8)

    t = t + delta_t
    if t > X_train.shape[2]:
        t = X_train.shape[2]

    X_t = X_train[:, :, :t]
    X_t_test = X_test[:, :, :t]

    mask_t_train = mask_train[:, :, :t]
    mask_t_test = mask_test[:, :, :t]
    
    C_input = torch.from_numpy(np.arange(t)).reshape(t, 1).type(dtype).to(device)
    loss_best = 1e10
    new_data_ind = np.arange(t-delta_t, t)
   
    indexes = sample(aa, divide = divide, t = t, )
    ind = np.concatenate([indexes, new_data_ind], axis=0)
    # To further reduce the computational overhead, fix the size of the memory Buffer (this is an option)
    # if len(indexes) > 100:
    #     indexes = np.random.choice(indexes, 100, replace=False)
    for iter in range(every_iter):

        mask_train_here = mask_t_train[:,:,ind]
        X_t_here = X_t[:,:,ind]
        optimizier.zero_grad()
        C_input_here = torch.from_numpy(ind).unsqueeze(-1).type(dtype).to(device)
        X_Out_real = model(C_input_here)
        if iter == 0:
            flops_update, params = profile(model, inputs=(C_input_here,), verbose=False)
            flops_all.append(flops_update)

        loss = ((X_Out_real*mask_train_here - X_t_here*mask_train_here)**2).sum()

        if loss.item() < loss_best:
            loss_best = loss.item()
            best_params = model.state_dict()

        loss.backward() 
        optimizier.step()


    model.load_state_dict(best_params)
    X_Out_real = model(C_input)

    if torch.eq(mask_t_test[:, :, -1], 0).all():
        mask_t_test[:, :, -1] = (mask_t_test[:, :, -2] + mask_test[:, :, t]) / 2
        X_t_test[:, :, -1] = (X_t_test[:, :, -2] + X_test[:, :, t]) / 2
    
    nre = calcu_nre(X_t[:,:,:], X_Out_real[:,:,:], mask_t_train[:,:,:]).item()
    nre_test = calcu_nre(X_t_test[:,:,:], X_Out_real[:,:,:], mask_t_test[:,:,:]).item()
    end = time.time()
    time_cost = end - start

    return model, t, time_cost, nre, nre_test


def online_update_multi(alpha_beta, model, X_train, X_test, mask_train, mask_test, 
             A_t, B_t, C_t, A_delta, B_delta, C_delta, divide, 
             flops_all = None, every_iter = 500):
    params = []
    params += [x for x in model.parameters()]
    optimizier = optim.Adam(params, lr=0.001, weight_decay=10e-8)

    A_t = A_t + A_delta
    if A_t > X_train.shape[0]:
        A_t = X_train.shape[0]
    B_t = B_t + B_delta
    if B_t > X_train.shape[1]:
        B_t = X_train.shape[1]
    C_t = C_t + C_delta
    if C_t > X_train.shape[2]:
        C_t = X_train.shape[2]

    X_t = X_train[:A_t, :B_t, :C_t]
    X_t_test = X_test[:A_t, :B_t, :C_t]

    mask_t_train = mask_train[:A_t, :B_t, :C_t]
    mask_t_test = mask_test[:A_t, :B_t, :C_t]
    
    A_input = torch.from_numpy(np.arange(A_t)).reshape(A_t, 1).type(dtype).to(device)
    B_input = torch.from_numpy(np.arange(B_t)).reshape(B_t, 1).type(dtype).to(device)
    C_input = torch.from_numpy(np.arange(C_t)).reshape(C_t, 1).type(dtype).to(device)
    new_data_ind_A = np.arange(A_t-A_delta, A_t)
    new_data_ind_B = np.arange(B_t-B_delta, B_t)
    new_data_ind_C = np.arange(C_t-C_delta, C_t)

    loss_best = 1e10
    start_time = time.perf_counter()
    for iter in range(every_iter):
        indexes_A = sample(alpha_beta, divide = divide, t = A_t)
        indexes_B = sample(alpha_beta, divide = divide, t = B_t)
        indexes_C = sample(alpha_beta, divide = divide, t = C_t)

        ind_A = np.concatenate([indexes_A, new_data_ind_A], axis=0)
        ind_B = np.concatenate([indexes_B, new_data_ind_B], axis=0)
        ind_C = np.concatenate([indexes_C, new_data_ind_C], axis=0)

        mask_train_here = mask_t_train[:,:,ind_C]
        mask_train_here = mask_train_here[:,ind_B,:]
        mask_train_here = mask_train_here[ind_A,:,:]
        X_t_here = X_t[:,:,ind_C]
        X_t_here = X_t_here[:,ind_B,:]
        X_t_here = X_t_here[ind_A,:,:]

        optimizier.zero_grad()

        A_input_here = torch.from_numpy(ind_A).unsqueeze(-1).type(dtype).to(device)
        B_input_here = torch.from_numpy(ind_B).unsqueeze(-1).type(dtype).to(device)
        C_input_here = torch.from_numpy(ind_C).unsqueeze(-1).type(dtype).to(device)
        X_Out_real = model(A_input_here, B_input_here, C_input_here)
        if iter == 0:
            flops_update, params = profile(model, inputs=(A_input_here,B_input_here,C_input_here,), verbose=False)
            flops_all.append(flops_update)

        loss = ((X_Out_real*mask_train_here - X_t_here*mask_train_here)**2).sum()
        if loss.item() < loss_best:
            loss_best = loss.item()
            best_params = model.state_dict()
        loss.backward()
        optimizier.step()

    time_cost = time.perf_counter() - start_time
    model.load_state_dict(best_params)
    X_Out_real = model(A_input,B_input,C_input)

    if torch.eq(mask_t_test[:, :, -1], 0).all():
        mask_t_test[:, :, -1] = mask_t_test[:, :, -2]
        X_t_test[:, :, -1] = X_t_test[:, :, -2]
        
    nre = calcu_nre(X_t[:A_t,:B_t,:C_t], X_Out_real[:A_t,:B_t,:C_t], mask_t_train[:A_t,:B_t,:C_t]).item()
    nre_test = calcu_nre(X_t_test[:A_t,:B_t,:C_t], X_Out_real[:A_t,:B_t,:C_t], mask_t_test[:A_t,:B_t,:C_t]).item()
    
    return model, A_t, B_t, C_t, time_cost, nre, nre_test


def check_ftd_theory_alignment(model, A_t, B_t, C_t, delta=1.0, kappa=None):
    """Diagnostic metrics for FTD INR assumptions."""
    def _linear_stats(module):
        weights = []
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.detach().reshape(-1))
        if not weights:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        vec = torch.cat(weights)
        return {
            "mean": vec.mean().item(),
            "std": vec.std(unbiased=False).item(),
            "min": vec.min().item(),
            "max": vec.max().item(),
        }

    with torch.no_grad():
        A_coords = torch.arange(A_t, dtype=dtype, device=device).reshape(-1, 1)
        B_coords = torch.arange(B_t, dtype=dtype, device=device).reshape(-1, 1)
        C_coords = torch.arange(C_t, dtype=dtype, device=device).reshape(-1, 1)

        dA = (model.A_net(A_coords + delta) - model.A_net(A_coords)).abs().sum(dim=1) / max(delta, 1e-12)
        dB = (model.B_net(B_coords + delta) - model.B_net(B_coords)).abs().sum(dim=1) / max(delta, 1e-12)
        dC = (model.C_net(C_coords + delta) - model.C_net(C_coords)).abs().sum(dim=1) / max(delta, 1e-12)

    out = {
        "A_weight_stats": _linear_stats(model.A_net),
        "B_weight_stats": _linear_stats(model.B_net),
        "C_weight_stats": _linear_stats(model.C_net),
        "dA_l1_mean": dA.mean().item(),
        "dA_l1_max": dA.max().item(),
        "dB_l1_mean": dB.mean().item(),
        "dB_l1_max": dB.max().item(),
        "dC_l1_mean": dC.mean().item(),
        "dC_l1_max": dC.max().item(),
    }
    if kappa is not None:
        out["kappa"] = float(kappa)
        out["dA_within_kappa"] = bool((dA <= kappa).all().item())
        out["dB_within_kappa"] = bool((dB <= kappa).all().item())
        out["dC_within_kappa"] = bool((dC <= kappa).all().item())
    return out


def online_update_multi_ftd(alpha_beta, model, X_train, X_test, mask_train, mask_test,
             A_t, B_t, C_t, A_delta, B_delta, C_delta, divide,
             flops_all=None, every_iter=500, lr=1e-3, weight_decay=1e-8, boundary_lambda=0.0,
             deriv_lambda=0.0, kappa=-1.0, normalize_recon=True,
             coord_mode="raw", loss_scope="sampled", profile_flops=False,
             lr_a_mult=1.0, lr_b_mult=1.0, lr_c_mult=1.0, clip_grad_norm=1.0,
             optimizer=None, return_optimizer=False):
    """Online update for Online_FTD_net with optional theory regularizers."""
    if optimizer is None:
        optimizer = make_ftd_optimizer(
            model,
            lr=lr,
            weight_decay=weight_decay,
            lr_a_mult=lr_a_mult,
            lr_b_mult=lr_b_mult,
            lr_c_mult=lr_c_mult,
        )

    prev_A_t, prev_B_t, prev_C_t = A_t, B_t, C_t
    A_t = min(A_t + A_delta, X_train.shape[0])
    B_t = min(B_t + B_delta, X_train.shape[1])
    C_t = min(C_t + C_delta, X_train.shape[2])

    X_t = X_train[:A_t, :B_t, :C_t]
    X_t_test = X_test[:A_t, :B_t, :C_t]
    mask_t_train = mask_train[:A_t, :B_t, :C_t]
    mask_t_test = mask_test[:A_t, :B_t, :C_t]

    def _coord_transform(vals, size):
        if coord_mode == "raw":
            return vals
        denom = max(size - 1, 1)
        if coord_mode == "zero_one":
            return vals / denom
        if coord_mode == "minus_one_one":
            return 2.0 * (vals / denom) - 1.0
        raise ValueError(f"Unsupported coord_mode: {coord_mode}")

    A_input = _coord_transform(torch.arange(A_t, dtype=dtype, device=device), A_t).reshape(A_t, 1)
    B_input = _coord_transform(torch.arange(B_t, dtype=dtype, device=device), B_t).reshape(B_t, 1)
    C_input = _coord_transform(torch.arange(C_t, dtype=dtype, device=device), C_t).reshape(C_t, 1)
    new_data_ind_A = np.arange(max(A_t - A_delta, 0), A_t)
    new_data_ind_B = np.arange(max(B_t - B_delta, 0), B_t)
    new_data_ind_C = np.arange(max(C_t - C_delta, 0), C_t)

    boundary_A_coord = (
        _coord_transform(torch.tensor([prev_A_t - 1], dtype=dtype, device=device), prev_A_t).reshape(1, 1)
        if prev_A_t > 0 else None
    )
    boundary_B_coord = (
        _coord_transform(torch.tensor([prev_B_t - 1], dtype=dtype, device=device), prev_B_t).reshape(1, 1)
        if prev_B_t > 0 else None
    )
    boundary_C_coord = (
        _coord_transform(torch.tensor([prev_C_t - 1], dtype=dtype, device=device), prev_C_t).reshape(1, 1)
        if prev_C_t > 0 else None
    )

    boundary_targets = None
    if boundary_lambda > 0 and boundary_A_coord is not None and boundary_B_coord is not None and boundary_C_coord is not None:
        with torch.no_grad():
            boundary_targets = (
                model.A_net(boundary_A_coord).detach(),
                model.B_net(boundary_B_coord).detach(),
                model.C_net(boundary_C_coord).detach(),
            )

    loss_best = 1e10
    best_params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    boundary_rel_error = float("nan")

    def _choose_indices(t_cur, t_delta, replay_idx, new_idx):
        # Single-aspect streams keep fixed dimensions fully observed.
        if t_delta == 0:
            return np.arange(t_cur, dtype=int)
        return np.concatenate([replay_idx, new_idx], axis=0)

    start_time = time.perf_counter()
    for iter in range(every_iter):
        indexes_A = sample(alpha_beta, divide=divide, t=A_t)
        indexes_B = sample(alpha_beta, divide=divide, t=B_t)
        indexes_C = sample(alpha_beta, divide=divide, t=C_t)

        ind_A = _choose_indices(A_t, A_delta, indexes_A, new_data_ind_A)
        ind_B = _choose_indices(B_t, B_delta, indexes_B, new_data_ind_B)
        ind_C = _choose_indices(C_t, C_delta, indexes_C, new_data_ind_C)

        mask_train_here = mask_t_train[:, :, ind_C]
        mask_train_here = mask_train_here[:, ind_B, :]
        mask_train_here = mask_train_here[ind_A, :, :]
        X_t_here = X_t[:, :, ind_C]
        X_t_here = X_t_here[:, ind_B, :]
        X_t_here = X_t_here[ind_A, :, :]

        optimizer.zero_grad()

        A_input_here = _coord_transform(torch.from_numpy(ind_A).type(dtype).to(device), A_t).unsqueeze(-1)
        B_input_here = _coord_transform(torch.from_numpy(ind_B).type(dtype).to(device), B_t).unsqueeze(-1)
        C_input_here = _coord_transform(torch.from_numpy(ind_C).type(dtype).to(device), C_t).unsqueeze(-1)

        X_Out_real = model(A_input_here, B_input_here, C_input_here)

        if iter == 0 and profile_flops:
            flops_update, params = profile(model, inputs=(A_input_here, B_input_here, C_input_here,), verbose=False)
            if flops_all is not None:
                flops_all.append(flops_update)

        if loss_scope == "full":
            X_pred_full = model(A_input, B_input, C_input)
            sq_err = ((X_pred_full * mask_t_train - X_t * mask_t_train) ** 2).sum()
            obs_count = mask_t_train.sum()
        elif loss_scope == "sampled":
            sq_err = ((X_Out_real * mask_train_here - X_t_here * mask_train_here) ** 2).sum()
            obs_count = mask_train_here.sum()
        else:
            raise ValueError(f"Unsupported loss_scope: {loss_scope}")

        if normalize_recon:
            recon_loss = sq_err / obs_count.clamp_min(1.0)
        else:
            recon_loss = sq_err

        boundary_loss = torch.tensor(0.0, device=device)
        if boundary_targets is not None:
            cur_A = model.A_net(boundary_A_coord)
            cur_B = model.B_net(boundary_B_coord)
            cur_C = model.C_net(boundary_C_coord)
            boundary_loss = ((cur_A - boundary_targets[0]) ** 2).sum() / ((boundary_targets[0] ** 2).sum() + 1e-12)
            boundary_loss = boundary_loss + ((cur_B - boundary_targets[1]) ** 2).sum() / ((boundary_targets[1] ** 2).sum() + 1e-12)
            boundary_loss = boundary_loss + ((cur_C - boundary_targets[2]) ** 2).sum() / ((boundary_targets[2] ** 2).sum() + 1e-12)
            boundary_loss = boundary_loss / 3.0

        deriv_loss = torch.tensor(0.0, device=device)
        if deriv_lambda > 0:
            dA = (model.A_net(A_input_here + 1.0) - model.A_net(A_input_here)).abs().sum(dim=1)
            dB = (model.B_net(B_input_here + 1.0) - model.B_net(B_input_here)).abs().sum(dim=1)
            dC = (model.C_net(C_input_here + 1.0) - model.C_net(C_input_here)).abs().sum(dim=1)
            if kappa > 0:
                deriv_loss = torch.relu(dA - kappa).mean()
                deriv_loss = deriv_loss + torch.relu(dB - kappa).mean()
                deriv_loss = deriv_loss + torch.relu(dC - kappa).mean()
            else:
                deriv_loss = dA.mean() + dB.mean() + dC.mean()
            deriv_loss = deriv_loss / 3.0

        loss = recon_loss + boundary_lambda * boundary_loss + deriv_lambda * deriv_loss
        loss_value = loss.item()
        if math.isfinite(loss_value) and loss_value < loss_best:
            loss_best = loss_value
            best_params = {k: v.detach().clone() for k, v in model.state_dict().items()}
        loss.backward()
        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()

    time_cost = time.perf_counter() - start_time
    model.load_state_dict(best_params, strict=False)
    X_Out_real = model(A_input, B_input, C_input)

    if boundary_targets is not None:
        with torch.no_grad():
            final_A = model.A_net(boundary_A_coord)
            final_B = model.B_net(boundary_B_coord)
            final_C = model.C_net(boundary_C_coord)
            relA = torch.norm(final_A - boundary_targets[0]) / (torch.norm(boundary_targets[0]) + 1e-12)
            relB = torch.norm(final_B - boundary_targets[1]) / (torch.norm(boundary_targets[1]) + 1e-12)
            relC = torch.norm(final_C - boundary_targets[2]) / (torch.norm(boundary_targets[2]) + 1e-12)
            boundary_rel_error = ((relA + relB + relC) / 3.0).item()

    if torch.eq(mask_t_test[:, :, -1], 0).all():
        mask_t_test[:, :, -1] = mask_t_test[:, :, -2]
        X_t_test[:, :, -1] = X_t_test[:, :, -2]

    nre = calcu_nre(X_t[:A_t, :B_t, :C_t], X_Out_real[:A_t, :B_t, :C_t], mask_t_train[:A_t, :B_t, :C_t]).item()
    nre_test = calcu_nre(X_t_test[:A_t, :B_t, :C_t], X_Out_real[:A_t, :B_t, :C_t], mask_t_test[:A_t, :B_t, :C_t]).item()

    result = (model, A_t, B_t, C_t, time_cost, nre, nre_test, boundary_rel_error)
    if return_optimizer:
        return (*result, optimizer)
    return result
