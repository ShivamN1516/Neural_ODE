#! /home/linot/anaconda3/bin/python3
import sys
import time
import argparse
import pickle
import numpy as np
import matplotlib as pl
import matplotlib.pyplot as plt
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser("IRMAE + Neural ODE on delay-embedded data")

parser.add_argument('--mat_file', type=str, default='depth_80 cm_Amp_10 deg_k_0.2.mat')
parser.add_argument('--data_key', type=str, default='L')
parser.add_argument('--train_frac', type=float, default=0.8)

# Delay embedding / preprocessing

parser.add_argument('--n_delays', type=int, default=1000)
parser.add_argument('--tau', type=int, default=1)
parser.add_argument('--skip', type=int, default=50)
parser.add_argument('--r_pca', type=int, default=100)

# IRMAE arguments

parser.add_argument('--irmae_latent_dim', type=int, default=100)
parser.add_argument('--irmae_iters', type=int, default=10)
parser.add_argument('--irmae_batch_size', type=int, default=64)
parser.add_argument('--irmae_lr', type=float, default=1e-3)
parser.add_argument('--irmae_step_size', type=int, default=500)
parser.add_argument('--irmae_gamma', type=float, default=0.1)
parser.add_argument('--irmae_weight_decay_lin', type=float, default=1e-6)


# Neural ODE arguments

parser.add_argument('--ode_hidden_dim', type=int, default=128)
parser.add_argument('--ode_batch_time', type=int, default=9)     # excluding IC
parser.add_argument('--ode_batch_size', type=int, default=20)
parser.add_argument('--ode_niters', type=int, default=1000)
parser.add_argument('--ode_test_freq', type=int, default=10)
parser.add_argument('--ode_lr', type=float, default=1e-3)
parser.add_argument('--ode_step_size', type=int, default=500)
parser.add_argument('--ode_gamma', type=float, default=0.1)
parser.add_argument('--ode_weight_decay_lin', type=float, default=1e-6)


# ODE solver arguments

parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--adjoint', action='store_true')


# Device

parser.add_argument('--gpu', type=int, default=0)

args = parser.parse_args()

# Include IC in rollout window
args.ode_batch_time += 1

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# DEVICE / DTYPE

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

print(f"Using device: {device}")


# HELPER CLASSES / FUNCTIONS

class RunningAverageMeter(object):
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0.0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1.0 - self.momentum)
        self.val = val


def output(text, filename="Out.txt"):
    with open(filename, "a+") as f:
        f.write(text)


def delay_embed(data: np.ndarray, n_delays: int, tau: int) -> np.ndarray:

    Nt, Nx = data.shape
    embed_length = Nt - (n_delays - 1) * tau

    if embed_length <= 0:
        raise ValueError("embed_length <= 0; reduce n_delays or tau")

    delayed_list = []
    for k in range(n_delays):
        start = k * tau
        end = start + embed_length
        delayed_list.append(data[start:end, :])

    return np.concatenate(delayed_list, axis=1)


def get_batch(time_array_torch, true_y_torch, batch_time, batch_size):

    total_length = len(time_array_torch)
    max_start = total_length - batch_time

    if max_start <= 0:
        raise ValueError("Not enough points for chosen ode_batch_time.")

    actual_batch_size = min(batch_size, max_start)

    s = torch.from_numpy(
        np.random.choice(np.arange(max_start, dtype=np.int64), actual_batch_size, replace=False)
    ).to(device)

    batch_y0 = true_y_torch[s]  # (M, D)
    batch_t = time_array_torch[:batch_time]  # (T,)
    batch_y = torch.stack([true_y_torch[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)

    return batch_y0, batch_t, batch_y


def plot_curve(x, y, xlabel, ylabel, title, filename, semilogy=False):
    plt.figure()
    if semilogy:
        plt.semilogy(x, y, ".-")
    else:
        plt.plot(x, y, ".-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_3d_curve(data_3d, title, filename):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_3d_overlay(data1, data2, title, filename, label1="Input", label2="Reconstructed"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data1[:, 0], data1[:, 1], data1[:, 2], label=label1)
    ax.plot(data2[:, 0], data2[:, 1], data2[:, 2], '--', label=label2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# IRMAE MODEL

class IRMAE(nn.Module):
    def __init__(self, trunc: int, N: int):
        super(IRMAE, self).__init__()

        self.encode = nn.Sequential(
            nn.Linear(N, 1000),
            nn.ReLU(),
            nn.Linear(1000, trunc),
        )

        self.decode = nn.Sequential(
            nn.Linear(trunc, 1000),
            nn.ReLU(),
            nn.Linear(1000, N),
        )

        self.lin = nn.Sequential(
            nn.Linear(trunc, trunc, bias=False),
            nn.Linear(trunc, trunc, bias=False),
            nn.Linear(trunc, trunc, bias=False),
            nn.Linear(trunc, trunc, bias=False),
        )

    def forward(self, y):
        z = self.encode(y)
        z = self.lin(z)
        yhat = self.decode(z)
        return yhat

    def encode_only(self, y):
        return self.encode(y)

    def latent_after_lin(self, y):
        z = self.encode(y)
        z = self.lin(z)
        return z



# NEURAL ODE IN IRMAE LATENT SPACE

class ODEFunc(nn.Module):

    def __init__(self, dim, hidden_dim=128):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

        self.lin = nn.Linear(dim, dim, bias=False)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

        nn.init.xavier_normal_(self.lin.weight)

    def forward(self, t, y):
        return self.lin(y) + self.net(y)



# MAIN

if __name__ == "__main__":

    # LOAD DATA

    mat = sio.loadmat(args.mat_file)
    X = mat[args.data_key]

    Nt, Nx = X.shape
    print(f"Loaded data shape: X = {X.shape}")
    print(f"Nt = {Nt}, Nx = {Nx}")

    # Global normalization on raw signal
    X = (X - np.mean(X)) / np.std(X)

    # TRAIN / TEST SPLIT ON RAW DATA

    frac = args.train_frac
    Nt_train = int(frac * Nt)

    X_train_raw = X[:Nt_train:10, :]
    X_test_raw  = X[Nt_train::10, :]

    print("\nRAW SPLIT SHAPES")
    print(f"X_train_raw shape = {X_train_raw.shape}")
    print(f"X_test_raw  shape = {X_test_raw.shape}")


    # DELAY EMBEDDING

    n_delays = args.n_delays
    tau = args.tau

    X_train_embed = delay_embed(X_train_raw, n_delays=n_delays, tau=tau)
    X_test_embed  = delay_embed(X_test_raw,  n_delays=n_delays, tau=tau)

    print("\nAFTER DELAY EMBEDDING")
    print(f"X_train_embed shape = {X_train_embed.shape}")
    print(f"X_test_embed  shape = {X_test_embed.shape}")

    # SKIP AFTER EMBEDDING

    skip = args.skip
    X_train = X_train_embed[::skip, :]
    X_test  = X_test_embed[::skip, :]

    print("\nAFTER SKIPPING EMBEDDED SNAPSHOTS")
    print(f"skip = {skip}")
    print(f"X_train shape = {X_train.shape}")
    print(f"X_test  shape = {X_test.shape}")

    # STANDARDIZING USING TRAIN STATISTICS ONLY

    Xmean = np.mean(X_train, axis=0, keepdims=True)
    Xstd = np.std(X_train, axis=0, keepdims=True)
    Xstd[Xstd < 1e-12] = 1.0

    X_train_std = (X_train - Xmean) / Xstd
    X_test_std  = (X_test  - Xmean) / Xstd

    print("\nAFTER STANDARDIZATION")
    print(f"X_train_std shape = {X_train_std.shape}")
    print(f"X_test_std  shape = {X_test_std.shape}")

    # PCA ON TRAIN ONLY

    print("\nCOMPUTING PCA BASIS ON X_train_std ...")
    U, S_pca, Vt = np.linalg.svd(X_train_std.T, full_matrices=False)

    r_pca = min(args.r_pca, U.shape[1])
    U_pca = U[:, :r_pca]

    XPCA_train = X_train_std @ U_pca
    XPCA_test  = X_test_std  @ U_pca

    print("\nAFTER PCA PROJECTION")
    print(f"U shape          = {U.shape}")
    print(f"U_pca shape      = {U_pca.shape}")
    print(f"XPCA_train shape = {XPCA_train.shape}")
    print(f"XPCA_test shape  = {XPCA_test.shape}")


    # TORCH DATA FOR IRMAE

    true_y_pca = torch.tensor(XPCA_train, dtype=torch.double, device=device)
    test_y_pca = torch.tensor(XPCA_test, dtype=torch.double, device=device)

    loader = DataLoader(
        true_y_pca,
        batch_size=args.irmae_batch_size,
        shuffle=True,
        drop_last=False
    )


    # TRAIN IRMAE

    print("\n==================== TRAINING IRMAE ====================")

    irmae = IRMAE(trunc=args.irmae_latent_dim, N=r_pca).double().to(device)

    optimizer_irmae = optim.AdamW(
        [
            {"params": irmae.encode.parameters()},
            {"params": irmae.decode.parameters()},
            {"params": irmae.lin.parameters(), "weight_decay": args.irmae_weight_decay_lin},
        ],
        lr=args.irmae_lr
    )

    scheduler_irmae = StepLR(
        optimizer_irmae,
        step_size=args.irmae_step_size,
        gamma=args.irmae_gamma
    )

    irmae_train_err = []
    irmae_test_err = []
    latent_singular_values = []

    freq_latent = max(1, args.irmae_iters // 10)

    for itr in range(1, args.irmae_iters + 1):
        irmae.train()
        epoch_loss = 0.0
        nbatches = 0

        for batch in loader:
            optimizer_irmae.zero_grad()
            recon = irmae(batch)
            loss = torch.mean((recon - batch) ** 2)
            loss.backward()
            optimizer_irmae.step()

            epoch_loss += loss.item()
            nbatches += 1

        epoch_loss /= max(1, nbatches)
        scheduler_irmae.step()

        irmae.eval()
        with torch.no_grad():
            recon_test = irmae(test_y_pca)
            test_loss = torch.mean((recon_test - test_y_pca) ** 2).item()

        irmae_train_err.append(epoch_loss)
        irmae_test_err.append(test_loss)

        if itr % freq_latent == 0:
            with torch.no_grad():
                Z = irmae.latent_after_lin(true_y_pca).detach().cpu().numpy()
                Z_mean = np.mean(Z, axis=0, keepdims=True)
                Zc = Z - Z_mean
                C = (Zc.T @ Zc) / Zc.shape[0]
                _, Stemp, _ = np.linalg.svd(C)
                latent_singular_values.append(Stemp)

        print(f"IRMAE Iter {itr:5d} | Train Loss = {epoch_loss:.3e} | Test Loss = {test_loss:.3e}")


    # SAVE IRMAE TRAINING CURVE

    plt.figure()
    plt.semilogy(np.arange(1, args.irmae_iters + 1), np.asarray(irmae_train_err), ".-", label="Train")
    plt.semilogy(np.arange(1, args.irmae_iters + 1), np.asarray(irmae_test_err), ".-", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("IRMAE Training Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("irmae_Mse_epoch.png", dpi=300, bbox_inches="tight")
    plt.close()

    if len(latent_singular_values) > 0:
        col = np.flip(pl.cm.autumn(np.linspace(0, 1, len(latent_singular_values))), axis=0)
        plt.figure()
        for i in range(len(latent_singular_values)):
            plt.semilogy(
                latent_singular_values[i] / latent_singular_values[i][0],
                ".-",
                color=col[i]
            )
        plt.xlabel("i")
        plt.ylabel("SV / SV[0]")
        plt.ylim([1e-20, 10])
        plt.title("IRMAE Latent Singular Value Evolution")
        plt.grid(True)
        plt.savefig("irmae_latent_sValues.png", dpi=300, bbox_inches="tight")
        plt.close()


    # EXTRACT IRMAE LATENT TRAJECTORIES

    irmae.eval()
    with torch.no_grad():
        Z_train = irmae.latent_after_lin(true_y_pca).detach().cpu().numpy()
        Z_test  = irmae.latent_after_lin(test_y_pca).detach().cpu().numpy()

        XPCAhat_train_irmae = irmae(true_y_pca).detach().cpu().numpy()
        XPCAhat_test_irmae  = irmae(test_y_pca).detach().cpu().numpy()

    print("\nLATENT SHAPES")
    print(f"Z_train shape = {Z_train.shape}")
    print(f"Z_test  shape = {Z_test.shape}")


    # BUILD TIME ARRAYS FOR NEURAL ODE IN LATENT SPACE

    dt_effective = float(skip * tau)

    t_train = np.arange(Z_train.shape[0], dtype=np.float64) * dt_effective
    t_test  = np.arange(Z_test.shape[0], dtype=np.float64) * dt_effective

    true_z = torch.tensor(Z_train, dtype=torch.double, device=device)
    test_z = torch.tensor(Z_test, dtype=torch.double, device=device)
    t_train_torch = torch.tensor(t_train, dtype=torch.double, device=device)
    t_test_torch  = torch.tensor(t_test, dtype=torch.double, device=device)


    # TRAIN NEURAL ODE ON IRMAE LATENT TRAJECTORY

    print("\n==================== TRAINING NEURAL ODE ====================")

    odefunc = ODEFunc(dim=args.irmae_latent_dim, hidden_dim=args.ode_hidden_dim).double().to(device)

    optimizer_ode = optim.AdamW(
        [
            {"params": odefunc.net.parameters()},
            {"params": odefunc.lin.parameters(), "weight_decay": args.ode_weight_decay_lin},
        ],
        lr=args.ode_lr
    )

    scheduler_ode = StepLR(
        optimizer_ode,
        step_size=args.ode_step_size,
        gamma=args.ode_gamma
    )

    ode_train_err = []
    ode_test_err = []

    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    end = time.time()

    for itr in range(1, args.ode_niters + 1):
        odefunc.train()
        optimizer_ode.zero_grad()

        batch_y0, batch_t, batch_y = get_batch(
            t_train_torch, true_z, args.ode_batch_time, args.ode_batch_size
        )
        
        print(batch_t)

        pred_y = odeint(odefunc, batch_y0, batch_t, method=args.method)
        loss = torch.mean((pred_y - batch_y) ** 2)

        loss.backward()
        optimizer_ode.step()
        scheduler_ode.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.ode_test_freq == 0:
            odefunc.eval()
            with torch.no_grad():
                batch_y0_test, batch_t_test, batch_y_test = get_batch(
                    t_test_torch, test_z, args.ode_batch_time, args.ode_batch_size
                )
                pred_y_test = odeint(odefunc, batch_y0_test, batch_t_test, method=args.method)
                test_loss = torch.mean((pred_y_test - batch_y_test) ** 2)

                ode_train_err.append(loss.item())
                ode_test_err.append(test_loss.item())

                msg = (
                    f"ODE Iter {itr:05d} | "
                    f"Train Loss {loss.item():.6e} | "
                    f"Val Loss {test_loss.item():.6e} | "
                    f"Time {time.time() - end:.6f}\n"
                )
                print(msg.strip())
                output(msg)

        end = time.time()


    # SAVE ODE TRAINING CURVE

    plt.figure()
    plt.semilogy(
        np.arange(args.ode_test_freq, args.ode_niters + 1, args.ode_test_freq),
        np.asarray(ode_train_err),
        ".-",
        label="Train"
    )
    plt.semilogy(
        np.arange(args.ode_test_freq, args.ode_niters + 1, args.ode_test_freq),
        np.asarray(ode_test_err),
        ".-",
        label="Test"
    )
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Neural ODE Training Error in IRMAE Latent Space")
    plt.legend()
    plt.grid(True)
    plt.savefig("latent_ode_Mse_epoch.png", dpi=300, bbox_inches="tight")
    plt.close()


    # FULL ROLLOUT IN LATENT SPACE

    odefunc.eval()
    with torch.no_grad():
        # Train rollout
        z0_train = true_z[0:1, :]
        Zhat_train_torch = odeint(odefunc, z0_train, t_train_torch, method=args.method)
        Zhat_train = Zhat_train_torch[:, 0, :].detach().cpu().numpy()

        # Test rollout
        z0_test = test_z[0:1, :]
        Zhat_test_torch = odeint(odefunc, z0_test, t_test_torch, method=args.method)
        Zhat_test = Zhat_test_torch[:, 0, :].detach().cpu().numpy()

    print("\nROLLED LATENT SHAPES")
    print(f"Z_train shape = {Z_train.shape}")
    print(f"Zhat_train shape = {Zhat_train.shape}")
    print(f"Z_test shape = {Z_test.shape}")
    print(f"Zhat_test shape = {Zhat_test.shape}")


    # DECODE LATENT ODE TRAJECTORY BACK TO PCA SPACE

    with torch.no_grad():
        Zhat_train_t = torch.tensor(Zhat_train, dtype=torch.double, device=device)
        Zhat_test_t  = torch.tensor(Zhat_test, dtype=torch.double, device=device)

        XPCAhat_train = irmae.decode(Zhat_train_t).detach().cpu().numpy()
        XPCAhat_test  = irmae.decode(Zhat_test_t).detach().cpu().numpy()

    print("\nRECONSTRUCTION SHAPES IN PCA SPACE")
    print(f"XPCA_train shape    = {XPCA_train.shape}")
    print(f"XPCAhat_train shape = {XPCAhat_train.shape}")
    print(f"XPCA_test shape     = {XPCA_test.shape}")
    print(f"XPCAhat_test shape  = {XPCAhat_test.shape}")


    # BACK TO ORIGINAL DELAY SPACE

    Xhat_train_std = XPCAhat_train @ U_pca.T
    Xhat_test_std  = XPCAhat_test  @ U_pca.T

    Xin_train_std_from_pca = XPCA_train @ U_pca.T
    Xin_test_std_from_pca  = XPCA_test  @ U_pca.T

    Xin_train = Xin_train_std_from_pca * Xstd + Xmean
    Xin_test  = Xin_test_std_from_pca  * Xstd + Xmean

    Xhat_train = Xhat_train_std * Xstd + Xmean
    Xhat_test  = Xhat_test_std  * Xstd + Xmean

    print("\nRECONSTRUCTION SHAPES IN ORIGINAL DELAY SPACE")
    print(f"Xin_train shape   = {Xin_train.shape}")
    print(f"Xhat_train shape  = {Xhat_train.shape}")
    print(f"Xin_test shape    = {Xin_test.shape}")
    print(f"Xhat_test shape   = {Xhat_test.shape}")


    # PCA VISUALIZATION IN ORIGINAL DELAY SPACE

    Xtrain_center = Xin_train - np.mean(Xin_train, axis=0, keepdims=True)
    Xhat_train_center = Xhat_train - np.mean(Xin_train, axis=0, keepdims=True)

    Uv_train, Sv_train, Vtv_train = np.linalg.svd(Xtrain_center, full_matrices=False)
    Xtrain_pca = Uv_train[:, :3] * Sv_train[:3]
    Xhat_train_pca = Xhat_train_center @ Vtv_train[:3, :].T

    Xtest_center = Xin_test - np.mean(Xin_test, axis=0, keepdims=True)
    Xhat_test_center = Xhat_test - np.mean(Xin_test, axis=0, keepdims=True)

    Uv_test, Sv_test, Vtv_test = np.linalg.svd(Xtest_center, full_matrices=False)
    Xtest_pca = Uv_test[:, :3] * Sv_test[:3]
    Xhat_test_pca = Xhat_test_center @ Vtv_test[:3, :].T

    print("\nPCA SHAPES FOR VISUALIZATION")
    print(f"Xtrain_pca shape      = {Xtrain_pca.shape}")
    print(f"Xhat_train_pca shape  = {Xhat_train_pca.shape}")
    print(f"Xtest_pca shape       = {Xtest_pca.shape}")
    print(f"Xhat_test_pca shape   = {Xhat_test_pca.shape}")

    plot_3d_curve(
        Xtrain_pca,
        "PCA Time Delay Embedding (Input Train)",
        "PCA_Time_delay_Embedding_Input_Train.png"
    )
    plot_3d_curve(
        Xhat_train_pca,
        "PCA Time Delay Embedding (Predicted Train)",
        "PCA_Time_delay_Embedding_Predicted_Train.png"
    )
    plot_3d_overlay(
        Xtrain_pca,
        Xhat_train_pca,
        "PCA Time Delay Embedding (Input vs Predicted Train)",
        "PCA_Time_delay_Embedding_Input_vs_Predicted_Train.png",
        label1="Input Train",
        label2="Predicted Train"
    )

    plot_3d_curve(
        Xtest_pca,
        "PCA Time Delay Embedding (Input Test)",
        "PCA_Time_delay_Embedding_Input_Test.png"
    )
    plot_3d_curve(
        Xhat_test_pca,
        "PCA Time Delay Embedding (Predicted Test)",
        "PCA_Time_delay_Embedding_Predicted_Test.png"
    )
    plot_3d_overlay(
        Xtest_pca,
        Xhat_test_pca,
        "PCA Time Delay Embedding (Input vs Predicted Test)",
        "PCA_Time_delay_Embedding_Input_vs_Predicted_Test.png",
        label1="Input Test",
        label2="Predicted Test"
    )


    # ERRORS

    irmae_train_full_mse = np.mean((XPCAhat_train_irmae - XPCA_train) ** 2)
    irmae_test_full_mse  = np.mean((XPCAhat_test_irmae - XPCA_test) ** 2)

    ode_train_full_mse = np.mean((XPCAhat_train - XPCA_train) ** 2)
    ode_test_full_mse  = np.mean((XPCAhat_test - XPCA_test) ** 2)

    ode_train_rel_l2 = np.linalg.norm(XPCAhat_train - XPCA_train) / np.linalg.norm(XPCA_train)
    ode_test_rel_l2  = np.linalg.norm(XPCAhat_test - XPCA_test) / np.linalg.norm(XPCA_test)

    print("\nERROR SUMMARY")
    print(f"IRMAE Train MSE = {irmae_train_full_mse:.6e}")
    print(f"IRMAE Test  MSE = {irmae_test_full_mse:.6e}")
    print(f"ODE   Train MSE = {ode_train_full_mse:.6e}")
    print(f"ODE   Test  MSE = {ode_test_full_mse:.6e}")
    print(f"ODE   Train relL2 = {ode_train_rel_l2:.6e}")
    print(f"ODE   Test  relL2 = {ode_test_rel_l2:.6e}")


    # SAVE MODELS

    torch.save(irmae.state_dict(), "irmae_model_state_dict.pt")
    torch.save(irmae, "irmae_model.pt")

    torch.save(odefunc.state_dict(), "latent_ode_model_state_dict.pt")
    torch.save(odefunc, "latent_ode_model.pt")


    # SAVE OUTPUTS

    out_pkl = {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "X_train_embed": X_train_embed,
        "X_test_embed": X_test_embed,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_std": X_train_std,
        "X_test_std": X_test_std,
        "Xmean": Xmean,
        "Xstd": Xstd,
        "U_pca": U_pca,
        "S_pca": S_pca,
        "XPCA_train": XPCA_train,
        "XPCA_test": XPCA_test,
        "XPCAhat_train_irmae": XPCAhat_train_irmae,
        "XPCAhat_test_irmae": XPCAhat_test_irmae,
        "Z_train": Z_train,
        "Z_test": Z_test,
        "Zhat_train": Zhat_train,
        "Zhat_test": Zhat_test,
        "XPCAhat_train": XPCAhat_train,
        "XPCAhat_test": XPCAhat_test,
        "Xin_train": Xin_train,
        "Xin_test": Xin_test,
        "Xhat_train": Xhat_train,
        "Xhat_test": Xhat_test,
        "Xtrain_pca": Xtrain_pca,
        "Xhat_train_pca": Xhat_train_pca,
        "Xtest_pca": Xtest_pca,
        "Xhat_test_pca": Xhat_test_pca,
        "irmae_train_loss": np.asarray(irmae_train_err),
        "irmae_test_loss": np.asarray(irmae_test_err),
        "ode_train_loss": np.asarray(ode_train_err),
        "ode_test_loss": np.asarray(ode_test_err),
        "latent_singular_values": latent_singular_values,
        "n_delays": n_delays,
        "tau": tau,
        "skip": skip,
        "r_pca": r_pca,
        "irmae_latent_dim": args.irmae_latent_dim,
        "train_frac": frac,
        "dt_effective": dt_effective,
        "irmae_train_full_mse": irmae_train_full_mse,
        "irmae_test_full_mse": irmae_test_full_mse,
        "ode_train_full_mse": ode_train_full_mse,
        "ode_test_full_mse": ode_test_full_mse,
        "ode_train_rel_l2": ode_train_rel_l2,
        "ode_test_rel_l2": ode_test_rel_l2,
        "args": vars(args),
    }

    with open("irmae_latent_ode_results.pkl", "wb") as f:
        pickle.dump(out_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)


    print("\nSaved files:")
    print("\nirmae_latent_ode_results.pkl")
