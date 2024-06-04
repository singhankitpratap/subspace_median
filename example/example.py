import torch
from torch.linalg import qr, eigh
import random
from subspace_median import subspace_median

def make_positive_definite(matrix):
    epsilon = 1e-5
    return matrix + epsilon * torch.eye(matrix.shape[0])


def generate_data(n, q, L, r, L_byz):
    torch.manual_seed(0)
    U = torch.linalg.qr(torch.randn(n, n).float())[0]
    S = torch.diag(torch.tensor([15]*r + [1] + [0]*(n-r-1)).float())
    Sigma_star = U @ S @ U.T
    Sigma_star = make_positive_definite(Sigma_star)

    mvn = torch.distributions.MultivariateNormal(torch.zeros(n), Sigma_star)
    D = mvn.sample((q,)).T
    D_list = torch.chunk(D, L, dim=1)

    U_list = []
    for D_ell in D_list:
        Sigma_hat = (1 / (D_ell.shape[1] - 1)) * (D_ell @ D_ell.T)
        eigvals, eigvecs = eigh(Sigma_hat)
        U_ell = eigvecs[:, -r:]
        U_list.append(U_ell)
    
    # Randomly replace L_byz U_ell with 100 multiplied by a matrix of ones
    for _ in range(L_byz):
        idx = random.randint(0, L-1)
        U_list[idx] = 100 * torch.ones((n, r))

    return U_list, U[:, :r]

def sd_error(U_star, U_out):
    return torch.norm((torch.eye(U_star.shape[0]) - U_star @ U_star.T) @ U_out)

# Parameters
n = 1000
q = 1800
L = 3
r = 60
L_byz = 1
T_gm = 100
epsilon_gm = 1e-5

# Generate data
U_list, U_star = generate_data(n, q, L, r, L_byz)

# Run Subspace Median
U_out = subspace_median(U_list, T_gm, epsilon_gm)

# Calculate SD error
error = sd_error(U_star, U_out)
print("SD Error:", error.item())
