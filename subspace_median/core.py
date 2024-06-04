import torch
from torch.linalg import qr, eigh

def orthonormalize(U):
    Q, R = qr(U)
    return Q

def compute_projection_matrix(U):
    return U @ U.T

def approx_gm(P_list, T_gm, epsilon_gm):
    return weiszfeld_algorithm(P_list, T_gm, epsilon_gm)

def weiszfeld_algorithm(A, T_gm, epsilon_gm):
    A = torch.stack(A)
    L, n, m = A.shape
    z_t = A.mean(dim=0)

    for t in range(T_gm):
        numer = torch.zeros_like(z_t)
        denom = 0.0

        for i in range(L):
            dist = torch.norm(A[i] - z_t)
            if dist == 0:
                dist = 1.0
            weight = 1.0 / dist
            numer += weight * A[i]
            denom += weight

        z_t1 = numer / denom

        if torch.norm(z_t1 - z_t) < epsilon_gm:
            break
        z_t = z_t1

    return z_t

def subspace_median(U_list, T_gm, epsilon_gm):
    U_list = [orthonormalize(U) for U in U_list]
    P_list = [compute_projection_matrix(U) for U in U_list]
    P_gm = approx_gm(P_list, T_gm, epsilon_gm)
    l_best = torch.argmin(torch.tensor([torch.norm(P - P_gm) for P in P_list]))
    return U_list[l_best]
