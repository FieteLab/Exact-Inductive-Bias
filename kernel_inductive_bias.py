import torch
import numpy as np
from scipy.linalg import sqrtm
import utils import compute_inductive_bias

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k

    def __getitem__(self, index):
        data, target = self.dataset[index]
        # flattened_data = data.view(-1)
        # one_hot_target = torch.zeros(self.k)
        # one_hot_target[target] = 1
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def rbf_kernel(A, B, sigma=1.0):
    """
    A is n x d
    B is m x d
    output is n x m
    """
    return torch.exp(-1 * torch.sum((A[:, None, :] - B[None, :, :]) ** 2, dim=2) / (2 * sigma ** 2))


def compute_params_multi(train_batch_loader, train_group_loader, n_train, X_test, y_test, k, lr_alpha, lr_a_x,
                         num_epochs, save_path):
    X_test, y_test = X_test.to(device), y_test.to(device)

    print("Computing alpha...")
    alpha = torch.nn.parameter.Parameter(torch.zeros(n_train, k, requires_grad=True, device=device))

    alpha_losses = []

    try:
        alpha.data = torch.load(f"{save_path}_alpha.pt", map_location=torch.device('cpu'))
    except FileNotFoundError:
        optimizer = torch.optim.Adam([alpha], lr=lr_alpha)
        loss_fn = torch.nn.MSELoss(reduction='mean')

        for epoch in range(num_epochs):
            sum_loss = 0
            for X_batch, y_batch, _ in train_batch_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                y_pred = torch.zeros(k * X_batch.shape[0], 1).to(device)
                for X_group, _, indices in train_group_loader:
                    X_group = X_group.to(device)

                    ker = rbf_kernel(X_batch, X_group)  # batch_size x group_size
                    alpha_idx = alpha[indices].view(-1, k)  # group_size x k
                    prod = (ker @ alpha_idx).view(-1, 1)  # k * batch_size x 1

                    y_pred += prod.view(-1, 1)  # k * batch_size x 1

                loss = loss_fn(y_pred, y_batch.view(-1, 1))

                sum_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            size = len(train_batch_loader)
            print(f'epoch: {epoch}, loss: {sum_loss / size}')
            alpha_losses.append(sum_loss / size)

            # if (epoch+1) % 500 == 0:
            #     torch.save(alpha.data, f"{save_path}_alpha_checkpoint_{epoch+1}.pt")

        torch.save(alpha.data, f"{save_path}_alpha.pt")
        with open(f"{save_path}_alpha_losses.txt", "w") as f:
            f.write(str(alpha_losses))

    print(f"Computing a_x...")
    a_x = torch.nn.parameter.Parameter(torch.zeros(n_train, X_test.shape[0], requires_grad=True, device=device))

    a_x_losses = []

    try:
        a_x.data = torch.load(f"{save_path}_a_x.pt", map_location=torch.device('cpu'))
    except FileNotFoundError:
        optimizer = torch.optim.Adam([a_x], lr=lr_a_x)
        loss_fn = torch.nn.MSELoss(reduction='mean')

        for epoch in range(num_epochs):
            sum_loss = 0
            for X_batch, _, _ in train_batch_loader:
                X_batch = X_batch.to(device)
                ktt_batch = rbf_kernel(X_batch, X_test)  # batch_size x n_test

                ktt_pred = torch.zeros(X_batch.shape[0], X_test.shape[0]).to(device)
                for X_group, _, indices in train_group_loader:
                    X_group = X_group.to(device)
                    ker = rbf_kernel(X_batch, X_group)  # batch_size x group_size
                    ktt_pred += ker @ a_x[indices]

                loss = loss_fn(ktt_pred, ktt_batch)

                sum_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            size = len(train_batch_loader)
            print(f'epoch: {epoch}, loss: {sum_loss / size}')
            a_x_losses.append(sum_loss / size)

            # if (epoch+1) % 500 == 0:
            #     torch.save(a_x.data, f"{save_path}_a_x_checkpoint_{epoch+1}.pt")

        torch.save(a_x.data, f"{save_path}_a_x.pt")
        with open(f"{save_path}_a_x_losses.txt", "w") as f:
            f.write(str(a_x_losses))

    return alpha, a_x, alpha_losses, a_x_losses



def loss_distribution(train_group_loader, X_test, y_test, k, alpha, a_x, num_samples, save_path):
    X_test, y_test = X_test.to(device), y_test.to(device)
    samples = torch.randn(X_test.shape[0], k, num_samples)

    alpha_term = torch.zeros((X_test.shape[0], k)).to(device)
    for X_group, _, indices in train_group_loader:
        X_group = X_group.to(device)
        ker = rbf_kernel(X_test, X_group)  # n_test x group_size
        alpha_term += ker @ alpha[indices].to(device)  # n_test x k

    a_x_term = rbf_kernel(X_test, X_test)
    for X_group, _, indices in train_group_loader:
        X_group = X_group.to(device)
        ker = rbf_kernel(X_test, X_group)  # n_test x group_size
        a_x_term -= ker @ a_x[indices].to(device)  # n_test x n_test

    f_values = alpha_term.detach().cpu().numpy()[:, :, np.newaxis] + np.einsum('ab,bcd->acd',
                                                                               sqrtm(a_x_term.detach().cpu().numpy()),
                                                                               samples.detach().cpu().numpy())  # n_test x k x num_samples
    MSEs = np.mean((f_values - y_test.detach().cpu().numpy()[:, :, np.newaxis]) ** 2, axis=(0, 1))

    return MSEs



def main(X_train, y_train, X_test, y_test, out_dim, target_error):
    """
    X_train: torch.Tensor, shape (n_train, d)
    y_train: torch.Tensor, shape (n_train,)
    X_test: torch.Tensor, shape (n_test, d)
    y_test: torch.Tensor, shape (n_test,)
    out_dim: int, dimension of the output space
    target_error: float, target error for inductive bias
    """
    n_train = X_train.shape[0]
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_batch_loader = torch.utils.data.DataLoader(IndexedDataset(train_dataset, out_dim), batch_size=5, shuffle=True)
    train_group_loader = torch.utils.data.DataLoader(IndexedDataset(train_dataset, out_dim), batch_size=10, shuffle=True)

    alpha, a_x, alpha_losses, a_x_losses = compute_params_multi(train_batch_loader, train_group_loader, n_train, X_test,
                                                                y_test, out_dim, lr_alpha=1e-2, lr_a_x=1e-3,
                                                                num_epochs=10000,
                                                                save_path='test')

    MSEs = loss_distribution(train_group_loader, X_test, y_test, out_dim, alpha, a_x, num_samples=10000,
                                                 save_path='test')
    return compute_inductive_bias(MSEs, target_error)
