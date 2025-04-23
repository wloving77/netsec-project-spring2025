import torch
import torch.nn as nn
import torch.optim as optim
from .generator_discriminator import Generator, Discriminator
from torch.utils.data import DataLoader, TensorDataset


def train_gan(
    X_real,
    epochs=100,
    batch_size=128,
    noise_dim=100,
    lr=2e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    X_real_tensor = torch.tensor(X_real, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_real_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    G = Generator(noise_dim=noise_dim, output_dim=X_real.shape[1]).to(device)
    D = Discriminator(input_dim=X_real.shape[1]).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real_batch,) in enumerate(dataloader):
            batch_size = real_batch.size(0)
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, noise_dim).to(device)
            generated = G(z)
            validity = D(generated)
            g_loss = criterion(validity, valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_validity = D(real_batch)
            fake_validity = D(generated.detach())
            d_loss_real = criterion(real_validity, valid)
            d_loss_fake = criterion(fake_validity, fake)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

        print(
            f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}"
        )

    return G


def generate_synthetic_data(generator, num_samples, noise_dim=100, device="cuda"):
    z = torch.randn(num_samples, noise_dim).to(device)
    with torch.no_grad():
        synthetic_data = generator(z).cpu().numpy()
    return synthetic_data
