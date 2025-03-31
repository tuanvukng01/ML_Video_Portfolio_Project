import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pretrained_model import CompressAIWrapper
from data_loader import VideoFrameDataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


def loss_function(x_hat, x, likelihoods, lambda_rd=0.01):
    # Rate-Distortion Loss
    mse = torch.mean((x - x_hat) ** 2)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(torch.log(lk).sum() for lk in likelihoods.values()) / (-torch.log(torch.tensor(2.0)) * num_pixels)
    return mse + lambda_rd * bpp, mse.item(), bpp.item()


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset = VideoFrameDataset(root_dir=args.data_dir, img_size=(args.img_size, args.img_size), fraction = args.data_fraction)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = CompressAIWrapper(quality=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    logging.info("Starting training...")
    for epoch in range(args.epochs):
        total_loss, total_mse, total_bpp = 0.0, 0.0, 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        model.train()

        for batch in progress_bar:
            batch = batch.to(device)
            optimizer.zero_grad()

            recon_batch, likelihoods = model(batch)
            loss, mse, bpp = loss_function(recon_batch, batch, likelihoods)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse
            total_bpp += bpp

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", mse=f"{mse:.4f}", bpp=f"{bpp:.4f}")

        avg_loss = total_loss / len(dataloader)
        avg_mse = total_mse / len(dataloader)
        avg_bpp = total_bpp / len(dataloader)

        logging.info(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Avg MSE: {avg_mse:.4f}, Avg BPP: {avg_bpp:.4f}")

    torch.save(model.state_dict(), 'saved_models/compressai.pth')
    logging.info("Training complete. Model saved.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train CompressAI model for Neural Video Compression")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--data_fraction', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    train(args)