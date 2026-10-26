import argparse
import torch
import shutil
from pathlib import Path 
from utils.utils import *
from utils.models import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--content_dir", type=str, default="E:/Projects-2026/project-2/NST_Code/content_data")
    parser.add_argument("--style_dir", type=str, default="E:/Projects-2026/project-2/NST_Code/style_data")
    parser.add_argument("--vgg", type=str, default="E:/Projects-2026/project-2/NST_Code/vgg_normalised.pth")
    parser.add_argument("--experiment", type=str, default="experiment1")
    parser.add_argument("--final_size", type=int, default=256)
    parser.add_argument("--content_size", type=int, default=512)
    parser.add_argument("--style_size", type=int, default=512)
    parser.add_argument("--crop", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--content_weight", type=int, default=1.0)
    parser.add_argument("--style_weight", type=int, default=5.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    
    return parser.parse_args()


def save_checkpoint(save_dir, epoch, decoder, optimizer):
    checkpoint = {
        "epoch"               : epoch,
        "decoder_state_dict"  : decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"  [SAVED] checkpoint_epoch_{epoch}.pth")
    
    # stores output after sessions(if session died it can be helpful.)
    kaggle_output = Path("/kaggle/output/")
    if kaggle_output.exists():
        shutil.copy(checkpoint_path, kaggle_output / f"checkpoint_epoch_{epoch}.pth")
        print(f"  [BACKED UP] /kaggle/output/checkpoint_epoch_{epoch}.pth")


def main():
    args = parse_arguments()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / "args.txt", "w") as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key}:{value}\n")
            
    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform   = get_transform(args.style_size,   args.crop, args.final_size)      
    
    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset   = ImageFolderDataset(args.style_dir,   style_transform)
            
    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    style_dataloader   = DataLoader(style_dataset,   batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
            
    encoder   = VGGEncoder(args.vgg).to(device)   
    decoder   = Decoder().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch))

    start_epoch = 0
    if args.resume:
        if not args.checkpoint_path:
            print("[ERROR] --resume requires --checkpoint_path")
            exit()
        checkpoint  = torch.load(args.checkpoint_path)
        start_epoch = checkpoint["epoch"]
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"[RESUMED] Starting from epoch {start_epoch + 1}")
         
    mse_loss = nn.MSELoss()
    encoder.eval()
         
    for epoch in range(start_epoch, args.epochs):
        
        running_loss = running_content_loss = running_style_loss = 0.0
        
        progress_bar = tqdm(zip(content_dataloader, style_dataloader),
                            total=min(len(content_dataloader), len(style_dataloader)))
        
        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch   = style_batch.to(device)
            
            c_feats = encoder(content_batch)     
            s_feats = encoder(style_batch)
            t       = adaptive_instant_normalization(c_feats[-1], s_feats[-1])
            g       = decoder(t)
            g_feats = encoder(g)
             
            content_loss = mse_loss(g_feats[-1], t) * args.content_weight
            style_loss   = 0
            for g_feat, s_feat in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_feat)
                s_mean, s_std = calc_mean_std(s_feat)
                style_loss += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            style_loss = style_loss * args.style_weight
            loss       = content_loss + style_loss  
            
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
            
            progress_bar.set_description(f"loss:{loss.item():4f}, content:{content_loss.item():4f}, style:{style_loss.item():4f}")
            
            running_loss         += loss.item()
            running_content_loss += content_loss.item()
            running_style_loss   += style_loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(f"iter {epoch+1}: loss:{running_loss/len(content_dataloader):4f}, content:{running_content_loss/len(content_dataloader)}, style:{running_style_loss/len(content_dataloader)}")
        
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(save_dir, epoch + 1, decoder, optimizer)
            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f"output_{epoch+1}.png", nrow=args.batch_size)


if __name__ == "__main__":
    main()