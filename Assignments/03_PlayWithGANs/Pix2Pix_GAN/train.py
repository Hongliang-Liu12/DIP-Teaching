import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from GAN import Pix2PixModel, ImageGenerator, ImageDiscriminator
from torch.optim.lr_scheduler import StepLR

os.chdir(r"E:\Git\DIP-Teaching\Assignments\03_PlayWithGANs\Pix2Pix_GAN")

def save_and_display_images(batch_inputs, batch_targets, batch_outputs, save_folder, epoch_num, image_count=5):
    """
    Saves input, target, and output images for visualization.
    """
    os.makedirs(f'{save_folder}/epoch_{epoch_num}', exist_ok=True)
    for image_idx in range(image_count):
        # Convert tensors to numpy arrays
        input_image = batch_inputs[image_idx].cpu().detach().numpy()
        input_image = np.transpose(input_image, (1, 2, 0))
        input_image = ((input_image + 1) / 2 * 255).astype(np.uint8)
        
        target_image = batch_targets[image_idx].cpu().detach().numpy()
        target_image = np.transpose(target_image, (1, 2, 0))
        target_image = ((target_image + 1) / 2 * 255).astype(np.uint8)
        
        output_image = batch_outputs[image_idx].cpu().detach().numpy()
        output_image = np.transpose(output_image, (1, 2, 0))
        output_image = ((output_image + 1) / 2 * 255).astype(np.uint8)
        
        # Concatenate images horizontally
        comparison_image = np.hstack((input_image, target_image, output_image))
        # Save the image
        cv2.imwrite(f'{save_folder}/epoch_{epoch_num}/image_{image_idx + 1}.png', comparison_image)

def train_epoch(model, data_loader, optimizer_gen, optimizer_disc, loss_gen, loss_disc, device, current_epoch, total_epochs):
    model.train()
    total_loss_gen = 0.0
    total_loss_disc = 0.0

    for batch_idx, (real_imgs, semantic_imgs) in enumerate(data_loader):
        real_imgs = real_imgs.to(device)
        semantic_imgs = semantic_imgs.to(device)
        
        # Train Discriminator
        optimizer_disc.zero_grad()
        # Real images loss
        combined_real = torch.cat((semantic_imgs, real_imgs), dim=1)
        disc_real_output = model.image_discriminator(combined_real)
        loss_disc_real = loss_disc(disc_real_output, torch.ones_like(disc_real_output))
        
        # Fake images loss
        generated_imgs = model.image_generator(semantic_imgs)
        combined_fake = torch.cat((semantic_imgs, generated_imgs.detach()), dim=1)
        disc_fake_output = model.image_discriminator(combined_fake)
        loss_disc_fake = loss_disc(disc_fake_output, torch.zeros_like(disc_fake_output))
        
        if current_epoch % 5 == 0 and batch_idx == 0:
            save_and_display_images(semantic_imgs, real_imgs, generated_imgs, 'train_results', current_epoch)
        
        # Total discriminator loss
        total_disc_loss = (loss_disc_real + loss_disc_fake) / 2
        total_disc_loss.backward()
        optimizer_disc.step()
        
        # Train Generator
        optimizer_gen.zero_grad()
        # GAN loss
        disc_fake_output = model.image_discriminator(combined_fake)
        loss_gan = loss_disc(disc_fake_output, torch.ones_like(disc_fake_output))
        # L1 loss
        loss_l1 = loss_gen(generated_imgs, real_imgs)
        # Total generator loss
        total_gen_loss = loss_gan + 7 * loss_l1
        total_gen_loss.backward()
        optimizer_gen.step()
        
        total_loss_gen += total_gen_loss.item()
        total_loss_disc += total_disc_loss.item()
        
        print(f"Epoch [{current_epoch + 1}/{total_epochs}], Batch [{batch_idx + 1}/{len(data_loader)}], Gen Loss: {total_gen_loss.item():.4f}, Disc Loss: {total_disc_loss.item():.4f}")

def validate_model(model, val_loader, loss_function, device, current_epoch, total_epochs):
    model.eval()
    val_loss_gen = 0.0

    with torch.no_grad():
        for batch_idx, (real_imgs, semantic_imgs) in enumerate(val_loader):
            real_imgs = real_imgs.to(device)
            semantic_imgs = semantic_imgs.to(device)
            
            # Generate fake images
            generated_imgs = model.image_generator(semantic_imgs)
            # Compute L1 loss
            loss_l1 = loss_function(generated_imgs, real_imgs)
            val_loss_gen += loss_l1.item()
            
            # Save validation images every 5 epochs
            if current_epoch % 5 == 0 and batch_idx == 0:
                save_and_display_images(semantic_imgs, real_imgs, generated_imgs, 'val_results', current_epoch)
    
    # Calculate average generator loss
    avg_val_loss = val_loss_gen / len(val_loader)
    print(f'Epoch [{current_epoch + 1}/{total_epochs}], Validation Generator Loss: {avg_val_loss:.4f}')

def main():
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Datasets and loaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')
    
    train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False, num_workers=4)
    
    # Model, loss functions, and optimizers
    model = Pix2PixModel().to(device)
    loss_generator = nn.L1Loss()
    loss_discriminator = nn.BCELoss()
    optimizer_generator = optim.Adam(model.image_generator.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_discriminator = optim.Adam(model.image_discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
    
    # Learning rate schedulers
    scheduler_generator = StepLR(optimizer_generator, step_size=200, gamma=0.2)
    scheduler_discriminator = StepLR(optimizer_discriminator, step_size=200, gamma=0.2)
    
    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer_generator, optimizer_discriminator, loss_generator, loss_discriminator, device, epoch, num_epochs)
        validate_model(model, val_loader, loss_generator, device, epoch, num_epochs)
        
        # Update learning rates
        scheduler_generator.step()
        scheduler_discriminator.step()
        
        # Save model checkpoints every 20 epochs
        """
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')
        """

if __name__ == '__main__':
    main()