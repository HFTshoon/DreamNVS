import os
import argparse
import torch
import numpy as np

from utils.condition_encoder import SpatialGuidanceModel, TrajectoryGuidanceModel, load_spatial_guidance_model, load_trajectory_guidance_model
from utils.predict_3d import get_guidance_input

import random

class CO3DGuidanceTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = []
        # Iterate through each sequences for each object
        for obj_name in os.listdir(data_dir):
            obj_path = os.path.join(data_dir, obj_name)
            for seq_name in os.listdir(obj_path):
                seq_path = os.path.join(obj_path, seq_name)
                if not os.path.isdir(seq_path):
                    continue

                # For each subsets, randomly assign another subset for extrapolation
                subsets = []
                for file_name in os.listdir(seq_path):
                    if file_name[0].isdigit():
                        subsets.append(file_name[:-4])
                
                for subset in subsets:
                    if len(subsets) > 1:
                        extra = random.choice([k for k in subsets if k != subset])
                        self.data_files.append([os.path.join(obj_name, seq_name), subset, extra])

        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data = self.data_files[idx]
        base_path = os.path.join(self.data_dir, data[0])
        with open(os.path.join(base_path, f"image_{data[1]}.txt")) as f:
            image_path = f.read().splitlines()
        
        with open(os.path.join(base_path, f"image_{data[2]}.txt")) as f:
            image_path_extra = f.read().splitlines()
        
        spatial_info = np.load(os.path.join(base_path, f"{data[1]}.npy"))
        traj_info = np.load(os.path.join(base_path, f"poses_{data[1]}.npy"))
        traj_info_extra = np.load(os.path.join(base_path, f"poses_{data[2]}.npy"))
        focal = np.load(os.path.join(base_path, f"focals_{data[1]}.npy"))
        focal_extra = np.load(os.path.join(base_path, f"focals_{data[2]}.npy"))

        
        return {
            "image_path": image_path,
            "image_path_extra": image_path_extra,
            "spatial_info": spatial_info,
            "traj_info": traj_info,
            "traj_info_extra": traj_info_extra,
            "focal": focal,
            "focal_extra": focal_extra
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Train guidance')
    parser.add_argument('--diffusion_load_path', type=str, default="runwayml/stable-diffusion-v1-5", help='Diffusion model huggingface model load path')
    parser.add_argument('--spatial_guidance_model_load_path', type=str, help='Spatial guidance model load path')
    parser.add_argument('--trajectory_guidance_model_load_path', type=str, help='Trajectory guidance model load path')
    parser.add_argument('--guide_mode', type=str, default='guide_concat', help='Guide mode')
    
    parser.add_argument('--train_spatial_guidance', action='store_true', help='Train spatial guidance model')
    parser.add_argument('--train_trajectory_guidance', action='store_true', help='Train trajectory guidance model')
    
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Number of epochs')
    # parser.add_argument('--num_train_steps', type=int, default=200, help='Number of steps')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--scale_lr', action='store_true', help='Scale learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='constant', 
                        help='Learning rate scheduler ["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"]')
    parser.add_argument('--lr_warmup_steps', type=int, default=0, help='Number of warmup steps')
    parser.add_argument('--lr_num_cycles', type=int, default=1, help='Number of cycles')
    parser.add_argument('--lr_power', type=float, default=1.0, help='Power for polynomial learning rate scheduler')
    parser.add_argument('--dataloader_num_workers', type=int, default=0, help='Number of dataloader workers')
    
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    
    parser.add_argument('--output_dir', type=str, default='./guidance', help='Output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')

    return parser.parse_args()

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.diffusion_load_path,
        subfolder='tokenizer',
        revision=False,
        use_fast=False,
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        
    text_encoder = CLIPTextModel.from_pretrained(
        args.diffusion_load_path,
        subfolder='text_encoder',
        revision=False
    )
    
    vae = AutoEncoderKL.from_pretrained(
        args.diffusion_load_path,
        subfolder='vae',
        revision=False
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.diffusion_load_path,
        subfolder='unet',
        revision=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.guide_mode == 'guide_concat':
        out_dim = 768
        if args.spatial_guidance_path is not None and not args.spatial_guidance_path.endswith("_768.pth"):
            args.spatial_guidance_path = args.spatial_guidance_path.replace(".pth", "_768.pth")
        if args.trajectory_guidance_path is not None and not args.trajectory_guidance_path.endswith("_768.pth"):
            args.trajectory_guidance_path = args.trajectory_guidance_path.replace(".pth", "_768.pth")
    
    if args.spatial_guidance_model_load_path is None:
        spatial_guidance_model = SpatialGuidanceModel(output_dim=out_dim)
    else:
        # Load the spatial guidance model
        spatial_guidance_model = load_spatial_guidance_model(args.spatial_guidance_model_path)
    
    if args.trajectory_guidance_model_load_path is None:
        trajectory_guidance_model = TrajectoryGuidanceModel(output_dim=out_dim)
    else:   
        # Load the trajectory guidance model
        trajectory_guidance_model = load_trajectory_guidance_model(args.trajectory_guidance_model_path)  
    
    if args.guide_mode == 'guide_concat':
        spatial_guidance_model.change_pool("None")
        trajectory_guidance_model.change_pool("None")
        
    if vae is not None:
        vae.requires_grad_(False)
        vae.to(device)
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    unet.requires_grad_(False)
    unet.to(device)
    
    spatial_guidance_model.requires_grad_(args.train_spatial_guidance)
    trajectory_guidance_model.requires_grad_(args.train_trajectory_guidance)
    spatial_guidance_model.to(device)
    trajectory_guidance_model.to(device)
    
    params_to_optimize = None
    if args.train_spatial_guidance:
        spatial_parameters = filter(lambda p: p.requires_grad, spatial_guidance_model.parameters())
        params_to_optimize = spatial_parameters
    if args.train_trajectory_guidance:
        trajectory_parameters = filter(lambda p: p.requires_grad, trajectory_guidance_model.parameters())
        if params_to_optimize is not None:
            params_to_optimize = itertools.chain(params_to_optimize, trajectory_parameters)
        else:
            params_to_optimize = trajectory_parameters
    
    assert params_to_optimize is not None
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    
    train_dataset = CO3DGuidanceTrainDataset(
        args.data_dir
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True
    )
    
    max_train_steps = args.num_train_epochs * len(train_dataloader)
        
    # Set the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    for epoch in range(args.num_train_epochs):
        if vae is not None:
            vae.eval()
        unet.eval()
        text_encoder.eval()
        
        if args.train_spatial_guidance:
            spatial_guidance_model.train()
        if args.train_trajectory_guidance:
            trajectory_guidance_model.train()
        
        progress_bar = tqdm(range(0, args.max_train_steps))
        progress_bar.set_description("Steps")
        
        for step, batch in enumerate(train_dataloader):
            # TODO
            input = batch["input"]
            
            if vae is not None:
                model_input = vae.encode(input).latent_dist
                model_input = model_input.sample() * vae.config.scaling_factor
            else:
                model_input = input
                
            noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()
            
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
            guidance_text = encode_prompt(
                text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            )
            guidance_3d = spatial_guidance_model(batch["spatial_info"][0])
            guidance_traj = trajectory_guidance_model(batch["trajectory_info"][0])
            encoder_hidden_states = torch.cat([guidance_text, guidance_3d, guidance_traj], dim=1)
            
            model_pred = unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states,
            )
            
            if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
            global_step += 1
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break
        
    print("Training done.")
    if args.train_spatial_guidance:
        spatial_guidance_model.save_model(os.path.join(args.output_dir, "spatial_guidance_model_trained.pth"))
    if args.train_trajectory_guidance:
        trajectory_guidance_model.save_model(os.path.join(args.output_dir, "trajectory_guidance_model_trained.pth"))
        

            

    
    
if __name__ == '__main__':
    # data_dir = "../co3d_sample_preprocess"
    # train_dataset = CO3DGuidanceTrainDataset(
    #     data_dir
    # )
    # batch_size = 1
    # dataloader_num_workers = 1
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     num_workers=dataloader_num_workers,
    #     shuffle=True
    # )

    # for i, batch in enumerate(train_dataloader):
    #     print(len(batch["image_path"]))
    #     print(len(batch["image_path_extra"]))
    #     print(batch["spatial_info"].shape)
    #     print(batch["traj_info"].shape)
    #     print(batch["traj_info_extra"].shape)
    #     print(batch["focal"].shape)
    #     print(batch["focal_extra"].shape)
    #     break

    # Parse the arguments
    args = parse_args()
    # Run the main function
    main(args)