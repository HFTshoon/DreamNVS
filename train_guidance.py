
import os
import logging
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TORCH_USE_CUDA_DSA"] = '1'

import argparse
import itertools

from PIL import Image
import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

import numpy as np
from tqdm.auto import tqdm


from utils.condition_encoder import SpatialGuidanceModel, TrajectoryGuidanceModel, load_spatial_guidance_model, load_trajectory_guidance_model
from utils.predict_3d import get_guidance_input
from utils.util_traj import traj2vec

from transformers import AutoTokenizer, CLIPTextModel
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available

import random

def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger

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

def get_images(image_path_list):
    imgs = []
    img_size = []
    for image_path in image_path_list:
        img = Image.open(image_path[0])

        # if image is rgba, remove alpha channel (4 -> 3)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img_size.append(img.size)
        imgs.append(img)

    # get minimum h and w
    min_h = min([h for h, w in img_size])
    min_w = min([w for h, w in img_size])
    
    # make the longer side to 512
    longer_side = max(min_h, min_w)
    target_h = int(min_h / longer_side * 512)
    target_w = int(min_w / longer_side * 512)
    # print(min_h, min_w, " -> ", target_h, target_w)

    # crop the image to the minimum size from the center
    resized_imgs = [] # (N, H, W, 3)
    for img in imgs:
        h, w = img.size
        left = (w - min_w) // 2
        top = (h - min_h) // 2
        right = (w + min_w) // 2
        bottom = (h + min_h) // 2
        img = img.crop((top, left, bottom, right))
        img = img.resize((target_h, target_w))
        img = np.array(img)
        resized_imgs.append(img)

    resized_imgs = np.array(resized_imgs)
    resized_imgs = torch.from_numpy(resized_imgs).float() / 127.5 - 1
    resized_imgs = resized_imgs.permute(0, 3, 1, 2)
    return resized_imgs # (N, 3, H, W)

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
    parser.add_argument('--scale_lr', action='store_true', default=False, help='Scale learning rate')
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

    logger = setup_logger('train_log', os.path.join(args.output_dir, 'log.txt'))
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.diffusion_load_path,
        subfolder='tokenizer',
        revision=None,
        use_fast=False,
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(args.diffusion_load_path, subfolder="scheduler")
        
    text_encoder = CLIPTextModel.from_pretrained(
        args.diffusion_load_path,
        subfolder='text_encoder',
        revision=None
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.diffusion_load_path,
        subfolder='vae',
        revision=None
    )
    
    unet = UNet2DConditionModel.from_pretrained(
        args.diffusion_load_path,
        subfolder='unet',
        revision=None
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    if args.guide_mode == 'guide_concat':
        out_dim = 768
        # if args.spatial_guidance_model_load_path is not None and not args.spatial_guidance_model_load_path.endswith("_768.pth"):
        #     args.spatial_guidance_model_load_path = args.spatial_guidance_model_load_path.replace(".pth", "_768.pth")
        # if args.trajectory_guidance_model_load_path is not None and not args.trajectory_guidance_model_load_path.endswith("_768.pth"):
        #     args.trajectory_guidance_model_load_path = args.trajectory_guidance_model_load_path.replace(".pth", "_768.pth")
    
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
        vae.to(device, dtype = torch.float32)
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype = torch.float32)
    unet.requires_grad_(False)
    unet.to(device, dtype = torch.float32)
    unet.enable_xformers_memory_efficient_attention()
    
    spatial_guidance_model.requires_grad_(args.train_spatial_guidance)
    trajectory_guidance_model.requires_grad_(args.train_trajectory_guidance)
    spatial_guidance_model.to(device, dtype = torch.float32)
    trajectory_guidance_model.to(device, dtype = torch.float32)
    
    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters(), vae.parameters())
    # params_to_optimize = None

    # print("Spatial guidance model trainable parameters: ", sum(p.numel() for p in spatial_guidance_model.parameters() if p.requires_grad))
    # print("Trajectory guidance model trainable parameters: ", sum(p.numel() for p in trajectory_guidance_model.parameters() if p.requires_grad))
    if args.train_spatial_guidance:
        spatial_parameters = spatial_guidance_model.parameters()
        params_to_optimize = itertools.chain(params_to_optimize, spatial_parameters)
    if args.train_trajectory_guidance:
        trajectory_parameters = trajectory_guidance_model.parameters()
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
    
    start_extra_epoch = 0
    start_extra_step = 0
    use_extra_prob = 0.4
   
    for epoch in range(args.num_train_epochs):
        unet.train()
        
        if args.train_spatial_guidance:
            spatial_guidance_model.train()
        if args.train_trajectory_guidance:
            trajectory_guidance_model.train()
        
        progress_bar = tqdm(range(0, len(train_dataloader)))
        progress_bar.set_description("Steps")
        global_step = 0
        
        for step, batch in enumerate(train_dataloader):
            use_extra = epoch >= start_extra_epoch \
                    and step > start_extra_step \
                    and random.random() < use_extra_prob

            if use_extra:
                imgs = get_images(batch["image_path_extra"])
                traj_info = traj2vec(batch["traj_info_extra"][0].numpy()).astype(np.float32)
                # focals = batch["focal_extra"]
            else:
                imgs = get_images(batch["image_path"])
                traj_info = traj2vec(batch["traj_info"][0].numpy()).astype(np.float32)
                # focals = batch["focal"]
            spatial_info = batch["spatial_info"][0]
            
            imgs = imgs.to(device)
            traj_info = torch.tensor(traj_info).to(device)
            spatial_info = spatial_info.to(device)
            if vae is not None:
                model_input = vae.encode(imgs).latent_dist
                model_input = model_input.sample() * vae.config.scaling_factor
            else:
                model_input = imgs
                
            noise = torch.randn_like(model_input)
            bsz, channels, height, width = model_input.shape
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
            )
            timesteps = timesteps.long()
            
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
            input_ids = tokenizer(
                "",
                truncation=True,
                padding="max_length",
                max_length=77,
                return_tensors="pt",
            ).input_ids.to(device)
            guidance_text = text_encoder(
                input_ids,
                attention_mask=None,
            )[0]
            guidance_3d = spatial_guidance_model(spatial_info)
            guidance_traj = trajectory_guidance_model(traj_info.to(device))
            encoder_hidden_states_single = torch.cat([guidance_text, guidance_3d, guidance_traj], dim=1)
            # encoder_hidden_states_single = guidance_text
            # (1, 593, 768)

            encoder_hidden_states = encoder_hidden_states_single.repeat(noisy_model_input.shape[0], 1, 1)
            # (20, 593, 768)

            noisy_model_input = noisy_model_input.to(device)
            encoder_hidden_states_single = encoder_hidden_states_single.to(device)
            timesteps = timesteps.to(dtype=torch.float32)

            model_pred = unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states,
            ).sample

            if model_pred.shape[1] == 6:
                model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # model_pred, target (N, C, H, W)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # print("model_pred: ", model_pred.device, model_pred.dtype, model_pred.shape)
            # print("target: ", target.device, target.dtype, target.shape)
            # print(loss)
            logger.info(f"Epoch: {epoch}, Step: {step}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
            progress_bar.update(1)
            global_step += 1
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)
            
            if global_step >= max_train_steps:
                break

            if step % 100 == 0:
                if args.train_spatial_guidance:
                    spatial_guidance_model.save_model(os.path.join(args.output_dir, f"spatial_guidance_model_epoch{epoch}_step{step}.pth"))
                if args.train_trajectory_guidance:
                    trajectory_guidance_model.save_model(os.path.join(args.output_dir, f"trajectory_guidance_model_epoch{epoch}_step{step}.pth"))

        print("Training done.")
        if args.train_spatial_guidance:
            spatial_guidance_model.save_model(os.path.join(args.output_dir, f"spatial_guidance_model_epoch{epoch}_fin.pth"))
        if args.train_trajectory_guidance:
            trajectory_guidance_model.save_model(os.path.join(args.output_dir, f"trajectory_guidance_model_epoch{epoch}_fin.pth"))
    
if __name__ == '__main__':
    # data_dir = "/mydata/data/hyunsoo/co3d_sample_preprocess"
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