import os 
import torch
import numpy as np
import imageio
from PIL import Image
from pipeline import MovePipeline
from copy import deepcopy
import torchvision
import argparse
from diffusers import DDIMScheduler, AutoencoderKL
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl
from utils.predict import predict_z0, splat_flowmax
from utils.predict_opflow import predict_z0_opflow
from utils.predict_3d import predict_3d, get_trajectory
# from utils.condition_encoder_legacy import load_guidance_model
from utils.condition_encoder import load_spatial_guidance_model, load_trajectory_guidance_model
from diffusers.utils.import_utils import is_xformers_available

def main(args):
    os.makedirs(args.save_dir,exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.source_image = args.source_image.to(device)
    full_h, full_w = args.source_image.shape[2:4]
    sup_res_h = int(full_h/8)
    sup_res_w = int(full_w/8)
    # load model
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                            beta_schedule="scaled_linear", clip_sample=False,
                            set_alpha_to_one=False, steps_offset=1)
    model = MovePipeline.from_pretrained(args.model_path, scheduler=scheduler).to(device)
    model.modify_unet_forward()
    if args.vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(
            args.vae_path
        ).to(model.vae.device, model.vae.dtype)

    if is_xformers_available(): model.unet.enable_xformers_memory_efficient_attention()
    else: assert False
    
    # set lora
    if os.path.exists(os.path.join(args.lora_path,"lora.ckpt")): weight_name = "lora.ckpt"
    else:weight_name=None
    if args.lora_path == "":
        print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("applying lora: " + args.lora_path)
        model.unet.load_attn_procs(args.lora_path,weight_name=weight_name)

    # set guidance model
    print("loading guidance models")
    if args.guide_mode == "guide_concat":
        if not args.spatial_guidance_path.endswith("_768.pth"):
            args.spatial_guidance_path = args.spatial_guidance_path.replace(".pth", "_768.pth")
        if not args.trajectory_guidance_path.endswith("_768.pth"):
            args.trajectory_guidance_path = args.trajectory_guidance_path.replace(".pth", "_768.pth")
    model.spatial_guidance_model = load_spatial_guidance_model(args.spatial_guidance_path).to(device)
    model.trajectory_guidance_model = load_trajectory_guidance_model(args.trajectory_guidance_path).to(device)
    model.spatial_guidance_model.load_lora(args.spatial_guidance_lora_path)
    model.trajectory_guidance_model.load_lora(args.trajectory_guidance_lora_path)

    if args.guide_mode == "guide_concat":
        model.spatial_guidance_model.change_pool("None")
        model.trajectory_guidance_model.change_pool("None")

    # perform dust3r/mast3r and get pointmaps
    guidance_3d, guidance_traj, focals, pps, poses, pts3d, conf = predict_3d(model, args)

    if args.guide_mode == "baseline":
        guidance_3d = None
        guidance_traj = None
        invert_function = model.invert
    elif args.guide_mode == "guide_concat":
        assert guidance_3d is not None and guidance_traj is not None
        invert_function = model.invert
        predict_function = predict_z0_opflow

    elif args.guide_mode == "guide_cond":
        assert guidance_3d is not None and guidance_traj is not None
        invert_function = model.invert_guide_cond
    else:
        raise ValueError("Invalid mode")

    if args.guide_mode == "baseline":
        flow1to2,flow2to1=predict_z0(model, args, sup_res_h, sup_res_w, guidance_3d, guidance_traj)
    elif args.guide_mode == "guide_concat":
        flow1to2,flow2to1=predict_z0_opflow(model, args, sup_res_h, sup_res_w, pts3d, conf, guidance_3d, guidance_traj)
    else:
        raise NotImplementedError

    with torch.no_grad():
        invert_code, pred_x0_list = invert_function(args.source_image,
                                args.prompt,
                                guidance_3d,
                                guidance_traj,
                                guidance_scale=args.guidance_scale,
                                num_inference_steps=args.n_inference_step,
                                num_actual_inference_steps=args.n_actual_inference_step,
                                return_intermediates=True)
        init_code = deepcopy(invert_code)
        pred_code = deepcopy(pred_x0_list[args.n_actual_inference_step])
        
        src_mask = torch.ones(2, 1, init_code.shape[2], init_code.shape[3]).cuda()
        input_code = torch.cat([init_code, src_mask], 1)
        # inject self-attention 
        editor = MutualSelfAttentionControl(start_step=0,
                                        start_layer=10,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)
        # if args.lora_path == "":
        #    register_attention_editor_diffusers(model, editor, attn_processor='attn_proc')
        # else:
        #    register_attention_editor_diffusers(model, editor, attn_processor='lora_attn_proc')

        # apply inter frames attention 
        # setattr(editor, 'flag', True)
            
        input_latents = []
        input_pred = []
        input_latents.append(init_code[:1])
        input_pred.append(pred_code[:1])

        trajectory = get_trajectory(args.img_path, None)
        for i in range(1,args.Time): 
            time = i/args.Time
            metric1 = splat_flowmax(init_code[:1], init_code[1:], flow1to2, 1-time)
            metric2 = splat_flowmax(init_code[1:], init_code[:1], flow2to1, time)
            init = torch.where(metric1>=metric2,init_code[:1],init_code[1:])
            input_latents.append(init)
        
        input_pred.append(torch.load(os.path.join(args.save_dir,'pred_list.pt')))
        input_latents.append(init_code[1:])
        input_pred.append(pred_code[1:])
        input_latents=torch.cat(input_latents,dim=0)
        input_pred=torch.cat(input_pred,dim=0) 

        gen_image = model(
            prompt=args.prompt,
            guidance_3d=guidance_3d,
            guidance_traj=guidance_traj,
            batch_size=input_latents.shape[0],
            latents=input_latents,
            pred_x0=input_pred,
            guidance_scale=1.0,
            num_inference_steps=args.n_inference_step,
            num_actual_inference_steps=args.n_actual_inference_step,
            save_dir=args.save_dir
            )
        if args.save_inter:
            for i in range(gen_image.shape[0]):
                torchvision.utils.save_image(gen_image[i],os.path.join(args.save_dir,'%d.png' %i))
            pass
        return gen_image


if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser(description='frame interpolator model')

    parser.add_argument('--prompt', type=str, help='text prompt (should  consistent with lora prompt)', required=False, default="")
    parser.add_argument('--img_path', type=str, help='image pair directory', required=True)
    parser.add_argument('--model_path', type=str, help='diffusion model directory', required=False, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--vae_path', type=str, help='vae model directory', default='default')
    parser.add_argument('--lora_path', type=str, help='lora model directory', required=True)
    parser.add_argument('--spatial_guidance_path', type=str, help='guidance model directory', required=True)
    parser.add_argument('--trajectory_guidance_path', type=str, help='guidance model directory', required=True)
    parser.add_argument('--spatial_guidance_lora_path', type=str, help='guidance model directory', required=True)
    parser.add_argument('--trajectory_guidance_lora_path', type=str, help='guidance model directory', required=True)
    parser.add_argument('--save_dir', type=str, help='save results directory', required=False, default="./results")
    parser.add_argument('--guide_mode', type=str, help='model mode', default='guide_concat')
    parser.add_argument('--guidance_scale', type=float, help='CFG', default=2.0)
    parser.add_argument('--guidance_scale_3d', type=float, help='CFG', default=2.0)
    parser.add_argument('--guidance_scale_traj', type=float, help='CFG', default=2.0)
    parser.add_argument('--n_inference_step', type=int, help='total noisy timestamp', default=50)
    parser.add_argument('--feature_inversion', type=int, help='DDIM Inversion steps for feature maps', default=14)
    parser.add_argument('--n_actual_inference_step', type=int, help='DDIM Inversion steps for inference', default=30)
    parser.add_argument('--unet_feature_idx', type=str, help='which up-blocks for feature maps', default=[2])
    parser.add_argument('--Time', type=int, help='total interpolation frames', default=33)
    parser.add_argument('--save_inter', type=bool, help='save intermediate frames', default=True)
    

    args = parser.parse_args()

    img_pathlist = [os.path.join(args.img_path,'0.png'),os.path.join(args.img_path,'1.png')]
    img = []
    for img_path in img_pathlist:
        img.append(imageio.imread(img_path))
    image = np.stack((img[0],img[1]))
    source_image =  torch.from_numpy(image).float() / 127.5 - 1

    # if image is rgba, remove alpha channel (4 -> 3)
    if source_image.shape[3] == 4:
        source_image = source_image[:,:,:,:3]

    # resize to (2, 512, 512, 3)
    source_image = torch.nn.functional.interpolate(source_image.permute(0,3,1,2), size=(512, 512), mode='bilinear', align_corners=False)
    source_image = source_image.permute(0,2,3,1)

    args.source_image = source_image.permute(0,3,1,2)
    
    gen_image = main(args)
    gen_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()
    gen_image = (gen_image * 255).astype(np.uint8)
    img_array=[]
    for filename in gen_image:
        img_array.append(filename)
    imageio.mimwrite(os.path.join(args.save_dir, 'output.mp4'), img_array, fps=10, quality=8)
