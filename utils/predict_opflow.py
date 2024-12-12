import os
import torch
import torch.nn as nn
from third_party.softsplat import softsplat
import numpy as np
import cv2
from copy import deepcopy

from utils.predict_3d import get_extrinsics_intrinsics, get_trajectory
from utils.util_flow import compute_optical_flow, create_camera_intrinsics

def vis_flow(flow, save_path):
    # flow: (H, W, 2)
    h,w = flow.shape[:2]
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]

    # draw flow image with arrow
    flow_img = np.zeros((h+200, w+200, 3), dtype=np.uint8)

    for i in range(0, h, 20):
        for j in range(0, w, 20):
            cv2.arrowedLine(flow_img, (j+100, i+100), (j+100+int(flow_x[i,j])//10, i+100+int(flow_y[i,j])//10), (255, 255, 255), 1)
    cv2.imwrite(save_path.replace(".png", "_arrow.png"), flow_img)

    # clip flow value with w, h
    flow_x = np.clip(flow_x, -w, w)
    flow_y = np.clip(flow_y, -h, h)

    # change flow value to 0~255
    flow_x = (flow_x + w) * 255 / (2*w)
    flow_y = (flow_y + h) * 255 / (2*h)

    # create flow image (x: blue, y: red)
    flow_x = flow_x.astype(np.uint8)
    flow_y = flow_y.astype(np.uint8)
    
    # fill with white
    flow_img = np.zeros((h, w, 3), dtype=np.uint8)
    flow_img[:,:,0] = flow_x
    cv2.imwrite(save_path.replace(".png", "_x.png"), flow_x)

    flow_img = np.zeros((h, w, 3), dtype=np.uint8)
    flow_img[:,:,1] = flow_y
    cv2.imwrite(save_path.replace(".png", "_y.png"), flow_y)
    
def optical_flow(imgs, pts3d, extrinsics, intrinsics, forward, save_path, f_size):
    # imgs: (2, 3, real_H, real_W) 
    # pts3d: 2 X (H, W, 3)
    # poses: (2, 4, 4)
    # focals: (2, 1)
    focals = intrinsics["focals"]
    pps = intrinsics["principal_points"]
    K = create_camera_intrinsics(fx=focals[0], fy=focals[1], cx=pps[0][0].item(), cy=pps[0][1].item())

    if forward:
        h, w = pts3d[0].size()[:2]
        input_p = pts3d[0].reshape((-1,3)).detach().cpu().numpy()
        input_RT = extrinsics[0].detach().cpu().numpy()
        query_RT = extrinsics[1].detach().cpu().numpy()
    else:
        h, w = pts3d[1].size()[:2]
        input_p = pts3d[1].reshape((-1,3)).detach().cpu().numpy()
        input_RT = extrinsics[1].detach().cpu().numpy()
        query_RT = extrinsics[0].detach().cpu().numpy()
    flow = compute_optical_flow(input_p, input_RT[:3, :3], input_RT[:3, 3], query_RT[:3, :3], query_RT[:3, 3], K)
    flow = flow.reshape((h, w, 2))


    # pad flow to real_H, real_W
    real_h, real_w = imgs.size()[2:]
    real_flow = np.zeros((real_h, real_w, 2))
    shift_h = (real_h - h) // 2
    shift_w = (real_w - w) // 2
    real_flow[shift_h:shift_h+h, shift_w:shift_w+w] = flow

    # save x,y flow with color as image
    if save_path is not None:
        vis_flow(real_flow, save_path)

    # resize flow to f_size with bilinear interpolation
    flow = cv2.resize(flow, (f_size[1], f_size[0]), interpolation=cv2.INTER_LINEAR)
    flow = torch.tensor(flow).cuda().float()

    return flow

def backwarp(tenIn, tenFlow):
    backwarp_tenGrid = {}
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[3], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(start=-1.0, end=1.0, steps=tenFlow.shape[2], dtype=tenFlow.dtype, device=tenFlow.device).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0)], 1)
    return torch.nn.functional.grid_sample(input=tenIn, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)

def splat_flow(imgs, frame1, pts3d1, conf1, extrinsics, intrinsics, sup_res_h, sup_res_w):
    # frame1: tensor[1,c,h,w] cuda()
    # pts3d: 2 X (H, W, 3)
    flow = optical_flow(imgs, [pts3d1,None], extrinsics, intrinsics, forward = True, save_path=None, f_size=(sup_res_h, sup_res_w))
    flow = flow.permute(2,0,1).unsqueeze(0).cuda()

    h, w = pts3d1.size()[:2]
    real_h, real_w = imgs.size()[2:]
    shift_h = (real_h - h) // 2
    shift_w = (real_w - w) // 2
    conf = np.zeros((real_h, real_w))
    conf[shift_h:shift_h+h, shift_w:shift_w+w] = np.exp(conf1.detach().cpu().numpy())
    conf = cv2.resize(conf, (sup_res_h, sup_res_w), interpolation=cv2.INTER_LINEAR)
    tenMetric = torch.tensor(conf).unsqueeze(0).unsqueeze(0).cuda().float()

    out_soft = softsplat(tenIn=frame1, tenFlow=flow, tenMetric=(-10.0 * (np.exp(1.5)-tenMetric)).clip(-10.0, 10.0), strMode='soft')
    return out_soft

def splat_flowmax(frame1, frame2, flow, time):
    # frame1: tensor[1,c,h,w] cuda()
    # frame2: tensor[1,c,h,w] cuda()
    # flow  : tensor[h,w,2] cuda()
    flow = flow.permute(2,0,1).unsqueeze(0).cuda()
    tenMetric = torch.nn.functional.l1_loss(input=frame1, target=backwarp(tenIn=frame2, tenFlow=flow), reduction='none').mean([1], True)
    # out_soft, mask = softsplat(tenIn=frame1, tenFlow=flow*time, tenMetric=(0.3 - tenMetric).clip(0.001, 1.0), strMode='max')
    return ((0.3-tenMetric).clip(0.001, 1.0))*time

def predict_z0_opflow(model, args, sup_res_h, sup_res_w, pts3d, conf, guidance_3d=None, guidance_traj=None, mode = "baseline"):
    if args.guide_mode == "baseline":
        guidance_3d = None
        guidance_traj = None
        invert_function = model.invert
        forward_unet_features = model.forward_unet_features
    elif args.guide_mode == "guide_concat":
        assert guidance_3d is not None and guidance_traj is not None
        invert_function = model.invert
        forward_unet_features = model.forward_unet_features
    elif args.guide_mode == "guide_cond":
        assert guidance_3d is not None and guidance_traj is not None
        invert_function = model.invert_guide_cond
        forward_unet_features = model.forward_unet_features_guide_cond
    else:
        raise ValueError("Invalid mode")


    with torch.no_grad():
        invert_code, pred_x0_list = invert_function(args.source_image,
                                args.prompt,
                                guidance_3d=guidance_3d,
                                guidance_traj=guidance_traj,
                                guidance_scale=args.guidance_scale,
                                num_inference_steps=args.n_inference_step,
                                num_actual_inference_steps=args.n_actual_inference_step,
                                return_intermediates=True)
        
        init_code = deepcopy(invert_code) # (2,4,64,64)
        pred_code = deepcopy(pred_x0_list[args.n_actual_inference_step]) # (2,4,64,64)
        
        src_mask = torch.ones(2, 1, init_code.shape[2], init_code.shape[3]).cuda() # (2,1,64,64)
        input_code = torch.cat([pred_code, src_mask], 1) # (2,5,64,64)

        extrinsics, intrinsics = get_extrinsics_intrinsics(args.img_path)
        trajectory = get_trajectory(args.img_path, None)
        args.Time = len(trajectory)

        flow1to2   = optical_flow(args.source_image, pts3d, extrinsics, intrinsics, forward = True, save_path=os.path.join(args.save_dir, "flow01.png"), f_size=(sup_res_h, sup_res_w))
        flow2to1   = optical_flow(args.source_image, pts3d, extrinsics, intrinsics, forward = False, save_path=os.path.join(args.save_dir, "flow10.png"), f_size=(sup_res_h, sup_res_w))

        pred_list = []
        cv2.imwrite(os.path.join(args.save_dir,'pred_0.png'), cv2.cvtColor(model.latent2image(pred_code[:1]), cv2.COLOR_BGR2RGB))
        cv2.imwrite(os.path.join(args.save_dir,'pred_%d.png' %args.Time), cv2.cvtColor(model.latent2image(pred_code[-1:]), cv2.COLOR_BGR2RGB))

        for i in range(1,args.Time): 
            time = i/args.Time
            target_pose = torch.tensor(trajectory[i])
            out_soft12 = splat_flow(args.source_image, input_code[:1], pts3d[0], conf[0], [extrinsics[0],target_pose], intrinsics, sup_res_h, sup_res_w)
            out_soft21 = splat_flow(args.source_image, input_code[1:], pts3d[1], conf[1], [extrinsics[1],target_pose], intrinsics, sup_res_h, sup_res_w)
            mask =  out_soft12[:,-1] * out_soft21[:,-1]
            out_soft = \
                ((1. - time) * out_soft12[0,:-1] + time * out_soft21[0,:-1]) * mask \
                + out_soft21[0,:-1] * torch.clamp((out_soft21[:,-1]-mask),min=0) \
                + out_soft12[0,:-1] * torch.clamp((out_soft12[:,-1]-mask),min=0) \
                + ((1. - time) * input_code[0,:-1] + time * input_code[1,:-1]) * (1-torch.clamp((out_soft12[:,-1] + out_soft21[:,-1]),max=1))
            pred_list.append(out_soft.unsqueeze(0))
            cv2.imwrite(os.path.join(args.save_dir,'pred_%d.png'  %i), cv2.cvtColor(model.latent2image(out_soft.unsqueeze(0)), cv2.COLOR_BGR2RGB))
        pred_list = torch.cat(pred_list,dim=0)
        
        torch.save(pred_list, os.path.join(args.save_dir,"pred_list.pt"))
        return flow1to2, flow2to1