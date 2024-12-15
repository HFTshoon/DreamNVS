import os
import json
import tempfile
import numpy as np

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.model import AsymmetricCroCo3DStereo

from dust3r.inference import inference
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def recon_3d_mast3r(img_dir, extrinsics=None, intrinsics=None):
    model_path = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    min_conf_thr = 1.5

    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    
    img_list = []
    for file_name in os.listdir(img_dir):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):
            img_list.append(file_name)
    img_list.sort()
    img_path_list = [os.path.join(img_dir,img_name) for img_name in img_list]
    print(img_path_list)
    images = load_images(img_path_list, size=512)
    output = inference([tuple(images)], model, device, batch_size=1)

    # pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    # output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    if extrinsics is not None:
        scene.preset_pose(extrinsics)

    if intrinsics is not None:
        scene.preset_focal(intrinsics["focals"])
        scene.preset_principal_point(intrinsics["principal_points"])
    else:
        scene.preset_principal_point_zero()

    scene.min_conf_thr = min_conf_thr
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    imgs = scene.imgs                       # (2, H, W, 3)
    focals = scene.get_focals()             # (2, 1)
    poses = scene.get_im_poses()            # (2, 4, 4)
    pts3d = scene.get_pts3d()               # 2 X (H, W, 3)
    pps = scene.get_principal_points()      # (2, 2)
    confidence_masks = scene.get_masks()    # 2 X (H, W)
    conf = scene.get_conf()                # 2 X (H, W)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    if extrinsics is None:
        extrinsics_path = os.path.join(img_dir, 'extrinsics.json')
        extrinsics = {
            "extrinsics": poses.detach().cpu().numpy().tolist()
        }
        with open(extrinsics_path, 'w') as f:
            json.dump(extrinsics, f)

    if intrinsics is None:
        intrinsics_path = os.path.join(img_dir, 'intrinsics.json')
        intrinsics = {
            "focals": [focal[0] for focal in focals.detach().cpu().numpy().tolist()],
            "principal_points": pps.detach().cpu().numpy().tolist()
        }
        with open(intrinsics_path, 'w') as f:
            json.dump(intrinsics, f)

    return imgs, focals, poses, pps, pts3d, conf, confidence_masks, matches_im0, matches_im1

def recon_3d_dust3r(img_dir, extrinsics=None, intrinsics=None):
    model_path = 'checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    img_path_list = [os.path.join(img_dir,'0.png'),os.path.join(img_dir,'1.png')]
    images = load_images(img_path_list, size=512)
    output = inference([tuple(images)], model, device, batch_size=1)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    if extrinsics is not None:
        scene.preset_pose(extrinsics)

    if intrinsics is not None:
        scene.preset_focal(intrinsics["focals"])
        scene.preset_principal_point(intrinsics["principal_points"])
    else:
        scene.preset_principal_point_zero()

    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    imgs = scene.imgs                       # (2, H, W, 3)
    focals = scene.get_focals()             # (2, 1)
    poses = scene.get_im_poses()            # (2, 4, 4)
    pts3d = scene.get_pts3d()               # 2 X (H, W, 3)
    pps = scene.get_principal_points()      # (2, 2)
    confidence_masks = scene.get_masks()    # 2 X (H, W)
    conf = scene.get_conf()                # 2 X (H, W)

    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    return imgs, focals, poses, pps, pts3d, confidence_masks, matches_im0, matches_im1