import os
import tempfile

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs

def predict_3d(img_dir):
    model_path = 'checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    
    img_path_list = [os.path.join(img_dir,'0.png'),os.path.join(img_dir,'1.png')]
    images = load_images(img_path_list, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    cache_dir = tempfile.mkdtemp()
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(images, pairs, cache_dir, model, device=device)

    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d, depths, confidences = scene.get_dense_pts3d()

    breakpoint()

    # # at this stage, you have the raw dust3r predictions
    # view1, pred1 = output['view1'], output['pred1']
    # view2, pred2 = output['view2'], output['pred2']

    # desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # # find 2D-2D matches between the two images
    # matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
    #                                                device=device, dist='dot', block_size=2**13)

    # # ignore small border around the edge
    # H0, W0 = view1['true_shape'][0]
    # valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
    #     matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    # H1, W1 = view2['true_shape'][0]
    # valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
    #     matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    # valid_matches = valid_matches_im0 & valid_matches_im1
    # matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    # loss = scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)
    # scene.min_conf_thr = 1.5

    # # retrieve useful values from scene:
    # imgs = scene.imgs
    # focals = scene.get_focals()
    # poses = scene.get_im_poses()
    # pts3d = scene.get_pts3d()
    # confidence_masks = scene.get_masks()

    # breakpoint()

    # # visualize reconstruction
    # scene.show()

    # breakpoint()
    # pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
    # pts3d_im1 = pred2['pts3d'].squeeze(0).detach().cpu().numpy()
    # conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy()
    # conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()
    # desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    # desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()
    return 