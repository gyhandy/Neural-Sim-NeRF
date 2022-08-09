import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import torch
from utils.gumble import *  # use gumble sampling
from datetime import datetime

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, torch.cos(phi), -torch.sin(phi), 0],
    [0, torch.sin(phi), torch.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_phi_diff(phi):
    temp = torch.Tensor([
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1]]).float().requires_grad_()
    rot_phi = temp.clone()

    rot_phi[1, 1] = temp[1, 1] * torch.cos(phi)
    rot_phi[1, 2] = temp[1, 2] * (-torch.sin(phi))
    rot_phi[2, 1] = temp[2, 1] * torch.sin(phi)
    rot_phi[2, 2] = temp[2, 2] * torch.cos(phi)
    return rot_phi


rot_theta = lambda th: torch.Tensor([
    [torch.cos(th), 0, -torch.sin(th), 0],
    [0, 1, 0, 0],
    [torch.sin(th), 0, torch.cos(th), 0],
    [0, 0, 0, 1]]).float()


def rot_theta_diff(theta):
    temp = torch.Tensor([
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 1]]).float().requires_grad_()
    rot_theta = temp.clone()

    rot_theta[0, 0] = temp[0, 0] * torch.cos(theta)
    rot_theta[0, 2] = temp[0, 2] * (-torch.sin(theta))
    rot_theta[2, 0] = temp[2, 0] * torch.sin(theta)
    rot_theta[2, 2] = temp[2, 2] * torch.cos(theta)
    return rot_theta


def pose_spherical(theta, phi, radius):
    phi.requires_grad_()
    radius = torch.Tensor([radius]).requires_grad_()
    c2w = trans_t(radius)
    # c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_phi_diff(phi / 180. * np.pi) @ c2w
    # c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_theta_diff(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


# pose_spherical_nograd

rot_phi_nograd = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta_nograd = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical_nograd(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi_nograd(phi / 180. * np.pi) @ c2w
    c2w = rot_theta_nograd(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_LINEMOD_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for idx_test, frame in enumerate(meta['frames'][::skip]):
            fname = frame['file_path']
            if s == 'test':
                print(f"{idx_test}th test frame: {fname}")
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    focal = float(meta['frames'][0]['intrinsic_matrix'][0][0])
    K = meta['frames'][0]['intrinsic_matrix']
    print(f"Focal: {focal}")
    '''radius should be in the range of training set, blenderproc use 1~1.24, consistency'''
    render_poses = torch.stack([pose_spherical(angle, -30.0, 1.01) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    # render_poses = torch.stack([pose_spherical(angle, -50.0, 1.01) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    # render_poses = torch.stack([pose_spherical(angle, -10.0, 1.01) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    # render_poses = torch.stack([pose_spherical(angle, 10, 1.01) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    if half_res:
        scale_factor = 2
        # change K aswell
        K[0] = [i / scale_factor for i in K[0]]
        K[1] = [i / scale_factor for i in K[1]]
        H = H // scale_factor
        W = W // scale_factor
        focal = focal / scale_factor

        # imgs_half_res = np.zeros((imgs.shape[0], H, W, 3)) # rgb
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))  # rgba
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # near = np.floor(min(metas['train']['near'], metas['test']['near']))
    # far = np.ceil(max(metas['train']['far'], metas['test']['far']))
    near = min(metas['train']['near'], metas['test']['near']) - 1  # enlarge the gap between near and far
    far = max(metas['train']['far'], metas['test']['far']) + 1
    return imgs, poses, render_poses, [H, W, focal], K, i_split, near, far


def load_data_param(basedir, half_res=False, testskip=1):
    splits = ['train']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'nerf_traindata_info.json'), 'r') as fp:
            metas[s] = json.load(fp)
    sample = metas['train']['frames'][0]
    # H = W = 400 # easy for debug, if dataset change, this value should change as well
    H = metas['train']['H']
    W = metas['train']['W']
    focal = float(sample['intrinsic_matrix'][0][0])
    K = sample['intrinsic_matrix']
    print(f"Focal: {focal}")

    # if half_res:
    #     H = H // 2
    #     W = W // 2
    #     focal = focal / 2.
    #     # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    if half_res:
        scale_factor = 4
        # change K aswell
        K[0] = [i / scale_factor for i in K[0]]
        K[1] = [i / scale_factor for i in K[1]]
        H = H // scale_factor
        W = W // scale_factor
        focal = focal / scale_factor
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    # near = np.floor(min(metas['train']['near'], metas['test']['near']))
    # far = np.ceil(max(metas['train']['far'], metas['test']['far']))
    near = metas['train']['near'] - 0.5  # enlarge the gap between near and far
    far = metas['train']['far'] + 0.5
    return [H, W, focal], K, near, far


def sample_pose(categorical_prob, num_K, gumble_T, sample_log):
    '''
    Input
        categorical_prob: categorical distribution probability e.g., [0, 0, 0.5, 0, 0.5, 0]
        num_K: number of sample images from given distribution
    Output
        render_poses: [num_K, 4, 4]
    '''
    # sample phi, theta is fixed
    # real distribution setting
    n_cats = len(categorical_prob)
    n_samples = num_K
    cats = np.arange(n_cats) # 1 represent 0 degree
    degrees = torch.Tensor([0, 45, 90, 135, 180, 225, 270, 315]) + 22.5 # can change with variables
    # degrees = torch.Tensor([0, 45, 90, 135]) + 22.5  # can change with variables
    # degrees.requires_grad_() # unnecessary
    probs = categorical_prob
    logits = torch.log(probs)

    # load the ramdom sample from
    gumbel_noises = sample_log['gumbel_noises']
    uniform_noises = sample_log['uniform_noises']
    thetas = sample_log['thetas']

    # sample with  Gumble softmax
    differentiable_samples_1 = []
    for n_samp in range(n_samples):
        pose = differentiable_sample(logits, degrees, gumbel_noises[n_samp], gumble_T) # 0.1 is the degree of gumbel sample
        differentiable_samples_1.append(pose)
    differentiable_samples_1_uniform = []
    for i, sample in enumerate(differentiable_samples_1):
        uniform_noise = uniform_noises[i]
        pose_u = sample - 22.5 + 45 * uniform_noise # U(sample-a/2,sample+a/2) = (sample-a/2) + a * U(0,1)
        differentiable_samples_1_uniform.append(pose_u)



    '''radius should be in the range of training set, blenderproc use 1~1.24, consistency'''
    # theta is a random variable, phi is optimized variable
    render_poses = []
    for i, phi in enumerate(differentiable_samples_1_uniform):
        theta = torch.Tensor([thetas[i]])
        render_poses.append(pose_spherical(theta, phi - 180, 1.01))

    render_poses = torch.stack(render_poses, 0)  # (0~360) to (-180, 180)
    return render_poses


def sample_pose_nograd(categorical_prob, num_K, gumble_T):
    '''
    Input
        categorical_prob: categorical distribution probability e.g., [0, 0, 0.5, 0, 0.5, 0]
        num_K: number of sample images from given distribution
    Output
        render_poses: [num_K, 4, 4]
    '''


    # sample phi, theta is fixed
    # real distribution setting
    n_cats = len(categorical_prob)
    n_samples = num_K
    cats = np.arange(n_cats) # 1 represent 0 degree
    degrees = np.array([0, 45, 90, 135, 180, 225, 270, 315]) + 22.5 # can change with variables
    probs = categorical_prob
    logits = np.log(probs)


    # sample with  Gumble softmax
    differentiable_samples_1 = []
    gumbel_noises = []
    np.random.seed(datetime.now().second) # change the randomness
    for n_samp in range(n_samples):
        pose, gumbel_noise = differentiable_sample_nograd(logits, degrees, gumble_T) # 0.1 is the degree of gumbel sample
        differentiable_samples_1.append(pose)
        gumbel_noises.append(gumbel_noise.tolist()) # keep the sample noise for the second time gradient computation
    # differentiable_samples_1 = [differentiable_sample_nograd(logits, degrees, 0.1) for _ in range(n_samples)]
    # differentiable_samples_1_uniform = sample_uniform(differentiable_samples_1)
    differentiable_samples_1_uniform = []
    uniform_noises = []
    for sample in differentiable_samples_1:
        uniform_noise = np.random.uniform(0, 1)
        pose_u = sample - 22.5 + 45 * uniform_noise # U(sample-a/2,sample+a/2) = (sample-a/2) + a * U(0,1)
        differentiable_samples_1_uniform.append(pose_u)
        uniform_noises.append(uniform_noise)

    thetas = []
    render_poses = []
    for phi in differentiable_samples_1_uniform:
        # theta = np.random.uniform(-180, 180)
        theta = np.random.uniform(85, 95)  # limit range to accelerate the optimization
        render_poses.append(pose_spherical_nograd(theta, phi - 180, 1.01))
        thetas.append(theta)

    render_poses = torch.stack(render_poses, 0)  # (0~360) to (-180, 180)
    sample_log = {}
    sample_log['gumbel_noises'] = gumbel_noises
    sample_log['uniform_noises'] = uniform_noises
    sample_log['thetas'] = thetas
    return render_poses, sample_log


def sample_pose_nograd_gaussian(pose_mean_nograd, pose_var_nograd, num_K):
    '''
    Input
        categorical_prob: categorical distribution probability e.g., [0, 0, 0.5, 0, 0.5, 0]
        num_K: number of sample images from given distribution
    Output
        render_poses: [num_K, 4, 4]
    '''
    # sample phi, theta is fixed
    n_samples = num_K
    phis = np.random.normal(pose_mean_nograd, pose_var_nograd, n_samples)
    '''radius should be in the range of training set, blenderproc use 1~1.24, consistency'''
    # theta is a random variable, phi is optimized variable
    render_poses = []
    for phi in phis:
        if phi > 360:
            phi = phi % 360
        elif phi < 0:
            phi = phi % 360 + 360
        # theta = np.random.uniform(-180, 180) # totally free
        theta = np.random.uniform(85, 95) # limit range to accelerate the optimization
        render_poses.append(pose_spherical_nograd(theta, phi - 180, 1.01)) # (theta, phi - 180, zoom)

    render_poses = torch.stack(render_poses, 0)  # (0~360) to (-180, 180)
    return render_poses, phis