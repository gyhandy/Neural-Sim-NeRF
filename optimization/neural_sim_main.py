# End to end pipeline to train a bi-level optimization to use Nerf Autosimulate
'''
Forward optimization path
For number of iterations do:
    step 1 NeRF generate K = 100 images as D_train given /psi = categorical distribution of pose
    step 2 Fine-tune Faster RCNN model with D_train
    step 3 Compute d L_val / d \psi and update \psi
'''
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

import cv2
import matplotlib
import imageio
from PIL import Image

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import HookBase, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
import pycocotools
from utils.run_nerf_noscale import *
from utils.load_LINEMOD_noscale import *
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.events import TensorboardXWriter, EventStorage
from utils import dataset_mapper
from utils.defaults import *
class NeRF:
    def __init__(self, args): # create nerf
        self.param = args
        if not os.path.exists(os.path.join(self.param.basedir, self.param.expname)):
            os.makedirs(os.path.join(self.param.basedir, self.param.expname))  # this is the place to save Nerf pretrained model
        # Load data in LINEMOD format
        self.K = None
        self.hwf, self.K, self.near, self.far = load_data_param(self.param.datadir, self.param.half_res,
                                                                                    self.param.testskip)
        print(f'Loaded LINEMOD, images shape: {self.hwf[:2]}, hwf: {self.hwf}, K: {self.K}')
        print(f'[CHECK HERE] near: {self.near}, far: {self.far}.')


        # Cast intrinsics to right types
        H, W, focal = self.hwf
        H, W = int(H), int(W)
        self.hwf = [H, W, focal]

        if self.K is None:
            self.K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ])
        # use the pretrained nerf model
        self.param.ft_path = os.path.join(self.param.basedir, 'nerf_models', 'ycbvid{}.tar'.format(self.param.object_id))
        self.render_kwargs_train, self.render_kwargs_test, self.start, self.grad_vars, self.optimizer = create_nerf(self.param)


    def farward_hook(module, inp, outp):
        fmap_block['input'] = inp
        fmap_block['output'] = outp

    def backward_hook(module, grad_in, grad_out):
        grad_block['grad_in'] = grad_in
        grad_block['grad_out'] = grad_out

    def render_images(self, q_psi_categorical_prob, Optimiation_parameter):
        '''
        Input: render parameters
        Output: rendered images and saved path
        '''
        args = self.param
        torch_softmax = torch.nn.Softmax(dim=0)
        temperature = 0.25 # need adjust
        categorical_prob = torch_softmax(q_psi_categorical_prob / temperature)  # still tensor
        # sample poses with no gradient
        categorical_prob = np.array(categorical_prob, dtype=np.float16)
        # categorical_prob = torch.Tensor([0.4, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05, 0.05]).requires_grad_() # initial pose np.array([0, 45, 90, 135, 180, 225, 270, 315]) + 22.5
        num_K = Optimiation_parameter.n_samples_K
        render_poses, sample_log = sample_pose_nograd(categorical_prob, num_K, Optimiation_parameter.gumble_T)

        # Create log dir and copy the config file
        basedir = self.param.basedir
        expname = self.param.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(self.param)):
                attr = getattr(self.param, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if self.param.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.param.config, 'r').read())



        self.bds_dict = {
            'near': self.near,
            'far': self.far,
        }
        self.render_kwargs_train.update(self.bds_dict)
        self.render_kwargs_test.update(self.bds_dict)

        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)

        # Short circuit if only rendering out from trained model
        print('RENDER ONLY')
        # with torch.no_grad():
        images = None
        testsavedir = os.path.join(basedir, expname,
                                   'renderonly_{}'.format('test' if args.render_test else 'path'))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        render_path(categorical_prob, render_poses, self.hwf, self.K, self.param.chunk, self.render_kwargs_test, gt_imgs=images,
                              savedir=testsavedir, object_id=self.param.object_id, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

        return testsavedir, sample_log
    def render_images_grad(self, q_psi_categorical_prob, Optimiation_parameter, sample_log, grad_E):
        '''
        Input: render parameters
        Output: rendered images and saved path
        '''
        args = self.param
        torch_softmax = torch.nn.Softmax(dim=0)
        temperature = 0.25
        categorical_prob = torch_softmax(q_psi_categorical_prob / temperature)  # still tensor
        categorical_prob = categorical_prob.requires_grad_()
        # sample poses
        # categorical_prob = torch.Tensor([0.4, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05, 0.05]).requires_grad_() # initial pose np.array([0, 45, 90, 135, 180, 225, 270, 315]) + 22.5
        num_K = Optimiation_parameter.n_samples_K
        render_poses = sample_pose(categorical_prob, num_K, Optimiation_parameter.gumble_T, sample_log)

        # Create log dir and copy the config file
        basedir = self.param.basedir
        expname = self.param.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(self.param)):
                attr = getattr(self.param, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if self.param.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.param.config, 'r').read())



        self.bds_dict = {
            'near': self.near,
            'far': self.far,
        }
        self.render_kwargs_train.update(self.bds_dict)
        self.render_kwargs_test.update(self.bds_dict)

        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)
        # categorical_prob = categorical_prob.cuda()
        # Short circuit if only rendering out from trained model
        print('RENDER ONLY')
        # with torch.no_grad():
        images = None
        testsavedir = os.path.join(basedir, expname,
                                   'renderonly_{}'.format('test' if args.render_test else 'path'))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, dLdpsis = render_path_grad(categorical_prob, render_poses, self.hwf, self.K, self.param.chunk, grad_E, self.render_kwargs_test, gt_imgs=images,
                              savedir=testsavedir, object_id=self.param.object_id, render_factor=args.render_factor)
        # if you want to skip rendering step
        # rgbs = None
        print('Done rendering', testsavedir)
        # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

        return torch.mean(torch.stack(dLdpsis), 0) # average the gradient

    def train(self):
        args = self.param

        # Load data
        K = None
        if args.dataset_type == 'LINEMOD':
            images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
                                                                                        args.testskip)
            print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
            print(f'[CHECK HERE] near: {near}, far: {far}.')
            i_train, i_val, i_test = i_split

            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3]

        else:
            print('Unknown dataset type', args.dataset_type, 'exiting')
            return

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        if K is None:
            K = np.array([
                [focal, 0, 0.5 * W],
                [0, focal, 0.5 * H],
                [0, 0, 1]
            ])

        if args.render_test:
            render_poses = np.array(poses[i_test])

        # Create log dir and copy the config file
        basedir = args.basedir
        expname = args.expname
        os.makedirs(os.path.join(basedir, expname), exist_ok=True)
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())

        # Create nerf model
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
        global_step = start

        bds_dict = {
            'near': near,
            'far': far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        # Move testing data to GPU
        render_poses = torch.Tensor(render_poses).to(device)

        # Short circuit if only rendering out from trained model
        if args.render_only:
            print('RENDER ONLY')
            with torch.no_grad():
                if args.render_test:
                    # render_test switches to test poses
                    images = images[i_test]
                else:
                    # Default is smoother render_poses path
                    images = None

                testsavedir = os.path.join(basedir, expname,
                                           'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', render_poses.shape)

                rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                      savedir=testsavedir, render_factor=args.render_factor)
                print('Done rendering', testsavedir)
                imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

                return

        # Prepare raybatch tensor if batching random rays
        N_rand = args.N_rand
        use_batching = not args.no_batching
        if use_batching:
            # For random ray batching
            print('get rays')
            rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
            print('done, concats')
            rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
            rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
            rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
            rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
            rays_rgb = rays_rgb.astype(np.float32)
            print('shuffle rays')
            np.random.shuffle(rays_rgb)

            print('done')
            i_batch = 0

        # Move training data to GPU
        if use_batching:
            images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        if use_batching:
            rays_rgb = torch.Tensor(rays_rgb).to(device)

        N_iters = 200000 + 1
        print('Begin')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        # Summary writers
        # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

        start = start + 1
        for i in trange(start, N_iters):
            time0 = time.time()

            # Sample random ray batch
            if use_batching:
                # Random over all images
                batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
                batch = torch.transpose(batch, 0, 1)
                batch_rays, target_s = batch[:2], batch[2]

                i_batch += N_rand
                if i_batch >= rays_rgb.shape[0]:
                    print("Shuffle data after an epoch!")
                    rand_idx = torch.randperm(rays_rgb.shape[0])
                    rays_rgb = rays_rgb[rand_idx]
                    i_batch = 0

            else:
                # Random from one image
                img_i = np.random.choice(i_train)
                target = images[img_i]
                target = torch.Tensor(target).to(device)
                pose = poses[img_i, :3, :4]

                if N_rand is not None:
                    rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                    if i < args.precrop_iters:
                        dH = int(H // 2 * args.precrop_frac)
                        dW = int(W // 2 * args.precrop_frac)
                        coords = torch.stack(
                            torch.meshgrid(
                                torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                                torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                            ), -1)
                        if i == start:
                            print(
                                f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                    else:
                        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                             -1)  # (H, W, 2)

                    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                    select_coords = coords[select_inds].long()  # (N_rand, 2)
                    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    batch_rays = torch.stack([rays_o, rays_d], 0)
                    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True,
                                            **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

            dt = time.time() - time0
            # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
            #####           end            #####

            # Rest is logging
            if i % args.i_weights == 0:
                path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

            if i % args.i_video == 0 and i > 0:
                # Turn on testing mode
                with torch.no_grad():
                    rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

                # if args.use_viewdirs:
                #     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
                #     with torch.no_grad():
                #         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
                #     render_kwargs_test['c2w_staticcam'] = None
                #     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

            if i % args.i_testset == 0 and i > 0:
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[i_test].shape)
                with torch.no_grad():
                    render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                                gt_imgs=images[i_test], savedir=testsavedir)
                print('Saved test set')

            if i % args.i_print == 0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            """
                print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
                print('iter time {:.05f}'.format(dt))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                    tf.contrib.summary.scalar('loss', loss)
                    tf.contrib.summary.scalar('psnr', psnr)
                    tf.contrib.summary.histogram('tran', trans)
                    if args.N_importance > 0:
                        tf.contrib.summary.scalar('psnr0', psnr0)


                if i%args.i_img==0:

                    # Log a rendered validation view to Tensorboard
                    img_i=np.random.choice(i_val)
                    target = images[img_i]
                    pose = poses[img_i, :3,:4]
                    with torch.no_grad():
                        rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose,
                                                            **render_kwargs_test)

                    psnr = mse2psnr(img2mse(rgb, target))

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                        tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                        tf.contrib.summary.image('disp', disp[tf.newaxis,...,tf.newaxis])
                        tf.contrib.summary.image('acc', acc[tf.newaxis,...,tf.newaxis])

                        tf.contrib.summary.scalar('psnr_holdout', psnr)
                        tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])


                    if args.N_importance > 0:

                        with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                            tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                            tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis,...,tf.newaxis])
                            tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis,...,tf.newaxis])
            """

            global_step += 1
class Writer(HookBase):
    def __init__(self,write_iter):
        self._debug_info = {}
        self.write_iter = write_iter
        self.writer = TensorboardXWriter(...)
    def before_step(self):
        self._debug_info = {}
    def after_step(self):
        loss = self._debug_info['loss']
        self.writer.write(loss)
class ComputeGradHook(HookBase):
    def after_step(self):
        if self.trainer.iter % 5 == 0:
          print(f"finish iteration {self.trainer.iter}!")
    def after_train(self):
        print(f"Compute d/dI (dL/dtheta)")
        """
        Implement the standard training logic described above.
        """
        assert self.trainer.model.training, "[SimpleTrainer] model was changed to eval mode!"
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        datas = next(self.trainer._data_loader_iter)
        # set image as variable with requires_grad_()
        for i, data in enumerate(datas):
            datas[i]['image'] = datas[i]['image'].type(torch.FloatTensor).requires_grad_()  # test
        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.trainer.model(datas)
        losses = sum(loss_dict.values())
        """
        Computer d/dI (dL/dtheta)
        """
        # dL/dtheta
        first_derivatives = []
        for layer_index, layer in enumerate(self.trainer.optimizer.param_groups): #
          if layer_index > 70: # only compute the last several layers
              # dL/dtheta # gradient for first layer params
              first_d = torch.autograd.grad(losses, layer['params'][0], create_graph=True,
                                                     retain_graph=True)
              first_derivatives.append(first_d)
        first_derivatives = tuple(i for i in first_derivatives)
        # first_derivatives = tuple(zip(*first_derivatives))[0]
        # d/dI (dL/dtheta) d vectorA / d vectorB grad_outputs has same dim as vectorA, output has same dim as vectorB
        grad_outputs = tuple(torch.ones_like(i) for i in first_derivatives) # weights to sum y
        second_derivative = torch.autograd.grad(first_derivatives, datas[0]['image'], grad_outputs=grad_outputs, retain_graph=True)
        # self.trainer.optimizer.zero_grad() # torch.autograd.grad will not influence the variable.grad
        # datas[0]['image'].grad = None
from shutil import copyfile

class Trainer(DefaultTrainer): # for detectron
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)
        # self.register_hooks([ComputeGradHook()]) # can use hook but now we compute gradient after the training
        self._data_loader_iter = iter(self.data_loader)
        # self.train_data_loader = build_detection_test_loader(cfg, 'train_dataset',
        #                                 mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.Resize((50, 50))])) # load annotation as well
        self.train_data_loader = build_detection_test_loader(cfg, 'train_dataset',
                                        mapper=dataset_mapper.DatasetMapper(cfg, is_train=True, augmentations=[]))
        self.train_loader_iter = iter(self.train_data_loader)
        self.val_data_loader = build_detection_test_loader(cfg, 'val_dataset',
                                        mapper=dataset_mapper.DatasetMapper(cfg, is_train=True, augmentations=[])) # do not need augmentation, keep origin
        self.val_loader_iter = iter(self.val_data_loader)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        # data[0]['image'] = data[0]['image'].type(torch.FloatTensor).requires_grad_() # set image as variable
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        # data[0]['image'].grad = None
        losses.backward()

        # self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

class Detector:
    def __init__(self, args):
        # detector initialization
        cfg = get_cfg()
        self.args = args
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("train_dataset",)
        cfg.DATASETS.TEST = ("val_dataset",)
        cfg.DATALOADER.NUM_WORKERS = 2
        if self.args.pretrain:
            cfg.MODEL.WEIGHTS = self.args.pretrain_weight  # use pretrained model with uniform distribution
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

        cfg.SOLVER.IMS_PER_BATCH = 8
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 50  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.SOLVER.WARMUP_ITERS = 10
        cfg.SOLVER.STEPS = []  # do not decay learning rate
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8  # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # number of classes
        cfg.MODEL.RETINANET.NUM_CLASSES = 6  # if use Retinanet change this
        # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
        cfg.MODEL.BACKBONE.FREEZE_AT = 6 # freeze the whole Resnet backbone
        cfg.OUTPUT_DIR = os.path.join(args.basedir, args.expname, 'detectron_output')
        # cfg.OUTPUT_DIR = os.path.join('./output', args.expname) # save dir of latest model
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        cfg['INPUT']['MASK_FORMAT'] = 'bitmask'  # the mask format is not polegen
        self.cfg = cfg

    def createCocoJSONFromSynthetics(self, train_info, test_info, train_json, test_json, output_dir):
        '''
        input the dataset information, output a dir:
        -output_dir
            --train
                -coffecan
                    -000000.png
                    -000001.png
                    ...
                -cheesebox
                    -000000.png
                    -000001.png
                    ...
            --test
                -000000.png
                -000001.png
                ...
            train_json.json
            test_json.json
        e.g.,
        train_info = {'coffecan': '/YCBV/ycbvid1_900/train',
                      'cheesebox': '/YCBV/ycbvid2_100/train',
                      'sugerbox':'/YCBV/ycbvid3_800/train',
                      'bowl': '/ycbvid13_1000/train'}
        '''
        for s in ['train', 'test']:
            info = train_info if s == 'train' else test_info
            image_id = 1
            annotation_id = 1
            categories_list = []  # for the whole train/test set json
            images = []  # for the whole train/test set json
            annotations = []  # for the whole train/test set json
            for class_index, (class_name, class_img_path) in enumerate(
                    info.items()):  # go over each class and their images
                # save the category information
                category_id = class_index + 1
                categories_dict = {}
                categories_dict['supercategory'] = 'ycbv'
                categories_dict['id'] = category_id
                categories_dict['name'] = class_name
                print(
                    'processing class_index: {}, class_name: {}, class_img_path: {}, from image_id: {}, annotation_id: {}'.format(
                        class_index, class_name, class_img_path, image_id, annotation_id))
                file_list = [f for f in os.listdir(class_img_path) if os.path.splitext(f)[1] == ".png"]

                for f in file_list:  # for each image

                    # image
                    file_path = os.path.join(class_img_path, f)
                    # copy image to target dir
                    target_file_path = os.path.join(output_dir, s, class_name,
                                                    f)  # output_dir/train/cheesebox/000000.png
                    if not os.path.exists(os.path.join(output_dir, s, class_name)):
                        os.makedirs(os.path.join(output_dir, s, class_name))
                    copyfile(file_path, target_file_path)
                    # load image and compute annotation
                    bboxs, mask, height, width = self.get_annotation(file_path)
                    new_img = {}
                    new_img["license"] = 0
                    new_img["file_name"] = os.path.join(s, class_name, f)  # relative path
                    new_img["width"] = width
                    new_img["height"] = height
                    new_img["id"] = image_id
                    images.append(new_img)

                    if bboxs.shape[0] != 1:  # if have multiple objects, choose the largest one
                        bboxs = [bboxs[np.argmax(bboxs[:, -2] * bboxs[:, -1], axis=0)]]

                    for bbox in bboxs:
                        annotation = {
                            'iscrowd': False,
                            'image_id': image_id,
                            'category_id': category_id,
                            'id': annotation_id,
                            'bbox': [int(x) for x in list(bbox)],
                            "bbox_mode": BoxMode.XYWH_ABS,  # not sure if it is useful
                            'area': int(list(bbox)[2]) * int(list(bbox)[3])
                        }

                        annotations.append(annotation)
                        annotation_id += 1

                    image_id += 1
                categories_list.append(categories_dict)

            print("saving annotations to coco as json ")
            ### create COCO JSON annotations
            dataset_name = output_dir.split('/')[-1]
            my_dict = {}
            my_dict["info"] = {"description": dataset_name, "url": "", "version": "1", "year": 2020,
                               "contributor": "MSR CV Group", "date_created": "06/23/2021"}
            my_dict["licenses"] = [{"url": "", "id": 0, "name": "License"}]
            my_dict["images"] = images
            my_dict["categories"] = categories_list
            my_dict["annotations"] = annotations

            # TODO: specify coco file locaiton
            json_name = train_json if s == 'train' else test_json
            output_file_path = os.path.join(output_dir, json_name)
            with open(output_file_path, 'w+') as json_file:
                json_file.write(json.dumps(my_dict))

            print(">> complete. find coco json here: ", output_file_path)
            print("last annotation id: ", annotation_id)
            print("last image_id: ", image_id)
    def create_dataset(self, Nerf_imgs_savedir):
        '''
        Input: D_train
        '''
        with open(self.args.train_val_path_info) as f:
            dataset_info = json.load(f)
        train_info = dataset_info['train_info']
        test_info = dataset_info['test_info'][self.args.test_distribution]
        # update root folder
        for cate, path in train_info.items():
            if cate == self.args.object_id: # optimized class
                train_info[cate] = os.path.join(Nerf_imgs_savedir, self.args.object_id)
            else: # background class
                train_info[cate] = os.path.join(self.args.basedir, train_info[cate])
        for cate, path in test_info.items():
            test_info[cate] = os.path.join(self.args.basedir, test_info[cate])
        output_dir = os.path.join(Nerf_imgs_savedir.replace('/renderonly_path', '/'), 'D_train')

        '''save json'''
        self.createCocoJSONFromSynthetics(train_info, test_info, 'ycbv_train.json', 'ycbv_test.json', output_dir)
        train_basedir = output_dir
        test_basedir = output_dir
        # test_basedir = '/home/t-yunhaoge/PycharmProject/BlenderProc-old/examples/YCBV/ycbvid1215-1000-r115-allpose/coco_data'
        # have coco format json
        # train_json_path = os.path.join(train_basedir, 'coco_annotations.json')
        train_json_path = os.path.join(train_basedir, 'ycbv_train.json')
        test_json_path = os.path.join(test_basedir, 'ycbv_test.json')
        # test_json_path = os.path.join(test_basedir, 'coco_annotations.json')

        for d in ["train", "val"]:
            if d + "_dataset" in DatasetCatalog.list():
                DatasetCatalog.remove(d + "_dataset")
        # Use COCO register
        register_coco_instances("train_dataset", {}, train_json_path,
                                train_basedir)  # they need path to img dir because path can be relative path
        register_coco_instances("val_dataset", {}, test_json_path, test_basedir)
        # register_coco_instances("val_dataset", {}, test_json_path, test_basedir)

        self.metadata = MetadataCatalog.get("val_dataset")

        # change model output dimension based on dataset
        class_number = len(train_info.keys())
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_number
        self.cfg.MODEL.RETINANET.NUM_CLASSES = class_number
        '''visualize dataset'''
        # dataset_dicts = DatasetCatalog.get("val_dataset")
        # for d in random.sample(dataset_dicts, 15):
        #     img = cv2.imread(d["file_name"])
        #     visualizer = Visualizer(img[:, :, ::-1], metadata=self.metadata, scale=0.5)
        #     out = visualizer.draw_dataset_dict(d)
        #     plt.imshow(out.get_image()[:, :, ::-1])
        #     plt.show()





    def find_bbox(self, mask):
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        stats = stats[stats[:, 4].argsort()]
        return stats[:-1]

    def get_annotation(self, img_path):
        img = cv2.imread(img_path)  # for get mask and bbox
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        height, width = gray_img.shape
        (_, mask) = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)  # get mask
        bboxs = self.find_bbox(mask)[:, :-1]
        return bboxs, mask, height, width

    def get_ycbv_dicts(self, ycbv_basedir):

        dataset_dicts = []
        cates = os.listdir(ycbv_basedir)
        cates2id = {'1':1, '2':2, '15':15}
        image_index = 0
        for cate in cates:
            imgs = os.listdir(os.path.join(ycbv_basedir, cate)) # for each category
            for idx, img in enumerate(imgs):
                if '.png' in img:  # image
                    record = {}

                    filename = os.path.join(ycbv_basedir, cate, img)
                    bboxs, mask, height, width = self.get_annotation(filename)
                    if bboxs.shape[0] > 1:  # if have multiple objects, choose the largest one
                        bboxs = [bboxs[np.argmax(bboxs[:, -2] * bboxs[:, -1], axis=0)]]
                    record["file_name"] = filename
                    record["image_id"] = idx + image_index
                    record["height"] = height
                    record["width"] = width

                    objs = []  # for annotation, list[dict]
                    for bbox in bboxs:
                        obj = {
                            "bbox": list(bbox),
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "segmentation": pycocotools.mask.encode(np.asarray(mask, order="F")),
                            "category_id": cates2id[cate],
                        }
                        objs.append(obj)
                    record["annotations"] = objs
                    dataset_dicts.append(record)
                    image_index += 1
        return dataset_dicts

    def train(self, iteration):
        '''train RetinaNet'''
        # if iteration == 0:
        # self.trainer = DefaultTrainer(self.cfg)
        self.trainer = Trainer(self.cfg)
        # self.trainer.register_hooks([HelloHook()])
        if iteration > 0: # start from second iteration
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, 'model_final.pth')
        self.trainer.resume_or_load(resume=False)
        print('Detectron model:', self.cfg.MODEL.WEIGHTS)
        # comment when do not want to discuss train
        self.trainer.train()

    def inference(self, epoch):
        evaluator = COCOEvaluator("val_dataset", ("bbox",), False, output_dir = self.cfg.OUTPUT_DIR)
        val_loader = self.trainer.val_data_loader
        result = inference_on_dataset(self.trainer.model, val_loader, evaluator)
        with open(os.path.join(self.cfg.OUTPUT_DIR, 'save_result.txt'), 'a', encoding='utf-8') as f:
            f.write('epoch: {}'.format(epoch) + str(result['bbox']))
            f.write('\n')

    def compute_grad_E(self, inverse_hvp):
        '''
        inverse_hvp is the weight to compute d/dI (dL/dtheta)
        '''
        print('computing Grad_E')
        print(f"Compute d/dI (dL/dtheta)")
        """
        Implement the standard training logic described above.
        """
        assert self.trainer.model.training, "[SimpleTrainer] model was changed to eval mode!"
        self.trainer.optimizer.zero_grad()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        grad_Es = []
        for i, datas in enumerate(tqdm(self.trainer.train_data_loader)):
        # datas = next(self.trainer._data_loader_iter)
        # set image as variable with requires_grad_()
        # for i, data in enumerate(datas):
            if datas[0]['file_name'].split('/')[-2] == self.args.object_id: # only compute optimized class
                image_index = int(datas[0]['file_name'].split('/')[-1].split('.')[0])
                if image_index < 100: # can release the constraint
                    grad_E = {} # log the imaga name and gradient
                    print('computing image {}'.format(i))
                    grad_E['image_index'] = image_index# with

                    datas[0]['image'] = datas[0]['image'].type(torch.FloatTensor).requires_grad_()  # test
                    """
                    If you want to do something with the losses, you can wrap the model.
                    """
                    with EventStorage() as storage:
                        loss_dict = self.trainer.model(datas)
                    losses = sum(loss_dict.values())

                    """
                    Computer d/dI (dL/dtheta)
                    """
                    # dL/dtheta
                    first_derivatives = []
                    for layer_index, layer in enumerate(self.trainer.optimizer.param_groups): #
                      if layer_index >= 0: # only compute the last several layers
                          # dL/dtheta # gradient for first layer params
                          # first_d = torch.autograd.grad(losses, layer['params'][0], create_graph=True,
                          #                                        retain_graph=True)
                          first_d = torch.autograd.grad(losses, layer['params'][0], create_graph=True)
                          first_derivatives.append(first_d[0])

                    first_derivatives = tuple(i for i in first_derivatives)
                    # d/dI (dL/dtheta) d vectorA / d vectorB grad_outputs has same dim as vectorA, output has same dim as vectorB
                    # grad_outputs = tuple(torch.ones_like(i) for i in first_derivatives) # weights to sum y
                    # second_derivative = torch.autograd.grad(first_derivatives, datas[0]['image'], grad_outputs=inverse_hvp, retain_graph=True)
                    second_derivative = torch.autograd.grad(first_derivatives, datas[0]['image'], grad_outputs=inverse_hvp)
                    grad_E['grad_E'] = second_derivative
                    grad_Es.append(grad_E)
        grad_Es.sort(key= lambda x: x['image_index'])
        torch.cuda.empty_cache() # release the memory
        return grad_Es
    def compute_inverse_hvp(self, criterion=None, device='0', weight_decay=0.01, approx_type='cg',
                        approx_params={'scale':25, 'recursion_depth':5000, 'damping':0, 'batch_size':1, 'num_samples':10},
                        force_refresh=True, test_description=None, X=None, Y=None, cg_max_iter=0, stoc_hessian=True, gauss_newton=0):
        '''
        This part only related to detector and D_train, D_val
        input model, D_train, D_val
        output inverse_hvp
        '''

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val()

        print('Norm of test gradient: %s' % np.linalg.norm(
            np.concatenate([x.flatten().cpu() for x in test_grad_loss_no_reg_val])))

        start_time = time.time()
        if cg_max_iter == -1:
            return test_grad_loss_no_reg_val
        if cg_max_iter == -2:
            return [torch.ones_like(elmn) for elmn in test_grad_loss_no_reg_val]

        inverse_hvp = self.get_inverse_hvp(test_grad_loss_no_reg_val,
                                      criterion,
                                      device, approx_type, approx_params, weight_decay, cg_max_iter, stoc_hessian,
                                      gauss_newton)
        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)
        return inverse_hvp
    def get_test_grad_loss_no_reg_val(self):
        print(f"Compute dL_val/dtheta")
        """
        Implement the standard training logic described above.
        """
        assert self.trainer.model.training, "[SimpleTrainer] model was changed to eval mode!"
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        for i_iter, datas in enumerate(tqdm(self.trainer.val_loader_iter)): # go over all test images
        # for i_iter, datas in enumerate(tqdm(self.trainer._data_loader_iter)):
        # datas = next(self.trainer.trainer._data_loader_iter)
        #     if i_iter > 10: # go over 10 images
        #         break
            # set image as variable with requires_grad_()
            for i, data in enumerate(datas):
                datas[i]['image'] = datas[i]['image'].type(torch.FloatTensor).requires_grad_()  # test
            """
            If you want to do something with the losses, you can wrap the model.
            """
            with EventStorage() as storage:
                loss_dict = self.trainer.model(datas)
            losses = sum(loss_dict.values())
            if not torch.isfinite(losses):
                print('WARNING: non-finite loss, ending training ', losses)
                return
            """
            Computer (dL_val/dtheta)
            """
            # dL_val/dtheta
            losses.backward() # keep accumulate the gradient
        test_grad = []
        for layer_index, layer in enumerate(self.trainer.optimizer.param_groups):  #
            if layer_index >= 0:  # only compute the last several layers
                # dL/dtheta # gradient for first layer params
                test_grad.append(layer['params'][0].grad.clone()) # clone will cut the link of _zero_grad()
        test_grad = tuple(i for i in test_grad)
        torch.cuda.empty_cache() # release the memory
        return test_grad

    def get_inverse_hvp(self, v, criterion, device, approx_type='lissa', approx_params=None, weight_decay =0.01,
                        cg_max_iter=10, stoc_hessian=True, gauss_newton=0, verbose=False):
        print('computing inverse hessian')
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return get_inverse_hvp_lissa(model, train_generator, v, criterion, **approx_params)
        elif approx_type == 'cg':
            if cg_max_iter == 0:
                return self.minibatch_hessian_vector_val(v, stoc_hessian, gauss_newton, device)
            if cg_max_iter == -3:
                return [2 * a - b for (a, b) in zip(v, minibatch_hessian_vector_val(model, train_generator, v,
                                                                                    stoc_hessian, gauss_newton,
                                                                                    device))]
            else:
                return get_inverse_hvp_cg(model, train_generator, validation_generator, v, weight_decay, cg_max_iter,
                                          stoc_hessian, gauss_newton, device, verbose)
    def minibatch_hessian_vector_val(self, v, stoc_hessian, gauss_newton, device, criterion = nn.CrossEntropyLoss()):
        damping = 1e-2
        hessian_vector_val = None
        for i_iter, datas in enumerate(tqdm(self.trainer._data_loader_iter)): # use only one batch to represent the whole dataset
        # datas = next(self.trainer.trainer._data_loader_iter)
        #     if i_iter > 10: # go over 10 images
        #         break
            hessian_vector_val_temp = self.hessian_vector_product(datas, v, gauss_newton)

            if hessian_vector_val is None:
                hessian_vector_val = [b for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + b for (a, b) in zip(hessian_vector_val, hessian_vector_val_temp)]

            ###################################################
            ########### DOING ONLY FOR ONE BATCH ##############
            ###################################################
            if stoc_hessian:
                break

        hessian_vector_val = [a / float(i_iter + 1) + damping * b for (a, b) in
                              zip(hessian_vector_val, v)]  # why adding them? update the old v = v-damping dv

        return hessian_vector_val
    def hessian_vector_product(self, datas, v, gauss_newton):
        assert self.trainer.model.training, "[SimpleTrainer] model was changed to eval mode!"
        self.trainer.optimizer.zero_grad()
        if not gauss_newton:
            # Run model and Compute loss
            # set image as variable with requires_grad_() DO NOT need gradient of image
            # for i, data in enumerate(tqdm(datas)):
            #     datas[i]['image'] = datas[i]['image'].type(torch.FloatTensor).requires_grad_()  # test
            """
            If you want to do something with the losses, you can wrap the model.
            """
            with EventStorage() as storage:
                loss_dict = self.trainer.model(datas)
            losses = sum(loss_dict.values())
            if not torch.isfinite(losses):
                print('WARNING: non-finite loss, ending training ', losses)
                return
            if torch.isnan(losses):
                print('WARNING: non-finite loss, ending training ', losses)
                return
            # losses *= datas.shape[0] / 64.0

            # dL/dtheta
            first_derivatives = []
            for layer_index, layer in enumerate(self.trainer.optimizer.param_groups):  #
                if layer_index >= 0:  # only compute the last several layers
                    # dL/dtheta # gradient for first layer params
                    first_d = torch.autograd.grad(losses, layer['params'][0], create_graph=True,
                                                  retain_graph=True)
                    first_derivatives.append(first_d[0])
            first_derivatives = tuple(i for i in first_derivatives)
            # d/dI (dL/dtheta) d vectorA / d vectorB grad_outputs has same dim as vectorA, output has same dim as vectorB

            second_derivatives = []
            for layer_index, layer in enumerate(self.trainer.optimizer.param_groups):  #
                if layer_index >= 0:  # only compute the last several layers
                    # dL/dtheta # gradient for first layer params
                    second_d = torch.autograd.grad(first_derivatives, layer['params'][0], grad_outputs=v,
                                                  retain_graph=True) # compute Hv
                    # second_derivatives.append(second_d[0])
                    second_derivatives.append(second_d[0].detach().data)
            del first_derivatives
            # second_derivatives = [x.cpu() for x in second_derivatives] # release memory
            second_derivatives = tuple(i for i in second_derivatives)

            return_grads = [grad_elem for grad_elem in second_derivatives]

        # print('Norm of second order gradient: %s' % np.linalg.norm(
        #     np.concatenate([x.flatten().cpu() for x in return_grads])))
        torch.cuda.empty_cache() # release the required memory
        return return_grads


def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg

def update_psi(psi_categorical_prob, grad_psi, opt_lr=0.00001):
    lr = opt_lr
    psi_categorical_prob = psi_categorical_prob + lr * grad_psi
    return psi_categorical_prob.detach()

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        params = np.array(params)
        grads = np.array(grads)
        params += - self.lr * grads
        return torch.Tensor(params)

class Momentum:
    def __init__(self, lr=0.01, momemtum=0.9):
        self.lr = lr
        self.momemtum = momemtum
        self.v = None

    def update(self, params, grads):
        params = np.array(params)
        grads = np.array(grads)
        if self.v is None:
            self.v = np.zeros_like(params)

        self.v = self.momemtum * self.v - self.lr * grads
        params += self.v

        return torch.Tensor(params)
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        params = np.array(params)
        grads = np.array(grads)
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        self.m += (1 - self.beta1) * (grads - self.m)
        self.v += (1 - self.beta2) * (grads**2 - self.v)

        params -= lr_t * self.m / (np.sqrt(self.v) + 1e-7)
        return torch.Tensor(params)


def adjust_learning_rate(epoch, base_lr, max_epoch):
    if epoch <= 5:  # quickly improve the lr to base lr
      return base_lr * epoch / 5
    else:
      return base_lr * (1 - epoch/max_epoch)


def bilevel_optimization(my_nerf, my_detector, Optimiation_parameter):
    '''
    input
    nerf, detector and parameter
    output, after several epochs, the optimal \psi and optimal detector
    '''
    epochs = Optimiation_parameter.n_epochs # optimization epochs
    torch_softmax = torch.nn.Softmax(dim=0)
    # psi categorical distribution initialization
    if Optimiation_parameter.psi_pose_cats_mode == 'uniform':
        q_psi_categorical_prob = torch.Tensor([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])
    elif Optimiation_parameter.psi_pose_cats_mode == 'two_13':
        q_psi_categorical_prob = torch.Tensor([0.44, 0.02, 0.44, 0.02, 0.02, 0.02, 0.02, 0.02])
    elif Optimiation_parameter.psi_pose_cats_mode == 'two_27':
        q_psi_categorical_prob = torch.Tensor([0.02, 0.44, 0.02, 0.02, 0.02, 0.02, 0.44, 0.02])
    elif Optimiation_parameter.psi_pose_cats_mode == 'three_123':
        q_psi_categorical_prob = torch.Tensor([0.3, 0.3, 0.3, 0.02, 0.02, 0.02, 0.02, 0.02])
    elif Optimiation_parameter.psi_pose_cats_mode == 'three_147':
        q_psi_categorical_prob = torch.Tensor([0.3, 0.02, 0.02, 0.3, 0.02, 0.02, 0.3, 0.02])
    else:
        q_psi_categorical_prob = torch.Tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])
        q_psi_categorical_prob[int(Optimiation_parameter.psi_pose_cats_mode)-1] = 0.86 # one bin dominate

    # optimization method
    if Optimiation_parameter.opt_method == 'momentum':
        psi_optimizer = Momentum(Optimiation_parameter.opt_lr, momemtum=0.9)
    elif Optimiation_parameter.opt_method == 'Adam':
        psi_optimizer = Adam(Optimiation_parameter.opt_lr, beta1=0.9, beta2=0.999)
    else: # use sgd
        psi_optimizer = SGD(Optimiation_parameter.opt_lr)

    for i in tqdm(range(epochs)):
        print('strating iteration {}'.format(i))
        ## 1 NeRF generate K = 50/100 images as D_train given /psi = categorical distribution of pose
        # This forward path only record the sample value, do not requires gradient
        q_psi_categorical_prob_nograd = q_psi_categorical_prob.clone() # start a now epoch with no history gradient
        Nerf_imgs_savedir, sample_log = my_nerf.render_images(q_psi_categorical_prob_nograd, Optimiation_parameter)
        ## 2 Fine-tune Faster RCNN model with D_train
        # 2.1 create coco format dataset and register it
        my_detector.create_dataset(Nerf_imgs_savedir)
        # 2.2 train faster RCNN
        my_detector.train(i)
        my_detector.inference(i)
        print('##########################################AFTER  my_detector.train(i)  ##################################################')

        '''optimization'''
        if Optimiation_parameter.optimization:
            ## 3 Compute d L_val / d \psi and update \psi
            # 3.1 Computer inverse_hvp using CG (write accurate H)
            inverse_hvp = my_detector.compute_inverse_hvp() # inverse_hvp = H^-1 * dL_val/dtheta fix backbone only update last layers (accurate)
            print('##########################################AFTER  inverse_hvp  ##################################################')
            # 3.2 Computer gradient of expectations (first part d (dL_train/dtheta) / d I)
            grad_E = my_detector.compute_grad_E(inverse_hvp) # d (dL_train/dtheta) / d I * inverse_hvp (weight)
            print('##########################################AFTER  grad_E  ##################################################')
            # 3.3 Compute dI/d\psi in image patch wise, Run forward of NeRF again for backward.
            q_psi_categorical_prob = q_psi_categorical_prob.requires_grad_()
            grad_psi = my_nerf.render_images_grad(q_psi_categorical_prob, Optimiation_parameter, sample_log, grad_E) # dL_val/d_\psi = dI/d\psi * grad_E (weight)
            print('##########################################AFTER  grad_psi  ##################################################')
            # 3.4 Updata \psi by descending the gradient, should has same dimension as \psi
            q_psi_categorical_prob = psi_optimizer.update(q_psi_categorical_prob.detach(), grad_psi)
            print('q_psi_categorical_prob', q_psi_categorical_prob)
            print('grad_psi', grad_psi)
            print('##########################################AFTER  q_psi_categorical_prob  ##################################################')
            # save results
            with open(os.path.join(my_detector.cfg.OUTPUT_DIR, 'save_result.txt'), 'a', encoding='utf-8') as f:
                f.write('epoch: {}'.format(i) + str(torch_softmax(q_psi_categorical_prob / Optimiation_parameter.gumble_T)))
                f.write('\n')
            # update learning rate
            psi_optimizer.lr = adjust_learning_rate(epoch=i, base_lr=Optimiation_parameter.opt_lr, max_epoch=Optimiation_parameter.n_epochs)


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    '''
    nerf parameters
    '''

    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    '''
    detector parameters, most parameters are in Detector class
    '''
    parser.add_argument("--pretrain",   type=int, default=0,
                        help='optimization flag')
    parser.add_argument("--pretrain_weight",   type=str, default='/path/of/pretrained/detectron/model',
                        help='optimization flag')
    '''
    optimization nerf parameter \psi 
    '''
    parser.add_argument("--expname", type=str, default='exp_ycb_synthetic',
                        help='experiment name')
    parser.add_argument("--optimization",   type=int, default=1,
                        help='optimization flag')
    parser.add_argument("--n_samples_K",   type=int, default=50,
                        help='in each iteration number of sample images')
    parser.add_argument("--n_epochs",   type=int, default=50,
                        help='number of epochs to optimize')
    parser.add_argument("--object_id", type=str, default='2',
                        help='1~21')
    parser.add_argument('--psi_pose_cats_mode', type=str,  default='5', help='1~8, uniform, two_13, two_27, three_123, three_147')
    parser.add_argument('--train_val_path_info', type=str, default='../configs/ycb_synthetic_train_val_path_info.json',
                        help='json that save the images')

    parser.add_argument("--opt_lr", type=float, default=5e-5,
                        help='learning rate of the optimization')
    parser.add_argument("--gumble_T", type=float, default=0.1,
                        help='gumble softmax temperature [0~1]')

    parser.add_argument('--test_distribution', type=str,  default='one_1', help='one_1~one_8, two_12, two_15, three_135')
    parser.add_argument('--opt_method', type=str,  default='momentum', help='sgd, momentum, Adam')

    return parser


def main():
    # Load parameters of nerf
    parser = config_parser()
    args = parser.parse_args()
    Nerf_parameter = args
    my_nerf = NeRF(Nerf_parameter) # nerf initialization

    # Load parameters of Detector (RetinaNet)
    Detector_parameter = args
    my_detector = Detector(Detector_parameter) # Detector initialization

    # Load parameters for bi-level optimization and \psi (the rendering parameter we want to optimize)
    Optimiation_parameter = args

    # Optimization
    results = bilevel_optimization(my_nerf, my_detector, Optimiation_parameter)



if __name__=='__main__':
    main()


