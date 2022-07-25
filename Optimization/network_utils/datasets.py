import torch
from torch.utils import data
import os
import os.path as osp
import numpy as np
import cv2
import torchvision.utils as vutils
from PIL import Image

SEED = 10
np.random.seed(SEED)

class Dataset_1dobject(data.Dataset):
    def __init__(self, root, train=1, max_iters=None, crop_size=(32, 32), mean=(128, 128, 128), scale=True, mirror=True, transform=None, writer=None, iter=0, last_k=0):
        self.root = root
        self.train = train
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mean = mean
        self.is_mirror = mirror
        self.files = []
        # self.transform = transforms.Compose([
        #                 transforms.RandomHorizontalFlip(),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize(rgb_mean, rgb_std),
        #                 ])
        # for split in ["train", "trainval", "val"]:
        images=[]
        for subdir, dirs, files in sorted(os.walk(root)):
          train_leng = 0.9*len(files)
          for file in sorted(files):
              # print(os.path.join(subdir, file))
              filepath = subdir + '/' + file
              img_file_1 = filepath
              lbl_file_1 = file.split('_')[0]
              
              if lbl_file_1 == 'sphere':
                lbl_file_1 = 1
              elif lbl_file_1 == 'cube':
                lbl_file_1 = 0
              elif lbl_file_1 == 'cylinder':
                lbl_file_1 = 2

              if self.train == 1:
                if int(file.split('_')[-1].split('.')[0]) > 89:
                  continue
              elif self.train ==0:
                if int(file.split('_')[-1].split('.')[0]) <= 89:
                  continue

              try:
                img1 = cv2.imread(img_file_1, cv2.IMREAD_COLOR)
                img1.shape
              except:
                print(img_file_1)
                continue
              else:
                self.files.append({
                      "img_1": img1,
                      "lab_1": lbl_file_1})
                if writer is not None and last_k==1:
                  img = cv2.resize(img1, (4*self.crop_h, 4*self.crop_w), interpolation=cv2.INTER_CUBIC)
                  # img=img1
                  img = img.transpose((2, 0, 1))
                  if images == []:
                    images=[img]
                  else:
                    images.append(img)
                  # writer.add_image('Image', img)
              # print(img_file_1, lbl_file_1)
              
              #######################################################################
              ###################   WRITING TENSORLOG of IMAGES #####################
              # # Clear out any prior log data.
              # !rm -rf logs

              # # Sets up a timestamped log directory.
              # logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
              # # Creates a file writer for the log directory.
              # file_writer = tf.summary.create_file_writer(logdir)

              #######################################################################
              #######################################################################
        if writer is not None and last_k==1:
          images = torch.tensor(np.stack(images, axis=0)).float()
          xvn=vutils.make_grid(images, normalize=True, scale_each=True)

          writer.add_image('Image', xvn, iter)

        if not max_iters == None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = datafiles["img_1"]
        # image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        label_1 = datafiles["lab_1"]
        try:
          image_1 = cv2.resize(image_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_CUBIC)
        except:
          print(datafiles["img_1"])
        size = image_1.shape

        image_1 = np.asarray(image_1, np.float32)
        image_1 -= self.mean

        '''
        img_h, img_w = label_1.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image_1 = np.asarray(image_1[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label_1 = np.asarray(label_1[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        '''

        # if self.transform:
        #     sample = self.transform(sample)

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image_1 = image_1[:, :, ::flip]

        return image_1.copy(), label_1

class Dataset_segobject(data.Dataset):
    def __init__(self, root, train=1, max_iters=None, crop_size=(32, 32), mean=(128, 128, 128), scale=True, mirror=True, transform=None, writer=None, iter=0, last_k=1):
        self.root = root
        self.train = train
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mean = mean
        self.is_mirror = mirror
        self.files = []
        self.val=0
        # self.transform = transforms.Compose([
        #                 transforms.RandomHorizontalFlip(),
        #                 transforms.ToTensor(),
        #                 transforms.Normalize(rgb_mean, rgb_std),
        #                 ])
        # for split in ["train", "trainval", "val"]:
        images=[]
        for subdir, dirs, files in sorted(os.walk(root)):
          train_leng = 0.9*len(files)
          for file in sorted(files):
              filepath = subdir + '/' + file
              if file[0] == '.':
                # print(file)
                continue
              # print(os.path.join(subdir, file))
              img_file_1 = filepath
              lbl_file_1 = subdir[:-7] + os.sep + 'labels' + os.sep + file
              # print(subdir[:-7][-3:])
              if subdir[:-7][-3:] == 'val':
                self.val=1
              # print(lbl_file_1)
              
              # if lbl_file_1 == 'sphere':
              #   lbl_file_1 = 1
              # elif lbl_file_1 == 'cube':
              #   lbl_file_1 = 0
              # elif lbl_file_1 == 'cylinder':
              #   lbl_file_1 = 2
              img1 = cv2.imread(img_file_1, cv2.IMREAD_COLOR)
              label1 = cv2.imread(lbl_file_1, cv2.IMREAD_COLOR)
              # label1 = Image.open(lbl_file_1)
              # label_1 = np.array(label1, dtype=np.uint8)
              label_1 = label1
              new_label1=np.zeros((label_1.shape[0],label_1.shape[1]))
              for p in range(label_1.shape[0]):
                for lt in range(label_1.shape[1]):
                  if label_1[p,lt][0] == 64:
                    new_label1[p,lt] = 0
                  elif label_1[p,lt][0] == 255:
                    new_label1[p,lt] = 1
                    # print('got cube')
                  elif label_1[p,lt][1] == 255:
                    new_label1[p,lt] = 2
                    # print('got sphere')
                  elif label_1[p,lt][2] == 255:
                    new_label1[p,lt] = 3
                    # print('got cylinder')
              try:
                img1 = cv2.imread(img_file_1, cv2.IMREAD_COLOR)
                label1 = cv2.imread(lbl_file_1, cv2.IMREAD_COLOR)
                # label1 = Image.open(lbl_file_1)
                # label_1 = np.array(label_1, dtype=np.uint8)
                label_1 = label1
                new_label1=np.zeros((label_1.shape[0],label_1.shape[1]))
                for p in range(label_1.shape[0]):
                  for lt in range(label_1.shape[1]):
                    if label_1[p,lt][0] == 64:
                      new_label1[p,lt] = 0
                    elif label_1[p,lt][0] == 255:
                      new_label1[p,lt] = 1
                      # print('got cube')
                    elif label_1[p,lt][1] == 255:
                      new_label1[p,lt] = 2
                      # print('got sphere')
                    elif label_1[p,lt][2] == 255:
                      new_label1[p,lt] = 3
                      # print('got cylinder')
              except:
                print('hello',img_file_1)
                continue
              else:
                self.files.append({
                      "img_1": img1,
                      "lab_1": new_label1})
                if writer is not None and last_k==1:
                  img = cv2.resize(img1, (4*self.crop_h, 4*self.crop_w), interpolation=cv2.INTER_CUBIC)
                  # img=img1
                  img = img.transpose((2, 0, 1))
                  if images == []:
                    images=[img]
                  else:
                    images.append(img)

        if writer is not None and last_k==1:
          images = torch.tensor(np.stack(images, axis=0)).float()
          xvn=vutils.make_grid(images, normalize=True, scale_each=False, nrow=10)

          if self.val==0:
            writer.add_image('Image', xvn, iter)
          else:
            writer.add_image('Val Images', xvn, iter)

        if not max_iters == None:
            self.files = self.files * int(np.ceil(float(max_iters) / len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_1 = datafiles["img_1"]
        # image_1 = cv2.imread(datafiles["img_1"], cv2.IMREAD_COLOR)
        label_1 = datafiles["lab_1"]
        try:
          image_1 = cv2.resize(image_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_CUBIC)
          label_1 = cv2.resize(label_1, (self.crop_h, self.crop_w), interpolation=cv2.INTER_NEAREST)
        except:
          print(datafiles["img_1"])
        size = image_1.shape
        cv2.imwrite('sh.png',image_1)
        cv2.imwrite('sh.png',label_1)

        image_1 = np.asarray(image_1, np.float32)
        label_1 = np.asarray(label_1, np.float32)
        image_1 -= self.mean

        '''
        img_h, img_w = label_1.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image_1 = np.asarray(image_1[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label_1 = np.asarray(label_1[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        '''

        # if self.transform:
        #     sample = self.transform(sample)

        #image = image[:, :, ::-1]  # change to BGR
        image_1 = image_1.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image_1 = image_1[:, :, ::flip]
            label_1 = label_1[:, ::flip]
        # THIS FUNCTION IS WORKING FINE
        # image_1 = image_1[np.newaxis, ...]
        # label_1 = label_1[np.newaxis, ...]

        return image_1.copy(), label_1.copy()