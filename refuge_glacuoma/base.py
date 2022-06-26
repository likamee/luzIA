import os
import shutil
import json
import csv
import random
import pickle
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


import PIL
from PIL import Image, ImageOps
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.measurements import label
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Dataset class
class RefugeDataset(Dataset):

    def __init__(self, root_dir, split='train', output_size=(256,256)):
        # Define attributes
        self.output_size = output_size
        self.root_dir = root_dir
        self.split = split
        
        # Load data index
        with open(os.path.join(self.root_dir, self.split, 'index.json')) as f:
            self.index = json.load(f)
            
        self.images = []
        for k in range(len(self.index)):
            print('Loading {} image {}/{}...'.format(split, k, len(self.index)), end='\r')
            img_name = os.path.join(self.root_dir, self.split, 'images', self.index[str(k)]['IMG_NAME'])
            img = np.array(Image.open(img_name).convert('RGB'))
            img = transforms.functional.to_tensor(img)
            img = transforms.functional.resize(img, self.output_size, interpolation=Image.BILINEAR)
            self.images.append(img)
            
        # Load ground truth for 'train' and 'val' sets
        if split != 'test':
            self.segs = []
            for k in range(len(self.index)):
                print('Loading {} segmentation {}/{}...'.format(split, k, len(self.index)), end='\r')
                seg_name = os.path.join(self.root_dir, self.split, 'gts', self.index[str(k)]['IMG_NAME'].split('.')[0]+'.bmp')
                seg = np.array(Image.open(seg_name)).copy()
                seg = 255. - seg
                od = (seg>=127.).astype(np.float32)
                oc = (seg>=250.).astype(np.float32)
                od = torch.from_numpy(od[None,:,:])
                oc = torch.from_numpy(oc[None,:,:])
                od = transforms.functional.resize(od, self.output_size, interpolation=Image.NEAREST)
                oc = transforms.functional.resize(oc, self.output_size, interpolation=Image.NEAREST)
                seg = torch.cat([od, oc], dim=0)
                self.segs.append(seg)
                
        print('Succesfully loaded {} dataset.'.format(split) + ' '*50)
            
            
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Image
        img = self.images[idx]
    
        # Return only images for 'test' set
        if self.split == 'test':
            return img
        
        # Else, images and ground truth
        else:
            # Label
            lab = torch.tensor(self.index[str(idx)]['Label'], dtype=torch.float32)

            # Segmentation masks
            seg = self.segs[idx]

            # Fovea localization
            f_x = self.index[str(idx)]['Fovea_X']
            f_y = self.index[str(idx)]['Fovea_Y']
            fov = torch.FloatTensor([f_x, f_y])
        
            return img, lab, seg, fov, self.index[str(idx)]['IMG_NAME']
# Metrics
EPS = 1e-7

def compute_dice_coef(input, target):
    '''
    Compute dice score metric.
    '''
    batch_size = input.shape[0]
    return sum([dice_coef_sample(input[k,:,:], target[k,:,:]) for k in range(batch_size)])/batch_size

def dice_coef_sample(input, target):
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return (2. * intersection) / (iflat.sum() + tflat.sum())


def vertical_diameter(binary_segmentation):
    '''
    Get the vertical diameter from a binary segmentation.
    The vertical diameter is defined as the "fattest" area of the binary_segmentation parameter.
    '''

    # get the sum of the pixels in the vertical axis
    vertical_axis_diameter = np.sum(binary_segmentation, axis=1)

    # pick the maximum value
    diameter = np.max(vertical_axis_diameter, axis=1)

    # return it
    return diameter



def vertical_cup_to_disc_ratio(od, oc):
    '''
    Compute the vertical cup-to-disc ratio from a given labelling map.
    '''
    # compute the cup diameter
    cup_diameter = vertical_diameter(oc)
    # compute the disc diameter
    disc_diameter = vertical_diameter(od)

    return cup_diameter / (disc_diameter + EPS)

def compute_vCDR_error(pred_od, pred_oc, gt_od, gt_oc):
    '''
    Compute vCDR prediction error, along with predicted vCDR and ground truth vCDR.
    '''
    pred_vCDR = vertical_cup_to_disc_ratio(pred_od, pred_oc)
    gt_vCDR = vertical_cup_to_disc_ratio(gt_od, gt_oc)
    vCDR_err = np.mean(np.abs(gt_vCDR - pred_vCDR))
    return vCDR_err, pred_vCDR, gt_vCDR


def classif_eval(classif_preds, classif_gts):
    '''
    Compute AUC classification score.
    '''
    auc = roc_auc_score(classif_gts, classif_preds)
    return auc


def fov_error(pred_fov, gt_fov):
    '''
    Fovea localization error metric (mean root squared error).
    '''
    err = np.sqrt(np.sum((gt_fov-pred_fov)**2, axis=1)).mean()
    return err
# Post-processing functions
def refine_seg(pred):
    '''
    Only retain the biggest connected component of a segmentation map.
    '''
    np_pred = pred.numpy()
        
    largest_ccs = []
    for i in range(np_pred.shape[0]):
        labeled, ncomponents = label(np_pred[i,:,:])
        bincounts = np.bincount(labeled.flat)[1:]
        if len(bincounts) == 0:
            largest_cc = labeled == 0
        else:
            largest_cc = labeled == np.argmax(bincounts)+1
        largest_cc = torch.tensor(largest_cc, dtype=torch.float32)
        largest_ccs.append(largest_cc)
    largest_ccs = torch.stack(largest_ccs)
    
    return largest_ccs
# Network
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epoch = 0

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.output_layer = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.output_layer(out)
        out = torch.sigmoid(out)
        return out
    
class RN(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.epoch = 0

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.output_layer = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.output_layer(out)
        out = torch.sigmoid(out)
        return out

    
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Use the normal convolutions to reduce the number of channels
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    '''
    Simple convolution.
    '''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
# Settings
## DATA AUGMENTATION
dir_path="data/refuge_data/"
df_train = pd.read_json(dir_path+"train/index.json").T.rename(columns={"ImgName" : "IMG_NAME"})
df_val = pd.read_json(dir_path+"val/index.json").T.rename(columns={"ImgName" : "IMG_NAME"})
df_test = pd.read_json(dir_path+"test/index.json").T.rename(columns={"ImgName" : "IMG_NAME"})

IMG_SIZE = 512
NUM_CLASSES = 5
SEED = 77
TRAIN_NUM = 1000
import torch
from torchvision import transforms
df_aug_train = df_train
df_preproc_val = df_val
df_preproc_test = df_test
loader_transform = transforms.RandomRotation(180)
def pre_process(image, color=True, gaussian=False, kernel=IMG_SIZE//10):
    if color:
        image = image
    else:
        image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    if gaussian:
        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , kernel) ,-4 ,128)
    else:
        image=cv2.addWeighted ( image,4, cv2.medianBlur(image, kernel) ,-4 ,128)
        
    return PIL.Image.fromarray(image, "RGB")

def aug_train_img_creator(tranform, df_init=df_train, df_augmented=df_aug_train):
    try:
        shutil.rmtree('refuge_data')
    except:
        print("no such directory1")
    
    shutil.copytree('data/refuge_data','data/refuge_data' )
    try:
        shutil.rmtree('data/refuge_data/train/gts')
        shutil.rmtree('data/refuge_data/train/images')
        os.remove('data/refuge_data/train/index.json')
    except:
        print("no such directory2")
    
    os.mkdir('data/refuge_data/train/images')
    os.mkdir('data/refuge_data/train/gts')
    i = 0
    for image in df_init.IMG_NAME:
        i = i+1
        if i%10 ==0:
            print(i)
        path_to_img = dir_path+"train/images/"+image
        bmp = image.replace('.jpg', '.bmp')
        path_to_bmp = dir_path+"train/gts/"+bmp
        img = cv2.imread(path_to_img)
        bmpimg = PIL.Image.open(path_to_bmp)
        #original = pre_process(img)
        original = PIL.Image.fromarray(img, "RGB")
        name = image.replace('.jpg', '')
        bmpimg.save("data/refuge_data/train/gts/"+bmp)
        original.save("data/refuge_data/train/images/"+name+".jpg")
        for k in range(6):
            new_img = loader_transform(original)
            name = image.replace('.jpg', '') + str(k)
            new_img.save("data/refuge_data/train/images/"+name+".jpg")
            bmpimg.save("data/refuge_data/train/gts/"+name+".bmp")
            new_sample = df_init[df_init.IMG_NAME == image]
            new_sample.IMG_NAME = name+".jpg"
            df_augmented = df_augmented.append(new_sample, ignore_index=True)
    print('Augmentation ok')
    return df_augmented
df_val
def preprocess_val_test( df_vinit = df_val , df_tinit = df_test):
    try:
        shutil.rmtree('data/refuge_data/val/images')
        shutil.rmtree('data/refuge_data/val/gts')
        os.remove('data/refuge_data/val/index.json')
        shutil.rmtree('data/refuge_data/test/images')
        os.remove('data/refuge_data/test/index.json')
    except:
        print("no such directory2")
    
    os.mkdir('data/refuge_data/val/images')
    os.mkdir('data/refuge_data/val/gts')
    os.mkdir('data/refuge_data/test/images')
    df_v = pd.DataFrame(columns=['IMG_NAME','Label', 'Fovea_X', 'Fovea_Y', 'Size_X', 'Size_Y'])
    df_t = pd.DataFrame(columns=['IMG_NAME', 'Size_X', 'Size_Y'])
    for image in df_vinit.IMG_NAME:
        path_to_img = dir_path+"val/images/"+image
        path_to_bmp = dir_path+"val/gts/"+image.replace('.jpg', '.bmp')
        img = cv2.imread(path_to_img)
        bmp = PIL.Image.open(path_to_bmp)
        #new_img = pre_process(img)
        new_img = PIL.Image.fromarray(img, "RGB")
        new_img.save("data/refuge_data/val/images/"+image)
        bmp.save("data/refuge_data/val/gts/"+image.replace('.jpg', '.bmp'))
        sample = df_vinit[df_vinit.IMG_NAME == image]
        df_v = df_v.append(sample, ignore_index=True)
    
    print('Preprocessing validation ok')
    for image in df_tinit.IMG_NAME:
        path_to_img = dir_path+"test/images/"+image
        img = cv2.imread(path_to_img)
        #new_img = pre_process(img)
        new_img = PIL.Image.fromarray(img, "RGB")
        new_img.save("data/refuge_data/test/images/"+image)
        sample = df_tinit[df_tinit.IMG_NAME == image]
        df_t = df_t.append(sample, ignore_index=True)
    print('Preprocessing test ok')
    return df_v, df_t
        
loader_transform = transforms.RandomRotation(10)
df_aug_train = aug_train_img_creator(loader_transform)
df_preproc_val, df_preproc_test = preprocess_val_test()
def write_jsons(train=df_aug_train, val=df_preproc_val, test=df_preproc_test):
    train = train.T
    val = val.T
    test = test.T
    train.to_json('data/refuge_data/train/index.json')
    val.to_json('data/refuge_data/val/index.json')
    test.to_json('data/refuge_data/test/index.json')
    
    print('Write JSONs ok')
write_jsons()
# Create datasets and data loaders
import matplotlib.pyplot as plt
root_dir = 'data/refuge_data'
lr = 1e-4
batch_size = 8
num_workers = 8
total_epoch = 100
# Datasets


train_set = RefugeDataset(root_dir, 
                          split='train')

val_set = RefugeDataset(root_dir, 
                        split='val')

test_set = RefugeDataset(root_dir, 
                         split='test')

# Dataloaders
train_loader = DataLoader(train_set, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          num_workers=num_workers,
                          pin_memory=True,
                         )
val_loader = DataLoader(val_set, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=True,
                        )
test_loader = DataLoader(test_set, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        num_workers=num_workers,
                        pin_memory=True)
# Device, model, loss and optimizer
import torchvision.models as models
# Device
device = torch.device("cuda:0")

model = UNet(n_channels=3, n_classes=2).to(device)
# model = models.inception_v3(pretrained=True).to(device)
# model.AuxLogits.fc = nn.Linear(768, 2)
# model.fc = nn.Linear(2048, 2)

#model = models.resnet50(pretrained=True).to(device)
#model.fc=nn.Linear(512, 2)



# Loss
seg_loss = torch.nn.BCELoss(reduction='mean')

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# Train for OC/OD segmentation
# Define parameters
nb_train_batches = len(train_loader)
nb_val_batches = len(val_loader)
nb_iter = 0
best_val_auc = 0.
epoch = 50
epoch_c=0

while epoch_c < total_epoch:
    epoch_c+=1
    # Accumulators
    train_vCDRs, val_vCDRs = [], []
    train_classif_gts, val_classif_gts = [], []
    train_loss, val_loss = 0., 0.
    train_dsc_od, val_dsc_od = 0., 0.
    train_dsc_oc, val_dsc_oc = 0., 0.
    train_vCDR_error, val_vCDR_error = 0., 0.
    
    ############
    # TRAINING #
    ############
    model.train()
    train_data = iter(train_loader)
    for k in range(nb_train_batches):
        # Loads data
        imgs, classif_gts, seg_gts, fov_coords, names = train_data.next()
        imgs, classif_gts, seg_gts = imgs.to(device), classif_gts.to(device), seg_gts.to(device)

        # Forward pass
        logits = model(imgs)
        print(logits.shape)
        loss = seg_loss(logits, seg_gts)
 
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / nb_train_batches
        
        with torch.no_grad():
            # Compute segmentation metric
            pred_od = refine_seg((logits[:,0,:,:]>=0.5).type(torch.int8).cpu()).to(device)
            pred_oc = refine_seg((logits[:,1,:,:]>=0.5).type(torch.int8).cpu()).to(device)
            gt_od = seg_gts[:,0,:,:].type(torch.int8)
            gt_oc = seg_gts[:,1,:,:].type(torch.int8)
            dsc_od = compute_dice_coef(pred_od, gt_od)
            dsc_oc = compute_dice_coef(pred_oc, gt_oc)
            train_dsc_od += dsc_od.item()/nb_train_batches
            train_dsc_oc += dsc_oc.item()/nb_train_batches


            # Compute and store vCDRs
            vCDR_error, pred_vCDR, gt_vCDR = compute_vCDR_error(pred_od.cpu().numpy(), pred_oc.cpu().numpy(), gt_od.cpu().numpy(), gt_oc.cpu().numpy())
            train_vCDRs += pred_vCDR.tolist()
            train_vCDR_error += vCDR_error / nb_train_batches
            train_classif_gts += classif_gts.cpu().numpy().tolist()
            
        # Increase iterations
        nb_iter += 1
        
        # Std out
        print('Epoch {}, iter {}/{}, loss {:.6f}'.format(model.epoch+1, k+1, nb_train_batches, loss.item()) + ' '*20, 
              end='\r')
        
    # Train a logistic regression on vCDRs
    train_vCDRs = np.array(train_vCDRs).reshape(-1,1)
    train_classif_gts = np.array(train_classif_gts)
    clf = LogisticRegression(random_state=0, solver='lbfgs').fit(train_vCDRs, train_classif_gts)
    train_classif_preds = clf.predict_proba(train_vCDRs)[:,1]
    train_auc = classif_eval(train_classif_preds, train_classif_gts)
    
    ##############
    # VALIDATION #
    ##############
    model.eval()
    with torch.no_grad():
        val_data = iter(val_loader)
        for k in range(nb_val_batches):
            # Loads data
            imgs, classif_gts, seg_gts, fov_coords, names = val_data.next()
            imgs, classif_gts, seg_gts = imgs.to(device), classif_gts.to(device), seg_gts.to(device)

            # Forward pass
            logits = model(imgs)
            val_loss += seg_loss(logits, seg_gts).item() / nb_val_batches

            # Std out
            print('Validation iter {}/{}'.format(k+1, nb_val_batches) + ' '*50, 
                  end='\r')
            
            # Compute segmentation metric
            pred_od = refine_seg((logits[:,0,:,:]>=0.5).type(torch.int8).cpu()).to(device)
            pred_oc = refine_seg((logits[:,1,:,:]>=0.5).type(torch.int8).cpu()).to(device)
            gt_od = seg_gts[:,0,:,:].type(torch.int8)
            gt_oc = seg_gts[:,1,:,:].type(torch.int8)
            dsc_od = compute_dice_coef(pred_od, gt_od)
            dsc_oc = compute_dice_coef(pred_oc, gt_oc)
            val_dsc_od += dsc_od.item()/nb_val_batches
            val_dsc_oc += dsc_oc.item()/nb_val_batches
            
            # Compute and store vCDRs
            vCDR_error, pred_vCDR, gt_vCDR = compute_vCDR_error(pred_od.cpu().numpy(), pred_oc.cpu().numpy(), gt_od.cpu().numpy(), gt_oc.cpu().numpy())
            val_vCDRs += pred_vCDR.tolist()
            val_vCDR_error += vCDR_error / nb_val_batches
            val_classif_gts += classif_gts.cpu().numpy().tolist()
            

    # Glaucoma predictions from vCDRs
    val_vCDRs = np.array(val_vCDRs).reshape(-1,1)
    val_classif_gts = np.array(val_classif_gts)
    val_classif_preds = clf.predict_proba(val_vCDRs)[:,1]
    val_auc = classif_eval(val_classif_preds, val_classif_gts)
        
    # Validation results
    print('VALIDATION epoch {}'.format(model.epoch+1)+' '*50)
    print('LOSSES: {:.4f} (train), {:.4f} (val)'.format(train_loss, val_loss))
    print('OD segmentation (Dice Score): {:.4f} (train), {:.4f} (val)'.format(train_dsc_od, val_dsc_od))
    print('OC segmentation (Dice Score): {:.4f} (train), {:.4f} (val)'.format(train_dsc_oc, val_dsc_oc))
    print('vCDR error: {:.4f} (train), {:.4f} (val)'.format(train_vCDR_error, val_vCDR_error))
    print('Classification (AUC): {:.4f} (train), {:.4f} (val)'.format(train_auc, val_auc))
    
    # Save model if best validation AUC is reached
    if val_auc > best_val_auc:
        torch.save(model.state_dict(), '/kaggle/working/best_AUC_weights.pth')
        with open('/kaggle/working/best_AUC_classifier.pkl', 'wb') as clf_file:
            pickle.dump(clf, clf_file)
        best_val_auc = val_auc
        print('Best validation AUC reached. Saved model weights and classifier.')
    print('_'*50)
        
    # End of epoch
    model.epoch += 1
        

# Load best model + classifier
# Load model and classifier
model = UNet(n_channels=3, n_classes=2).to(device)
model.load_state_dict(torch.load('/kaggle/working/best_AUC_weights.pth'))
with open('/kaggle/working/best_AUC_classifier.pkl', 'rb') as clf_file:
    clf = pickle.load(clf_file)
# Check performance is maintained on validation
model.eval()
val_vCDRs = []
val_classif_gts = []
val_loss = 0.
val_dsc_od = 0.
val_dsc_oc = 0.
val_vCDR_error = 0.
with torch.no_grad():
    val_data = iter(val_loader)
    for k in range(nb_val_batches):
        # Loads data
        imgs, classif_gts, seg_gts, fov_coords, names = val_data.next()
        imgs, classif_gts, seg_gts = imgs.to(device), classif_gts.to(device), seg_gts.to(device)

        # Forward pass
        logits = model(imgs)
        val_loss += seg_loss(logits, seg_gts).item() / nb_val_batches

        # Std out
        print('Validation iter {}/{}'.format(k+1, nb_val_batches) + ' '*50, 
              end='\r')

        # Compute segmentation metric
        pred_od = refine_seg((logits[:,0,:,:]>=0.5).type(torch.int8).cpu()).to(device)
        pred_oc = refine_seg((logits[:,1,:,:]>=0.5).type(torch.int8).cpu()).to(device)
        gt_od = seg_gts[:,0,:,:].type(torch.int8)
        gt_oc = seg_gts[:,1,:,:].type(torch.int8)
        dsc_od = compute_dice_coef(pred_od, gt_od)
        dsc_oc = compute_dice_coef(pred_oc, gt_oc)
        val_dsc_od += dsc_od.item()/nb_val_batches
        val_dsc_oc += dsc_oc.item()/nb_val_batches

        # Compute and store vCDRs
        vCDR_error, pred_vCDR, gt_vCDR = compute_vCDR_error(pred_od.cpu().numpy(), pred_oc.cpu().numpy(), gt_od.cpu().numpy(), gt_oc.cpu().numpy())
        val_vCDRs += pred_vCDR.tolist()
        val_vCDR_error += vCDR_error / nb_val_batches
        val_classif_gts += classif_gts.cpu().numpy().tolist()


# Glaucoma predictions from vCDRs
val_vCDRs = np.array(val_vCDRs).reshape(-1,1)
val_classif_gts = np.array(val_classif_gts)
val_classif_preds = clf.predict_proba(val_vCDRs)[:,1]
val_auc = classif_eval(val_classif_preds, val_classif_gts)

# Validation results
print('VALIDATION '+' '*50)
print('LOSSES: {:.4f} (val)'.format(val_loss))
print('OD segmentation (Dice Score): {:.4f} (val)'.format(val_dsc_od))
print('OC segmentation (Dice Score): {:.4f} (val)'.format(val_dsc_oc))
print('vCDR error: {:.4f} (val)'.format(val_vCDR_error))
print('Classification (AUC): {:.4f} (val)'.format(val_auc))
# Predictions on test set
nb_test_batches = len(test_loader)
model.eval()
test_vCDRs = []
with torch.no_grad():
    test_data = iter(test_loader)
    for k in range(nb_test_batches):
        # Loads data
        imgs = test_data.next()
        imgs = imgs.to(device)

        # Forward pass
        logits = model(imgs)

        # Std out
        print('Test iter {}/{}'.format(k+1, nb_test_batches) + ' '*50, 
              end='\r')
            
        # Compute segmentation
        pred_od = refine_seg((logits[:,0,:,:]>=0.5).type(torch.int8).cpu()).to(device)
        pred_oc = refine_seg((logits[:,1,:,:]>=0.5).type(torch.int8).cpu()).to(device)
            
        # Compute and store vCDRs
        pred_vCDR = vertical_cup_to_disc_ratio(pred_od.cpu().numpy(), pred_oc.cpu().numpy())
        test_vCDRs += pred_vCDR.tolist()
            

    # Glaucoma predictions from vCDRs
    test_vCDRs = np.array(test_vCDRs).reshape(-1,1)
    test_classif_preds = clf.predict_proba(test_vCDRs)[:,1]
    
# Prepare and save .csv file
def create_submission_csv(prediction, submission_filename='/kaggle/working/submission.csv'):
    """Create a sumbission file in the appropriate format for evaluation.

    :param
    prediction: list of predictions (ex: [0.12720, 0.89289, ..., 0.29829])
    """
    
    with open(submission_filename, mode='w') as csv_file:
        fieldnames = ['Id', 'Predicted']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for i, p in enumerate(prediction):
            writer.writerow({'Id': "T{:04d}".format(i+1), 'Predicted': '{:f}'.format(p)})

create_submission_csv(test_classif_preds)

# The submission.csv file is under /kaggle/working/submission.csv.
# If you want to submit it, you should download it before closing the current kernel.