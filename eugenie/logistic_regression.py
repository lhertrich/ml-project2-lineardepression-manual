import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


root_dir = "data/training/"
image_dir = root_dir + "augmented/images/"
gt_dir = root_dir + "augmented/masks/"


# Helper functions
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches

# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0, 1))
    feat_v = np.var(img, axis=(0, 1))
    feat = np.append(feat_m, feat_v)
    return feat


# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat


# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray(
        [extract_features_2d(img_patches[i]) for i in range(len(img_patches))]
    )
    return X

foreground_threshold = (
    0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
)


def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0
    
# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j : j + w, i : i + h] = labels[idx]
            idx = idx + 1
    return im


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img



if __name__ == "__main__":

    files = os.listdir(image_dir)
    n =len(files)
    imgs = [load_image(image_dir + files[i]) for i in range(n)]

    gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

    # Ensure equal number of images and masks
    assert len(imgs) == len(gt_imgs), "Number of images and masks do not match!"

    # Split the dataset into training and test sets
    train_images, test_images, train_masks, test_masks = train_test_split(
        imgs, gt_imgs, test_size=0.2, random_state=42
    )

    # Extract patches from input images
    patch_size = 16  # each patch is 16*16 pixels
    n_train = len(train_images)
    img_patches = [img_crop(train_images[i], patch_size, patch_size) for i in range(n_train)]
    gt_patches = [img_crop(train_masks[i], patch_size, patch_size) for i in range(n_train)]

    n_test = len(test_images)
    test_img_patches = [img_crop(test_images[i], patch_size, patch_size) for i in range(n_test)]
    test_gt_patches = [img_crop(test_masks[i], patch_size, patch_size) for i in range(n_test)]

    # Linearize list of patches
    img_patches = np.asarray(
        [
            img_patches[i][j]
            for i in range(len(img_patches))
            for j in range(len(img_patches[i]))
        ]
    )
    gt_patches = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )

    test_img_patches = np.asarray(
        [
            test_img_patches[i][j]
            for i in range(len(test_img_patches))
            for j in range(len(test_img_patches[i]))
        ]
    )
    test_gt_patches = np.asarray(
        [
            test_gt_patches[i][j]
            for i in range(len(test_gt_patches))
            for j in range(len(test_gt_patches[i]))
        ]
    )

    # Compute features for each image patch
    X = np.asarray([extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    X_test = np.asarray([extract_features_2d(test_img_patches[i]) for i in range(len(test_img_patches))])
    Y_test = np.asarray([value_to_class(np.mean(test_gt_patches[i])) for i in range(len(test_gt_patches))])

    # we create an instance of the classifier and fit the data
    logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
    logreg.fit(X, Y)

    # Predict on the training set
    Z = logreg.predict(X_test)

    # Get non-zeros in prediction and grountruth arrays
    Zn = np.nonzero(Z)[0]
    Yn = np.nonzero(Y_test)[0]

    TPR = len(list(set(Yn) & set(Zn))) / float(len(Z))
    print("True positive rate = " + str(TPR))

    F1_score = f1_score(Y_test, Z)
    print("F1 score =" + str(F1_score))