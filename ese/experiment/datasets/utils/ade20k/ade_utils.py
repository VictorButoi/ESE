import numpy as np
from PIL import Image


def convertFromADE(img_np, lab_np):
    indexMapping = np.loadtxt('/local/vbutoi/projects/misc/research-code/ese/sceneparsing/convertFromADE/mapFromADE.txt').astype(int)
    # Ensure image and label are of the same dimensions
    assert img_np.shape[:2] == lab_np.shape[:2], "Image and label dimensions mismatch!"

    # Resize
    h, w = img_np.shape[:2]
    h_new, w_new = h, w

    if h < w and h > 512:
        h_new = 512
        w_new = int(round(w / h * 512))
    elif w < h and w > 512:
        h_new = int(round(h / w * 512))
        w_new = 512

    img_np = np.array(Image.fromarray(img_np).resize((w_new, h_new), Image.BILINEAR))
    lab_np_resized = np.array(Image.fromarray(lab_np).resize((w_new, h_new), Image.NEAREST))

    # Convert
    labOut_np = convert(lab_np_resized, indexMapping)

    return img_np, labOut_np

def convert(lab, indexMapping):
    # Resize
    h, w = lab.shape[:2]
    h_new, w_new = h, w

    if h < w and h > 512:
        h_new = 512
        w_new = int(round(w / h * 512))
    elif w < h and w > 512:
        h_new = int(round(h / w * 512))
        w_new = 512

    lab = np.array(Image.fromarray(lab).resize((w_new, h_new), Image.NEAREST))

    # Map index
    labADE = (lab[:, :, 0].astype(np.uint16) // 10) * 256 + lab[:, :, 1].astype(np.uint16)
    labOut = np.zeros(labADE.shape, dtype=np.uint8)

    classes_unique = np.unique(labADE)
    for cls in classes_unique:
        if np.sum(cls == indexMapping[:, 1]) > 0:
            labOut[labADE == cls] = indexMapping[cls == indexMapping[:, 1], 0]

    return labOut
