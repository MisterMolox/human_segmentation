from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np

def encode_rle(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)

def decode_rle(rle_mask, shape=(320, 240)):
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for low, high in zip(starts, ends):
        img[low:high] = 1

    return img.reshape(shape)
    


model = load_model("foto_weights.h5", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
imgs_mask_valid = model.predict(x_valid, verbose=1)
imgs_mask_valid *= 255

mask = (imgs_mask_valid>120)
imgs_mask_valid[mask] = 255
imgs_mask_valid[np.logical_not(mask)] = 0
    
with open('/content/drive/My Drive/foo.csv','a') as fd:
  fd.write('id')
  fd.write(',')
  fd.write('rle_mask')
  fd.write('\n')
  
for i in range(145):
  rle_mask = encode_rle(imgs_mask_valid[i])
  with open('/content/drive/My Drive/foo.csv','a') as fd:
        fd.write(str(1315+i))
        fd.write(',')
        fd.write(rle_mask)
        fd.write('\n')    
