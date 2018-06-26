import numpy as np
import tensorflow as tf
from PIL import Image
import os
from tensorflow.python.platform import gfile
import glob
import tensorlayer as tl


#const variable
TRAIN_DATA_DIR = 'training'
TEST_DATA_DIR = 'validation'

CROP_SIZE = 512
IMG_HEIGHT = 2848
IMG_WIDTH = 4288

def data_to_tfrecord(image_dir, filename, is_train=True):
    """ Save data into TFRecord 
        image_dir:string, training or validation
        filename:string, name of tfrecords
    """
    
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None

    file_list = []
    file_glob = os.path.join(image_dir, 'images', '*' + '.jpg')
    file_list.extend(glob.glob(file_glob))
    no_of_images = len(file_list)
    print ('No. of images files: %d' % no_of_images)
    
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return
    print("Converting data into %s ..." % filename)
    
    writer = tf.python_io.TFRecordWriter(filename)
    for f in file_list: 
        img = Image.open(f)
        _,img,_ = img.split() #extract green channel
        label_name = os.path.splitext(f.split("/")[-1])[0]+'_EX'
        label_path = os.path.join(f.split("/")[0],'labels',label_name+'.tif')
        lab = Image.open(label_path)
        img = np.expand_dims(np.array(img),2)
        lab = np.expand_dims(np.array(lab),2)
        #first resize original image
        resize_img = tl.prepro.imresize(img, (CROP_SIZE, CROP_SIZE ), interp='bilinear')
        resize_lab = tl.prepro.imresize(lab, (CROP_SIZE, CROP_SIZE ), interp='bilinear')
        #resize_img = np.squeeze(resize_img)
        #resize_lab = np.squeeze(resize_lab)
#        print(resize_img.shape)
#        print(resize_lab.shape)
        img_raw = resize_img.tobytes()
        lab_raw = resize_lab.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    "lab_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[lab_raw])),
                }
            )
        )
        writer.write(example.SerializeToString())  # Serialize To String
        
        if is_train == True:
        #if train, random crop 4 images
            for i in range(4):
                while True:
                    crop_img, crop_lab = tl.prepro.crop_multi([img,lab],CROP_SIZE, CROP_SIZE, is_random=True)
                    crop_img, crop_lab = tl.prepro.flip_axis_multi([crop_img,crop_lab], axis=1, is_random=True)
                    if np.count_nonzero(crop_lab)!=0:
                        break
                #print(crop_img.shape)
                img_raw = crop_img.tobytes()
                lab_raw = crop_lab.tobytes()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            "lab_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[lab_raw])),
                        }
                    )
                )
                writer.write(example.SerializeToString())  # Serialize To String 
    writer.close()


def read_and_decode(filename):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'lab_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [CROP_SIZE, CROP_SIZE, 1])
    lab = tf.decode_raw(features['lab_raw'], tf.uint8)
    lab = tf.cast(lab, tf.int32)
    lab = tf.reshape(lab, [CROP_SIZE, CROP_SIZE, 1])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
 
    return img, lab

if __name__=="__main__": 

    data_to_tfrecord(TRAIN_DATA_DIR,'train.tfrecords',is_train=True)
    data_to_tfrecord(TEST_DATA_DIR,'test.tfrecords',is_train=False)
    
    print('begin reading tfrecord')
    img, label = read_and_decode("train.tfrecords")

    
    img_batch, label_batch = tf.train.shuffle_batch([img, label],batch_size=1, capacity=10000,min_after_dequeue=2000)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(2):
            val, l= sess.run([img_batch, label_batch])
            #val, l= sess.run([img, label])
            val = np.squeeze(val)
            img=Image.fromarray(val)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save('img_'+str(i)+'.jpg')
            
            print(val.shape, l.shape)
    
        