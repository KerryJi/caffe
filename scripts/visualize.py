
# coding: utf-8

# In[5]:

import numpy as np
import matplotlib.pyplot as plt
import cv2
#matplotlib inline
#　编写一个函数，用于显示各层数据
def show_data(data, padsize=1, padval=0):
    # data归一化
    data -= data.min()
    data /= data.max()
    
    # 根据data中图片数量data.shape[0]，计算最后输出时每行每列图片数n
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # padding = ((图片个数维度的padding),(图片高的padding), (图片宽的padding), ....)
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # 先将padding后的data分成n*n张图像
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    # 再将（n, W, n, H）变换成(n*w, n*H)
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.figure()
    plt.imshow(data,cmap='gray')
    plt.axis('off')
# Make sure that caffe is on the python path:
caffe_root = '/home/jkj/caffe/caffe-ssd/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import os
#if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#    print("Downloading pre-trained CaffeNet model...")
#    get_ipython().system(u'$caffe_root/scripts/download_model_binary.py $caffe_root/models/bvlc_reference_caffenet')


# In[6]:

threshold=0.2;
caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/VGGNet/coco/SSD_300x300/deploy.prototxt',
                caffe_root + 'models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_16000.caffemodel',
                caffe.TEST)


# In[7]:

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([104,117,123]) )#np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


# In[8]:
imagefile='/home/jkj/caffe/py-faster-rcnn/data/demo/004545.jpg' #caffe_root + 'examples/images/fish-bike.jpg'
net.blobs['data'].reshape(1,3,300,300)
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile))
out = net.forward()
#print("Predicted class is #{}.".format(out['prob'].argmax()))


# In[10]:

plt.imshow(transformer.deprocess('data', net.blobs['data'].data[0]))
imagenet_labels_filename = caffe_root + 'data/coco/labels.txt'
try:
    labels = np.loadtxt(imagenet_labels_filename, str, delimiter=',')
except:
    print "not found labels file"

# sort top k predictions from softmax output
predictions = net.blobs['detection_out'].data[0]
predictions.reshape(predictions.shape[1],predictions.shape[2])

scoredata=predictions[0,:,2] #np.array([i[2] for i in predictions[0]])
print predictions.shape,predictions.size
print scoredata

top_k = np.where(scoredata>threshold)[0] #scoredata.argsort()[-1:-6:-1]
print top_k
img=cv2.imread(imagefile)

for i in np.arange(top_k.size):
    label=predictions[0][top_k[i]][1]
    score=predictions[0][top_k[i]][2]
    x_min=int(predictions[0][top_k[i]][3]*img.shape[1])
    y_min=int(predictions[0][top_k[i]][4]*img.shape[0])
    x_max=int(predictions[0][top_k[i]][5]*img.shape[1])
    y_max=int(predictions[0][top_k[i]][6]*img.shape[0])
    print top_k[i], score,label,labels[label-1],x_min,y_min,x_max,y_max

    if score>threshold :
        cv2.putText(img,labels[label-1][2],(x_min,y_min),cv2.FONT_HERSHEY_SIMPLEX,0.4,[255,0,255])
        cv2.rectangle(img,(x_min,y_min),(x_max,y_max),[255,0,0])
    
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print net.blobs['conv1_1'].data[0].shape
show_data(net.blobs['conv1_1'].data[0])

print net.params['conv1_1'][0].data.shape
show_data(net.params['conv1_1'][0].data.reshape(64*3,3,3))

plt.show()


# In[ ]:




