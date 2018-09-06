import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pylab as plt

def predict(model, images, thres=0.7):
    '''
    make prediction on a set of input images (256 x 256 RGB image patches)
    '''

    # load tensorflow model
    with tf.Session(graph=tf.Graph()) as sess:
        print("load graph")
        with tf.gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        
        graph = tf.get_default_graph()
        
        # get tensorflow input and output
        x = graph.get_tensor_by_name("x:0")
        predict = graph.get_tensor_by_name("predict:0")
        
        # run prediction
        predict_val = sess.run(predict, feed_dict={x:images})

        return [decode_tensor(tensor, thres=thres) for tensor in predict_val]
 
def decode_tensor(tensor_val, thres=0.7, grid_num=5, num_types=4):
    '''
    translate tensor to coordinates (0 to 1), types and sizes (0 to 1) of trees
    '''
    mask = tensor_val[:,:,3] > thres # select by confidence value
    # coordinates (pixel)
    grid_coord = np.array([[[x,y] for x in range(grid_num)] for y in range(grid_num)])
    coord = ((tensor_val[:,:,0:2] + grid_coord) / grid_num)[mask]
    # type of trees
    types = np.argmax(tensor_val[:,:,4:4 + num_types],axis=2)[mask]
    # size of trees
    size = tensor_val[:,:,2][mask]
    
    return np.transpose([coord[:,0],coord[:,1], size, types])

def render(list_of_trees, img_size=256):
    '''
    render a list of trees (x, y, size, type)
    '''
    color_dict = {0:(255,0,0),1:(0,255,0),2:(0,0,255),3:(255,255,0)}
    img = np.ones((img_size,img_size,3),dtype=np.uint8) * 255
    for i in range(len(list_of_trees)):
        x = list_of_trees[i][0] * img_size
        y = list_of_trees[i][1] * img_size
        s = list_of_trees[i][2] * img_size / 1.414
        c = color_dict[list_of_trees[i][3]]

        cv2.rectangle(img,(int(x - s / 2), int(y - s / 2)),(int(x + s / 2),int(y + s / 2)),c,thickness=2)

    return img

images = np.array([(np.flip(cv2.imread("data/test/" + file),axis=2) - 128.0) / 128 for file in os.listdir("data/test")])
patches = predict("data/model.pb",images,0.7)
renders = [render(patch) for patch in patches]

for i in range(len(images)):
    print("patch",i)

    # print data
    [print("x",t[0],"| y", t[1],"| size",t[2],"| type",t[3]) for t in patches[i]]

    # show prediction
    plt.subplot(1,2,1)
    plt.imshow((images[i] + 1) / 2)

    plt.subplot(1,2,2)
    plt.imshow(renders[i] / 255.0)

    plt.show()