import numpy as np;
import tensorflow as tf;
from tensorflow.python.saved_model import tag_constants;

import time

'''
the working version of yolo based localizer
'''

BATCH_SIZE = 16;
EPOCHS = 200;
SHUFFLE_SEED = 3;
NUM_X = 137;  # currently a subset, change to 137 for complete dataset
NUM_Y = 194;  # currently a subset, change to 194 for complete dataset
SAMPLE_SIZE = NUM_X * NUM_Y;


def lrelu(x, alpha, name):
    return tf.subtract(tf.nn.relu(x), alpha * tf.nn.relu(-x), name=name);


def conv_valid_leaky(name, x, kernel_size_h, kernel_size_w, input_channel, output_channel, alpha=0.01, stride_h=1, stride_w=1, stddev=None):
    '''
    add conv2d layer with VALID padding method:
    
    output_size = (input_size - kernel_size) / stride + 1
    '''
    mean = 0.1;
    
    if (stddev is None):
        stddev = 1.0 / np.sqrt(kernel_size_h * kernel_size_w) * 1.41421;  # He initialization (stddev = 1/sqrt(n/2))
        
    w = tf.get_variable(name + "_w", shape=[kernel_size_h, kernel_size_w, input_channel, output_channel], initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev));
    b = tf.get_variable(name + "_b", shape=(output_channel), initializer=tf.constant_initializer(0.1));
    return lrelu(tf.nn.conv2d(x, w, strides=(1, stride_h, stride_w, 1), padding='VALID') + b, alpha=alpha, name=name);


def conv_valid_tanh(name, x, kernel_size_h, kernel_size_w, input_channel, output_channel, stride_h=1, stride_w=1, stddev=None):
    '''
    add conv2d layer with VALID padding method:
    
    output_size = (input_size - kernel_size) / stride + 1
    '''
    mean = 0;
    if (stddev is None):
        stddev = 1.0 / np.sqrt(kernel_size_h * kernel_size_w);
        
    w = tf.get_variable(name + "_w", shape=[kernel_size_h, kernel_size_w, input_channel, output_channel], initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev));
    b = tf.get_variable(name + "_b", shape=(output_channel), initializer=tf.constant_initializer(0.1));
    return tf.nn.tanh(tf.nn.conv2d(x, w, strides=(1, stride_h, stride_w, 1), padding='VALID') + b, name=name);


def conv_valid(name, x, kernel_size_h, kernel_size_w, input_channel, output_channel, af=tf.nn.sigmoid, stride_h=1, stride_w=1):
    '''
    add conv2d layer with VALID padding method:
    
    output_size = (input_size - kernel_size) / stride + 1
    '''
    mean = 0;
    stddev = 1.0 / np.sqrt(kernel_size_h * kernel_size_w);
    if (af == tf.nn.relu):
        stddev = stddev * 1.41421;  # He initialization (stddev = 1/sqrt(n/2))
        mean = 1.0;  # / (kernel_size_h * kernel_size_w);  # necessary?
        
    w = tf.get_variable(name + "_w", shape=[kernel_size_h, kernel_size_w, input_channel, output_channel], initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev));
    b = tf.get_variable(name + "_b", shape=(output_channel), initializer=tf.constant_initializer(0.1));
    return af(tf.nn.conv2d(x, w, strides=(1, stride_h, stride_w, 1), padding='VALID') + b, name=name);


def max_pooling_valid(name, x, kernel_size_h, kernel_size_w):
    return tf.nn.max_pool(x, [1, kernel_size_h, kernel_size_w, 1], [1, kernel_size_h, kernel_size_w, 1], padding='VALID', name=name)


def deconv_valid(name, x, batch_size, kernel_size_h, kernel_size_w, input_channel, output_channel, output_size, af=tf.nn.sigmoid, stride_h=1, stride_w=1):
    '''
    add deconv2d layer with VALID padding method:
    
    output_size = (input_size - 1) * stride + kernel_size.
    
    note that since conv2d_transpose does not accept
    batch size of -1, the batch_size should be either
    
    1. tf.shape(x)[0]
    
    or
    
    2. tf.stride_slice(tf.shape(x), [0], [1]), where x is the placeholder;
    
    where x is the placeholder of input
    '''
    mean = 0;
    stddev = 1.0 / np.sqrt(kernel_size_h * kernel_size_w);
    if (af == tf.nn.relu):
        stddev = stddev * 1.41421;
        mean = 1.0 / (kernel_size_h * kernel_size_w);
        
    w = tf.get_variable(name + '_w', shape=(kernel_size_h, kernel_size_w, output_channel, input_channel), initializer=tf.random_normal_initializer(mean=mean, stddev=stddev));
    b = tf.get_variable(name + '_b', shape=(output_channel,), initializer=tf.constant_initializer(0));
    
    return af(tf.nn.conv2d_transpose(x, w, output_shape=(batch_size, output_size, output_size, output_channel), strides=(1, stride_h, stride_w, 1), padding='VALID') + b, name=name);


def full_connect_layer(name, inputs, in_size, out_size, af=None):
    '''
    matrix multiplication as full connect layer
    '''
    
    mean = 0;
    stddev = 1.0 / np.sqrt(in_size);
    if (af == tf.nn.relu):
        stddev = stddev * 1.41421;
        mean = 1.0 / (in_size);
    
    weights = tf.Variable(tf.random_normal([in_size, out_size], mean=mean, stddev=stddev), name=name + "_w");
    biases = tf.Variable(tf.zeros([1, out_size]), name=name + "_b");
    if af is None:
        wx_plus_b = tf.add(tf.matmul(inputs, weights), biases, name=name);
        return wx_plus_b;
    else:
        wx_plus_b = tf.add(tf.matmul(inputs, weights), biases, name=name + "_add");
        return af(wx_plus_b, name=name);


def createFileQueue(filenames):
    '''
    create the operation that load files.
    
    shuffle can be done in two ways:
    
    1. shuffle in the queue, then use tf.train.batch (in one epoch, each data will be processed once)
    
    2. no shuffle in the queue, then use tf.train.shuffle_batch (in one epoch, some data may be processed many times)
    '''
    # make a filename queuer
    filename_queue = tf.train.string_input_producer(filenames, shuffle=True, seed=SHUFFLE_SEED, num_epochs=EPOCHS)
    # make a whole file reader
    file_reader = tf.WholeFileReader()
    # read the file(return key and value)
    key, value = file_reader.read(filename_queue)
    # image = tf.image.decode_png(value) # image is decoded from value
    return filename_queue, key, value;


def createPNGQueue(filenames):
    _, _, value = createFileQueue(filenames);
    image = tf.image.decode_png(value);  # image is decoded from value
    # or use resize image:
    # tf.image.resize_images(images, sizeTo, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.reshape(image, shape=[256, 256, 3]);  # or resize
    return image;

    
def createJPGQueue(filenames):
    _, _, value = createFileQueue(filenames);
    image = tf.image.decode_jpeg(value);  # image is decoded from value
    # or use resize image:
    # tf.image.resize_images(images, sizeTo, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.reshape(image, shape=[256, 256, 3]);  # or resize
    return image;


def createCSVQueue(filenames, csv_shape):
    csv_length = 1;
    for i in csv_shape:
        csv_length = csv_length * i;
    
    _, _, value = createFileQueue(filenames);
    default_csv = [];
    for _ in range(csv_length):
        default_csv.append([0.0]);
    csv = tf.decode_csv(value, default_csv, field_delim=",");
    csv = tf.reshape(csv, csv_shape);
    return csv;


def loadData():
    '''
    load the training data
    '''
    # a constant that matches the size of the csv
    csv_shape = [5, 5, 8];
    root = "../../data/train/";
    
    file_img = [];
    file_csv = [];
    for x in range(NUM_X):
        for y in range(NUM_Y):
            filename = str(x) + "_" + str(y);
            file_img.append(root + "image/" + filename + ".jpg");
            file_csv.append(root + "csv/" + filename + ".csv");
    
    images = createJPGQueue(file_img);   
    csvs = createCSVQueue(file_csv, csv_shape);
    
    # for debug: show the file name rather than the content
#     _, images, _ = createFileQueue(file_img);
#     _, csvs, _ = createFileQueue(file_csv);    
    return tf.train.batch([images, csvs], batch_size=BATCH_SIZE);



def arch_1(x):
    # 256-5+1 = 252
    prev = conv_valid("c1", x, kernel_size_h=5, kernel_size_w=5, input_channel=3, output_channel=16);
    # 84
    prev = max_pooling_valid("p1", prev, kernel_size_h=3, kernel_size_w=3);
    print(prev);
     
    # 80
    prev = conv_valid("c2", prev, kernel_size_h=5, kernel_size_w=5, input_channel=16, output_channel=32);
    # 40
    prev = max_pooling_valid("p2", prev, kernel_size_h=2, kernel_size_w=2);
    print(prev);
     
    # 36
    prev = conv_valid("c3", prev, kernel_size_h=5, kernel_size_w=5, input_channel=32, output_channel=64);
    # 18
    prev = max_pooling_valid("p3", prev, kernel_size_h=2, kernel_size_w=2);
    print(prev);
     
    # 14
    prev = conv_valid("c4", prev, kernel_size_h=5, kernel_size_w=5, input_channel=64, output_channel=128);
    # 7
    prev = max_pooling_valid("p4", prev, kernel_size_h=2, kernel_size_w=2);
    print(prev);
     
    # 3
    prev = conv_valid("c5", prev, kernel_size_h=5, kernel_size_w=5, input_channel=128, output_channel=256);
    # 1
    prev = conv_valid("c6", prev, kernel_size_h=3, kernel_size_w=3, input_channel=256, output_channel=512);
    print(prev);
     
    prev = tf.reshape(prev, [-1, 512]);
    print(prev);
    prev = full_connect_layer("f1", prev, 512, 2048, af=tf.nn.sigmoid);
    prev = full_connect_layer("f2", prev, 2048, 5 * 5 * 8, af=tf.nn.sigmoid);
    prev = tf.reshape(prev, [-1, 5, 5, 8], name="predict");
    print(prev);
    
    return prev;

def arch_1_leakyRelu(x):
    # 256-5+1 = 252
    prev = conv_valid_leaky("c1", x, kernel_size_h=5, kernel_size_w=5, input_channel=3, output_channel=16);
    # 84
    prev = max_pooling_valid("p1", prev, kernel_size_h=3, kernel_size_w=3);
    print(prev);
     
    # 80
    prev = conv_valid_leaky("c2", prev, kernel_size_h=5, kernel_size_w=5, input_channel=16, output_channel=32);
    # 40
    prev = max_pooling_valid("p2", prev, kernel_size_h=2, kernel_size_w=2);
    print(prev);
     
    # 36
    prev = conv_valid_leaky("c3", prev, kernel_size_h=5, kernel_size_w=5, input_channel=32, output_channel=64);
    # 18
    prev = max_pooling_valid("p3", prev, kernel_size_h=2, kernel_size_w=2);
    print(prev);
     
    # 14
    prev = conv_valid_leaky("c4", prev, kernel_size_h=5, kernel_size_w=5, input_channel=64, output_channel=128);
    # 7
    prev = max_pooling_valid("p4", prev, kernel_size_h=2, kernel_size_w=2);
    print(prev);
     
    # 3
    prev = conv_valid_leaky("c5", prev, kernel_size_h=5, kernel_size_w=5, input_channel=128, output_channel=256);
    # 1
    prev = conv_valid_leaky("c6", prev, kernel_size_h=3, kernel_size_w=3, input_channel=256, output_channel=512);
    print(prev);
     
    prev = tf.reshape(prev, [-1, 512]);
    print(prev);
    prev = full_connect_layer("f1", prev, 512, 2048, af=tf.nn.tanh);
    prev = full_connect_layer("f2", prev, 2048, 5 * 5 * 8, af=tf.nn.sigmoid);
    prev = tf.reshape(prev, [-1, 5, 5, 8], name="predict");
    print(prev);
    
    return prev;

def arch_1_tanh(x):
    '''
    architecture that currently working
    '''
    # 256-5+1 = 252
    prev = conv_valid_tanh("c1", x, kernel_size_h=5, kernel_size_w=5, input_channel=3, output_channel=16);
    # 84
    prev = max_pooling_valid("p1", prev, kernel_size_h=3, kernel_size_w=3);
    print(prev);
     
    # 80
    prev = conv_valid_tanh("c2", prev, kernel_size_h=5, kernel_size_w=5, input_channel=16, output_channel=32);
    # 40
    prev = max_pooling_valid("p2", prev, kernel_size_h=2, kernel_size_w=2);
    print(prev);
     
    # 36
    prev = conv_valid_tanh("c3", prev, kernel_size_h=5, kernel_size_w=5, input_channel=32, output_channel=64);
    # 18
    prev = max_pooling_valid("p3", prev, kernel_size_h=2, kernel_size_w=2);
    print(prev);
     
    # 14
    prev = conv_valid_tanh("c4", prev, kernel_size_h=5, kernel_size_w=5, input_channel=64, output_channel=128);
    # 7
    prev = max_pooling_valid("p4", prev, kernel_size_h=2, kernel_size_w=2);
    print(prev);
     
    # 3
    prev = conv_valid_tanh("c5", prev, kernel_size_h=5, kernel_size_w=5, input_channel=128, output_channel=256);
    # 1
    prev = conv_valid_tanh("c6", prev, kernel_size_h=3, kernel_size_w=3, input_channel=256, output_channel=512);
    print(prev);
     
    prev = tf.reshape(prev, [-1, 512]);
    print(prev);
    prev = full_connect_layer("f1", prev, 512, 2048, af=tf.nn.tanh);
    prev = full_connect_layer("f2", prev, 2048, 5 * 5 * 8, af=tf.nn.sigmoid);
    prev = tf.reshape(prev, [-1, 5, 5, 8], name="predict");
    print(prev);
    
    return prev;


def buildAndTrain():
    # input
    x = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name="x");
    y = tf.placeholder(dtype=tf.float32, shape=[None, 5, 5, 8], name="y");
    
    # architecture of the model
    prev = arch_1_tanh(x);
    
    # loss function
    conf_predict = prev[:, :, :, 3:4];
    conf_y = y[:, :, :, 3:4];
     
    box_predict = prev[:, :, :, 0:3];
    box_y = y[:, :, :, 0:3];
     
    cls_predict = prev[:, :, :, 4:8];
    cls_y = y[:, :, :, 4:8];
     
    loss_conf = 10 * tf.reduce_sum(tf.square(conf_predict - conf_y));
    loss_box = tf.reduce_sum(conf_y * tf.square(box_predict - box_y));  # conf_y is binary 
    loss_cls = tf.reduce_sum(conf_y * tf.square(cls_predict - cls_y));  # conf_y is binary 
     
    loss = tf.add(tf.add(loss_conf, loss_cls), loss_box, name="loss");
    
    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss);
    
    # build data-reading graph
    batch_images, batch_csvs = loadData();
    
    # yolo_validata is trained  with all data for 30 rounds, the loss approx. 7, it works
    builder = tf.saved_model.builder.SavedModelBuilder("../../data/models/yolo_as_final_2");
    
    start_time = int(round(time.time()));
    
    with tf.Session() as sess:
        # initiate the variables within the string_input_producer
        tf.local_variables_initializer().run();
        tf.global_variables_initializer().run();
        # start queue runners as a thread
        # (see http://blog.csdn.net/buptgshengod/article/details/72956846)
        coord = tf.train.Coordinator();
        enqueue_threads = tf.train.start_queue_runners(sess, coord);
        
        # watch the size(range) here!
        for epoch in range(EPOCHS):
            print("epoch", epoch, "batches");
            loss_val = 0;
            batch_num = SAMPLE_SIZE // BATCH_SIZE;
            for i in range(batch_num):
                val_images, val_csvs = sess.run([batch_images, batch_csvs]);
                
                val_images = (val_images - 128.0) / 128.0; # normalize
                
                sess.run(train_op, feed_dict={x:val_images, y:val_csvs});
                 
                if (epoch % 1 == 0):
                    batch_loss = sess.run(loss, feed_dict={x:val_images, y:val_csvs});
                    loss_val = loss_val + batch_loss;
                    if (i % 64 == 0):
                        print(i + 1, "of", batch_num);
                        cur_time = int(round(time.time())) - start_time;
                        print("    time elapsed", formatTime(cur_time));
                        percent = ((epoch * batch_num + i) / (EPOCHS * batch_num));
                        if (percent > 0):
                            time_remain = int(cur_time / percent - cur_time);
                            print("    time left", formatTime(time_remain));
                    
            if (epoch % 1 == 0):
                print("epoch", epoch, "loss", loss_val / SAMPLE_SIZE);
         
        coord.request_stop();
        coord.join(enqueue_threads);
        
        builder.add_meta_graph_and_variables(sess, [tag_constants.TRAINING]);
        builder.save(as_text=True);


def formatTime(sec):
    hour = sec // 3600;
    sec = sec - hour * 3600;
    minu = sec // 60;
    sec = sec - minu * 60;
    return  str(hour) + "h " + str(minu) + "m " + str(sec) + "s";


def test_model(model_name):
    '''
    test the trained model.
    '''
    
    # the name of the image for testing
    test_img_name = "1_70";
    
    IMAGE_SIZE = 256
    GRID_NUM = 5;
    
    with tf.Session(graph=tf.Graph()) as sess:
        
        root = "D:\\dataset\\tree detection\\"
        img_name = root + "image\\" + test_img_name + ".jpg";
        label_name = root + "csv\\" + test_img_name + ".csv";
        
        image = tf.read_file(img_name)
        image = tf.image.decode_jpeg(image);
        image = tf.reshape(image, shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3]);  # or resize
        
        label = np.loadtxt(label_name, delimiter=",").reshape((1, GRID_NUM, GRID_NUM, 8));
    
        print("load graph")
        with tf.gfile.FastGFile(model_name, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        
        graph = tf.get_default_graph();
        
        x = graph.get_tensor_by_name("x:0");
        predict = graph.get_tensor_by_name("predict:0");
        
        img_val = sess.run(image);
        
        print(img_val.dtype)
        print(img_val[0,0,0])
        img_val = (img_val - 128.0)
        print(img_val[0,0,0])
        
        img_val = (img_val / 128.0)
        print(img_val[0,0,0])
        
        predict_val = sess.run(predict, feed_dict={x:img_val});
        
        conf_predict = predict_val[:, :, :, 4:8];
        conf_y = label[:, :, :, 4:8];
        
        np.set_printoptions(precision=2, suppress=True);
        print(conf_predict);
        print(conf_y);


def convert_to_pb_model(model_name, output_name, save_name):
    '''
    write the trained model as a .pb file so it can be distributed
    '''
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tag_constants.TRAINING], model_name);
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_name);
        with tf.gfile.FastGFile(save_name, 'wb') as f:
            f.write(output_graph_def.SerializeToString());

            
if __name__ == '__main__':
    '''
    validate the models
    
    '''

#epoch 199 loss 1.383511925209671
    buildAndTrain();
#     test_model("../../data/models/yolo_as_final_2.pb");
#     convert_to_pb_model("../../data/models/yolo_as_final_2", ["predict"], "../../data/models/yolo_as_final_2.pb")
