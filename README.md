# Automated tree localization/classification and street detection from aerial imagery
This is a project for the [OpenAI challenge](https://werobotics.org/blog/2018/05/16/announcing-winners-open-ai-challenge/) that aimed to detect and localize trees and streets from aerial imagery. The project consist of two CNN model for trees and street respectively. And some code for path optimization, as an example of putting these two parts together. The video of the project is [here](https://www.youtube.com/watch?v=6w0wYUuKb4U)
# The code
All the codes for this project are uploaded. The tree localizer was trained with tensorflow in python, and a prototype application of it was implemented in JAVA. The trained model was also uploaded as a [.pb file](https://github.com/guozifeng91/south-pacific-aerial-image/tree/master/trained_model/tree_localization). The street detector was trained and validated in [Mathematica](https://www.wolfram.com/mathematica/) with the reference of [Mathematica implementation of SegNet](http://community.wolfram.com/groups/-/m/t/1250199). The rest part (street filtering and path optimization) were also in Mathematica
# The training data
The training data are not provided with the code as they are just too large to upload. The data for tree localization is from the OpenAI challenge, in the format of raw aerial imagery and the geo-location of all trees. The data for street recognition is from ISPRS commission II/4 dataset.
# How to run
we are working on a easy-use runable package so that you dont have to bother the following steps anymore
## Python example:
A simple python example is given [here](https://github.com/guozifeng91/south-pacific-aerial-image/tree/master/localizer%20python)(It uses the same .pb file with the java example).
Download the .py file and the data folder, put them together in a new folder, then type in terminal: <b>python tree_detect.py</b>. It will search for all the images in data/test and run prediction for each image, the results will look like this<br>

![alt text](https://github.com/guozifeng91/south-pacific-aerial-image/blob/master/localizer%20python/data/example1.png)
![alt text](https://github.com/guozifeng91/south-pacific-aerial-image/blob/master/localizer%20python/data/example2.png)

dependence: [tensorflow](https://www.tensorflow.org/), [numpy](http://www.numpy.org/), [opencv (cv2)](https://pypi.org/project/opencv-python/), [matplotlib](https://matplotlib.org/)<br>

## JAVA app prototype:
Please follow this [file](https://github.com/guozifeng91/south-pacific-aerial-image/blob/master/guide/guide.pdf) for details<br>

The trained model is [here](https://github.com/guozifeng91/south-pacific-aerial-image/tree/master/trained_model/tree_localization) in .pb format, which is a serialized tensorflow graph definition with all variables as constants. Loading it restores both the graph definition and the weights. The method on how to operate it can be found [here](https://www.tensorflow.org/mobile/prepare_models#using_the_graph_transform_tool) and [here](https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/tools/freeze_graph.py)<br>

Copy all the files from "localizer_test" folder to your own JAVA project folder, having these packages in your build path:<br>

[processing 2.2.1](https://processing.org/download/)<br>
[tensorflow for java](https://www.tensorflow.org/install/install_java#using_tensorflow_with_jdk)<br>
[openCV 3.3](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.3.1/opencv-3.3.1-vc14.exe/download)<br>
[apache commons compress](http://commons.apache.org/proper/commons-compress/)<br>
[JTS 1.13](https://github.com/locationtech/jts)<br>

The <b>MakePrediction.java</b> makes prediction on a full-size aerial imagery, and export the result as both csv file and equal-sized bounding box image. the csv file shows all the trees found in the imagery in pixel space, as (x, y, type). the rendered image can be overlapped with the input imagery for visualization. Set the variables: <b>root, satelliteImage, predictedCsv, rendering</b> to your input and output file and the <b>model.loadModel()</b> in <b>setup</b> function to where the .pb file is placed. Then launch the program and wait for the result.<br>

```
String root = "where your imagery is, end with \\";
String satelliteImage = root + "name of input imagery";
String predictedCsv = root + "name of output.csv";
String rendering = root + "name of output.jpg";

model.loadModel("path, end with \\", "name.pb");
```

The <b>TestTrainedModel_AnyPosition_ForVideo.java</b> is an interactive prototype that allows user to navigate through the input imagery and see the prediction at the location he/she points to. Similarly, set <b>file</b> to the location of input imagery and <b>model.loadModel()</b> in <b>setup</b> function to where the .pb model is placed.

## Street detection:

The trained model is [here](https://github.com/guozifeng91/south-pacific-aerial-image/tree/master/trained_model/street_segmentation), as the original file is too large to upload, it is packed in zip patches. Download all the files and unzip the <b>street segmentation model.zip</b> gives you the trained model (.wlnet) format<br>

To run the model in Mathematica, see file [ValidateModel.nb](https://github.com/guozifeng91/south-pacific-aerial-image/blob/master/segmentation/ValidateModel.nb)
