# south pacific aerial image
This is a project for the OpenAI challenge that aiming to detect and localize trees and streets from aerial imagery. The project consist of two CNN model for trees and street respectively, and some code for path optimization, as an example of putting these two parts together.
# The code
All the codes for this project are uploaded. The tree localizer was trained with tensorflow in python, and a prototype application of it was implemented in JAVA. The trained model was also uploaded as the .pb file. The street detector was trained and validated in Mathematica with the reference of Mathematica implementation of SegNet (link). The rest part (street filtering and path optimization) were also in Mathematica
# The training data
The training data are not provided with the code as they are just too large to upload. The data for tree localization is from the OpenAI challenge, in the format of raw aerial imagery and the geo-location of all trees. The data for street recognition is from ISPRS commission II/4 dataset.
# How to run
## Training the tree detector:
this will coming soon.

## Prototype of the treet detector:
The trained model is at [here](wait to upload)
<br></br>
copy all the files from "localizer_test" folder to your own JAVA project folder, having these packages in your build path:
<br>[processing 2.2.1](https://processing.org/download/)
<br>[tensorflow for java](https://www.tensorflow.org/install/install_java#using_tensorflow_with_jdk)
<br>[openCV 3.3](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.3.1/opencv-3.3.1-vc14.exe/download)
<br>[apache commons compress](http://commons.apache.org/proper/commons-compress/)
<br>[JTS 1.13](https://github.com/locationtech/jts)
<br></br>
The <b>MakePrediction.java</b> makes prediction on a full-size aerial imagery, and export the result as both csv file and equal-sized bounding box image. the csv file shows all the trees found in the imagery in pixel space, as (x, y, type). the rendered image can be overlapped with the input imagery for visualization. Set the variables: <b>root, satelliteImage, predictedCsv, rendering</b> to your input and output file and the <b>model.loadModel()</b> in <b>setup</b> function to where the .pb file is placed. Then launch the program and wait for the result.
<br>
</br>
The <b>TestTrainedModel_AnyPosition_ForVideo.java</b> is an interactive prototype that allows user to navigate through the input imagery and see the prediction at the location he/she points to. Similarly, set <b>file</b> to the location of input imagery and <b>model.loadModel()</b> in <b>setup</b> function to where the .pb model is placed.
<br></br>
## Trained model for street detector:
The trained model is at here
