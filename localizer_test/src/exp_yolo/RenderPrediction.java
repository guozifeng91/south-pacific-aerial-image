package exp_yolo;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * render the predicted csv into a equal size image so that it can be overlapped
 * with the original satellite image for validation and comparison
 * 
 * @author guozifeng
 *
 */
public class RenderPrediction {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String[] args) {
		String root = "data\\validation\\";
		String[] treePredictionCsv = new String[] { root + "DSC09013_geotag.csv" };
		int width = 6000, height = 4000;
		String imageTo = root + "DSC09013_geotag_prediction.jpg";
		

//		int width = 17761, height = 25006;
//		String imageTo = root + "label.jpg";
//		String[] treePredictionCsv = new String[] { root + "eval_label_paired.csv", root + "eval_label_rest.csv" };

//		int width = 15337, height = 10722;
//		String imageTo = root + "validation_prediction.jpg";
//		String[] treePredictionCsv = new String[] { root + "validation\\prediction.csv" };

		/*
		 * color of different type of trees
		 */
		byte[][] treeColor = new byte[][] {

				new byte[] { (byte) 255, (byte) 0, (byte) 0 },

				new byte[] { (byte) 0, (byte) 255, (byte) 0 },

				new byte[] { (byte) 0, (byte) 0, (byte) 255 },

				new byte[] { (byte) 255, (byte) 255, (byte) 0 } };
		/*
		 * type of different type of trees
		 */
		int[] treeSize = new int[] { 70, 100, 160, 90 };

		int lineWidth = 5;

		Mat matImg = new Mat(height, width, CvType.CV_8UC3);
		/* set all values to 255 (white color) */
		Core.add(matImg, Scalar.all(255), matImg);

		try {
			for (String csv : treePredictionCsv) {
				BufferedReader reader = new BufferedReader(new FileReader(csv));
				String line;
				while ((line = reader.readLine()) != null) {
					String[] subStr = line.split(",");
					if (subStr.length == 3) {
						int x = Integer.valueOf(subStr[0]);
						int y = Integer.valueOf(subStr[1]);
						if (x >= 0 && x < width && y >= 0 && y < height) {
							int type = Integer.valueOf(subStr[2]);
							int size = treeSize[type];

							int startX = Math.max(0, x - size / 2);
							int startY = Math.max(0, y - size / 2);

							int endX = Math.min(width, x + size / 2);
							int endY = Math.min(height, y + size / 2);

							int widthSub = endX - startX;
							int heightSub = endY - startY;

							Mat subMatImg = matImg.submat(startY, endY, startX, endX);

							byte[] bytes = new byte[3 * widthSub * heightSub];

							/* copy the existing part of the image */
							for (int i = 0; i < widthSub; i++) {
								for (int j = 0; j < heightSub; j++) {
									double[] pixel = subMatImg.get(j, i);
									int index = (j * widthSub + i) * 3;

									bytes[index] = (byte) pixel[0];
									bytes[index + 1] = (byte) pixel[1];
									bytes[index + 2] = (byte) pixel[2];
								}
							}

							/* draw vertical line */
							for (int w = 0; w < lineWidth; w++) {
								int i = x - size / 2 + w - startX;
								if (i >= 0) {
									for (int j = 0; j < heightSub; j++) {
										int index = (j * widthSub + i) * 3;
										bytes[index] = treeColor[type][2];
										bytes[index + 1] = treeColor[type][1];
										bytes[index + 2] = treeColor[type][0];
									}
								}

								i = x + size / 2 - 1 - w - startX;
								if (i < widthSub) {
									for (int j = 0; j < heightSub; j++) {
										int index = (j * widthSub + i) * 3;
										bytes[index] = treeColor[type][2];
										bytes[index + 1] = treeColor[type][1];
										bytes[index + 2] = treeColor[type][0];
									}
								}
							}

							/* draw horizontal line */
							for (int w = 0; w < lineWidth; w++) {
								int j = y - size / 2 + w - startY;
								if (j >= 0) {
									for (int i = 0; i < widthSub; i++) {
										int index = (j * widthSub + i) * 3;
										bytes[index] = treeColor[type][2];
										bytes[index + 1] = treeColor[type][1];
										bytes[index + 2] = treeColor[type][0];
									}
								}

								j = y + size / 2 - 1 - w - startY;
								if (j < heightSub) {
									for (int i = 0; i < widthSub; i++) {
										int index = (j * widthSub + i) * 3;
										bytes[index] = treeColor[type][2];
										bytes[index + 1] = treeColor[type][1];
										bytes[index + 2] = treeColor[type][0];
									}
								}
							}

							subMatImg.put(0, 0, bytes);
						}
					}
				}

				Imgcodecs.imwrite(imageTo, matImg);
				reader.close();
			}
		} catch (IOException e) {

		}
	}

}
