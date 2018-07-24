package exp_yolo;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

import processing.core.PApplet;
import processing.core.PImage;

/**
 * recognize all trees from a given aerial image, save the result as a csv and render it.
 */
public class MakePrediction extends PApplet {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	String root = "data\\"; // the path of the data folder

	String satelliteImage = root + "validation\\DSC09889_geotag.jpg"; // source image
	String predictedCsv = root + "validation\\DSC09889_geotag.csv"; // target csv file
	String rendering = root + "validation\\DSC09889_geotag" + "_pre.jpg"; // target rendering

	ArrayList<int[]> predictedTrees = new ArrayList<int[]>();

	int patchSize = 256;
	int gridNum = 5;

	TrainedModel model = new TrainedModel(gridNum, 8);
	// parameters to draw the grids
	float gridSize = (float) patchSize / gridNum;
	int[] colorByType = new int[] { 0xffff0000, 0xff00ff00, 0xff0000ff, 0xffffff00 };

	Mat imageFull; // the original satellite image
	int cols, rows; // size of the satellite image

	double threshold = 0.7; // threshold of the confidence value

	private void loadTrainingData() {
		imageFull = Imgcodecs.imread(satelliteImage, Imgcodecs.IMREAD_COLOR);
		cols = imageFull.cols();
		rows = imageFull.rows();
	}

	private void evaluate_batch(int moveStep, float threshold, boolean norm1, boolean norm2, int batch_size) {
		ArrayList<float[][][]> batches = new ArrayList<float[][][]>();
		ArrayList<int[]> batches_position = new ArrayList<int[]>();
		for (int x = 0; x + patchSize < cols; x += moveStep) {
			for (int y = 0; y + patchSize < rows; y += moveStep) {
				PImage pimg = getPatchImage(x, y, patchSize);
				if (pimg == null)
					continue;
				batches.add(toVector(pimg, norm1, norm2));
				batches_position.add(new int[] { x, y });

				/* batch_prediction */
				if (batches.size() >= batch_size) {
					float[][][][] batch_img = batches.toArray(new float[batches.size()][][][]);
					float[][] batch_prediction = model.predictFlatten(batch_img, batch_img.length);

					get_prediction(batches_position.toArray(new int[batches_position.size()][]), batch_prediction);
					batches.clear();
					batches_position.clear();

					System.out.println(x + "-" + y);
				}
			}
		}

		/* batch_prediction, last batch */
		if (batches.size() > 0) {
			float[][][][] batch_img = batches.toArray(new float[batches.size()][][][]);
			float[][] batch_prediction = model.predictFlatten(batch_img, batch_img.length);

			get_prediction(batches_position.toArray(new int[batches_position.size()][]), batch_prediction);
			batches.clear();
			batches_position.clear();
		}
	}

	private void get_prediction(int[][] batch_position, float[][] batch_prediction) {
		assert batch_position.length == batch_prediction.length;
		int len = batch_position.length;
		for (int id = 0; id < len; id++) {
			float[] predictVector = batch_prediction[id];
			int x = batch_position[id][0];
			int y = batch_position[id][1];

			/* get prediction */
			for (int i = 0; i < gridNum; i++) {
				for (int j = 0; j < gridNum; j++) {
					int indexGrid = (j * gridNum + i) * (4 + 4);
					float confidence = predictVector[indexGrid + 3];

					/* confidence value exceeds threshold? */
					if (confidence > threshold) {
						/* location prediction */
						float center_x = (float) predictVector[indexGrid];
						float center_y = (float) predictVector[indexGrid + 1];

						center_x *= gridSize;
						center_y *= gridSize;

						center_x += i * gridSize;
						center_y += j * gridSize;

						center_x += x;
						center_y += y;

						/* size prediction */
						float tree_size = (float) predictVector[indexGrid + 2];
						tree_size *= patchSize;

						/* type */
						int type = 0;
						for (int k = 1; k < 4; k++) {
							if (predictVector[indexGrid + 4 + k] > predictVector[indexGrid + 4 + type]) {
								type = k;
							}
						}

						predictedTrees.add(new int[] { (int) center_x, (int) center_y, type });
					}
				}
			}
		}
	}

	private void eleminateOverlap(boolean sameTypeOnly, float threshold) {
		ArrayList<int[]> newList = new ArrayList<int[]>();

		int len = predictedTrees.size();
		boolean[] overlap = new boolean[len];

		for (int i = 0; i < len - 1; i++) {
			int[] a = predictedTrees.get(i);
			for (int j = i + 1; j < len; j++) {
				int[] b = predictedTrees.get(j);

				if (a[2] != b[2] && sameTypeOnly) {
					continue;
				}

				int dx = a[0] - b[0];
				int dy = a[1] - b[1];

				float dist = sqrt(dx * dx + dy * dy);
				if (dist < threshold) {
					overlap[i] = true;
				}
			}
		}

		for (int i = 0; i < len; i++) {
			if (!overlap[i])
				newList.add(predictedTrees.get(i));
		}

		System.out.println("merge from " + predictedTrees.size() + " to " + newList.size());
		predictedTrees = newList;
	}

	private void writeCsv() throws IOException {
		BufferedWriter csvPredict = new BufferedWriter(new FileWriter(predictedCsv));

		for (int i = 0; i < predictedTrees.size(); i++) {
			int[] p = predictedTrees.get(i);

			csvPredict.write(p[0] + "," + p[1] + "," + p[2]);
			csvPredict.newLine();

		}

		csvPredict.close();
	}

	private void renderPrediction() {
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
		int width = cols, height = rows;

		Mat matImg = new Mat(height, width, CvType.CV_8UC3);
		/* set all values to 255 (white color) */
		Core.add(matImg, Scalar.all(255), matImg);

		// for (String csv : treePredictionCsv) {
		// BufferedReader reader = new BufferedReader(new FileReader(csv));
		// String line;
		// while ((line = reader.readLine()) != null) {
		// String[] subStr = line.split(",");
		// if (subStr.length == 3) {

		for (int[] p : predictedTrees) {

			int x = p[0];
			int y = p[1];
			if (x >= 0 && x < width && y >= 0 && y < height) {
				int type = p[2];
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
		Imgcodecs.imwrite(rendering, matImg);
	}

	public void setup() {
		size(200, 200);
		try {
			model.loadModel("data\\models\\", "yolo_as_final_2.pb");
			loadTrainingData();
			evaluate_batch(patchSize / 2, 0.7f, false, true, 128);
			eleminateOverlap(true, 40);
			writeCsv();
			renderPrediction();
			exit();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private PImage getPatchImage(int x, int y, int size) {
		int colStart = x, colEnd = x + size, rowStart = y, rowEnd = y + size;

		if (colStart >= 0 && colEnd <= cols && rowStart >= 0 && rowEnd <= rows) {
			Mat subImg = imageFull.submat(rowStart, rowEnd, colStart, colEnd);
			return CVRenderUtil.toPImage(this, subImg);
		}

		return null;
	}

	private static float[][][] toVector(PImage pImage, boolean normalize, boolean normalize2) {
		pImage.loadPixels();
		float[][][] img = new float[256][256][3];
		for (int i = 0; i < pImage.height; i++) {
			for (int j = 0; j < pImage.width; j++) {
				int color = pImage.pixels[i * 256 + j];
				int r = color & 0x00ff0000;
				r >>= 16;
				int g = color & 0x0000ff00;
				g >>= 8;
				int b = color & 0x000000ff;

				img[i][j][0] = r;// / 255f;
				img[i][j][1] = g;// / 255f;
				img[i][j][2] = b;// / 255f;

				if (normalize) {
					img[i][j][0] /= 255f;
					img[i][j][1] /= 255f;
					img[i][j][2] /= 255f;
				}

				if (normalize2) {
					img[i][j][0] -= 128f;
					img[i][j][1] -= 128f;
					img[i][j][2] -= 128f;

					img[i][j][0] /= 128f;
					img[i][j][1] /= 128f;
					img[i][j][2] /= 128f;
				}
			}
		}

		return img;
	}
}
