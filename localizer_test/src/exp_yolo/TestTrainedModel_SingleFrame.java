package exp_yolo;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import processing.core.PApplet;
import processing.core.PImage;

public class TestTrainedModel_SingleFrame extends PApplet {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	String name = "1_70"; // which file is used to test the model

	String zipImgFile = "data\\sub_images.zip"; // the zip file, contains all the training patches

	ArrayList<PImage> images = new ArrayList<PImage>();
	ArrayList<float[][][]> vectors = new ArrayList<float[][][]>();

	TrainedModel model = new TrainedModel(5, 8);

	int size = 256;
	int gridNum = 5;

	float gridSize = (float) size / gridNum;

	int[] colorByType = new int[] { 0xffff0000, 0xff00ff00, 0xff0000ff, 0xffffff00 };

	public void setup() {
		size(400, 400);
		try {
			model.loadModel("data\\models\\", "yolo_as_final_2.pb");
			loadData();
		} catch (IOException e) {
			e.printStackTrace();
		}

		frameRate(10);

		rectMode(CENTER);
		textSize(30);
	}

	public void draw() {
		background(255);

		fill(255, 0, 0);
		text("data set validation:", 10, 30);
		translate((width - 256) / 2, (height - 256) / 2);

		int index = mouseX * images.size() / width;
		image(images.get(index), 0, 0);

		strokeWeight(3);
		noFill();
		float[][][] imgVec = vectors.get(index);

		float[] vector = model.predictFlatten(new float[][][][] { imgVec }, 1)[0];
		print("[");
		for (float v : vector)
			print(v, ", ");
		println("]");

		for (int i = 0; i < gridNum; i++) {
			for (int j = 0; j < gridNum; j++) {
				int indexGrid = (j * gridNum + i) * (4 + 4);
				if (vector[indexGrid + 3] > 0.8) {
					/* confidence value is large */
					float x = (float) vector[indexGrid];
					float y = (float) vector[indexGrid + 1];
					float s = (float) vector[indexGrid + 2];

					x *= gridSize;
					y *= gridSize;

					x += i * gridSize;
					y += j * gridSize;

					s *= size;

					/* category of tree */
					int type = 0;
					for (int k = 1; k < 4; k++) {
						if (vector[indexGrid + 4 + k] > vector[indexGrid + 4 + type]) {
							type = k;
						}
					}
					stroke(colorByType[type]);
					rect(x, y, s, s);
				}
			}
		}
	}

	private void loadData() throws IOException {
		ZipFile zipImg = new ZipFile(zipImgFile);
		Enumeration<? extends ZipEntry> entryImg = zipImg.entries();

		int maxSize = 2 * 1024 * 1024;
		byte[] buffer;

		while (entryImg.hasMoreElements()) {
			ZipEntry entry = entryImg.nextElement();
			if (entry.getName().equals(name + ".jpg")) {
				/* read image */
				InputStream input = zipImg.getInputStream(entry);
				buffer = read(input, maxSize);
				MatOfByte matOfByte = new MatOfByte();
				matOfByte.fromArray(buffer);
				Mat img = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_COLOR);
				images.add(CVRenderUtil.toPImage(this, img));
				vectors.add(toVector(images.get(images.size() - 1), false, true));
			}
		}

		zipImg.close();
	}

	static float[][][] toVector(PImage pImage, boolean normalize, boolean normalize2) {
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

					if (img[i][j][0] < 0) img[i][j][0] += 255;
					if (img[i][j][1] < 0) img[i][j][1] += 255;
					if (img[i][j][2] < 0) img[i][j][2] += 255;
					
					img[i][j][0] /= 128f;
					img[i][j][1] /= 128f;
					img[i][j][2] /= 128f;
				}
			}
		}

		System.out.println(img[0][0][0]);
		System.out.println(img[0][0][1]);
		System.out.println(img[0][0][2]);
		return img;
	}

	static byte[] read(InputStream source, int maxSize) throws IOException {
		int capacity = maxSize;
		byte[] buf = new byte[capacity];
		int nread = 0;
		int n;

		// read until all capacity is used or end of file detected.
		while ((n = source.read(buf, nread, capacity - nread)) > 0)
			nread += n;

		return (capacity == nread) ? buf : Arrays.copyOf(buf, nread);
	}
}
