package prepare_data;

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

public class ValidateDataset extends PApplet {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	String zipImgFile = "data\\sub_images.zip";
	String zipCSVFile = "data\\sub_bounding_h_w_c.zip";

	ArrayList<PImage> images = new ArrayList<PImage>();
	ArrayList<double[]> vectors = new ArrayList<double[]>();

	int size = 256;
	int gridNum = 5;

	float gridSize = (float) size / gridNum;

	int[] colorByType = new int[] { 0xffff0000, 0xff00ff00, 0xff0000ff, 0xffffff00 };

	public void setup() {
		size(400, 400);
		try {
			loadData();
		} catch (IOException e) {
			e.printStackTrace();
		}

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
		double[] vector = vectors.get(index);
		for (int i = 0; i < gridNum; i++) {
			for (int j = 0; j < gridNum; j++) {
				int indexGrid = (j * gridNum + i) * (4 + 4);
				if (vector[indexGrid + 3] > 0.5) {
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
		ZipFile zipCSV = new ZipFile(zipCSVFile);

		Enumeration<? extends ZipEntry> entryImg = zipImg.entries();
		Enumeration<? extends ZipEntry> entryCSV = zipCSV.entries();

		int maxSize = 2 * 1024 * 1024;
		byte[] buffer;

		while (entryImg.hasMoreElements()) {
			/* read image */
			InputStream input = zipImg.getInputStream(entryImg.nextElement());
			buffer = read(input, maxSize);
			MatOfByte matOfByte = new MatOfByte();
			matOfByte.fromArray(buffer);
			Mat img = Imgcodecs.imdecode(matOfByte, Imgcodecs.IMREAD_COLOR);
			images.add(CVRenderUtil.toPImage(this, img));

			/* read csv */
			input = zipCSV.getInputStream(entryCSV.nextElement());
			buffer = read(input, maxSize);
			String csv = new String(buffer);
			vectors.add(toDoubleArray(csv));
		}

		zipImg.close();
		zipCSV.close();
	}

	static double[] toDoubleArray(String csv) {
		String[] sub = csv.split(",");
		double[] vec = new double[sub.length];
		for (int i = 0; i < sub.length; i++)
			vec[i] = Double.valueOf(sub[i]);
		return vec;
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
