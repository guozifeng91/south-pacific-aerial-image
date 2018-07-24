package prepare_data;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Envelope;
import com.vividsolutions.jts.geom.Geometry;

import geoJSON2JTS.GeoJSON;

/**
 * preparing training data
 * 
 * @author guozifeng
 *
 */
public class PrepareDataset {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	private static void test() {
		double[] test = new double[200];
		for (int i = 0; i < 200; i++) {
			test[i] = i + 1;
		}
		test = changeRank(test, 5, 8);
		for (int i = 0; i < 200; i++) {
			System.out.println(test[i]);
		}
	}

	public static void main(String[] args) {

		try {
			// cutImg(256, "data\\train\\train.jpg",
			// "data\\sub_images.zip");

			getBoundingBox();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private static void getBoundingBox() throws IOException {
		String root = "data\\train\\";
		String[] source = new String[] {

				root + "Trees\\Banana_trees_musaceae\\bananaTrees.geojson",

				root + "Trees\\Coconut_tree_cocos_nucufera\\coconutsTrees.geojson",

				root + "Trees\\Mango_trees_mangifera_indica\\mangoTrees.geojson",

				root + "Trees\\Papaya_tress_carica_papaya\\papya_trees.geojson" };
		double[] radius = new double[] { 70, 100, 160, 90 };
		String boundary = root + "boundingBox.geojson";

		getBoundingBoxes(17761, 256, 5, radius, source, boundary, "data\\sub_bounding.zip");
	}

	private static void getBoundingBoxes(int width, int size, int gridNum, double[] radius, String[] source, String boundary, String dest) throws IOException {
		/*
		 * for each sub image, the output is a gridNum x gridNum * (4 + 4) vector, which
		 * means sub image is divided into gridNum x gridNum grids, each grid predict a
		 * bounding box (x, y, s) and a confidence value c and a one-hot vector which
		 * indicates the category of the bounding box
		 */

		ZipOutputStream zip = new ZipOutputStream(new FileOutputStream(dest));

		ArrayList<ArrayList<Geometry>> trees = new ArrayList<ArrayList<Geometry>>();
		for (String s : source) {
			trees.add(GeoJSON.toGeometries(s));
		}
		// ArrayList<Geometry> geos = GeoJSON.toGeometries(source);
		Envelope envelope = GeoJSON.toGeometry(boundary).getEnvelopeInternal();

		double scale = width / envelope.getWidth();
		int height = (int) Math.round(envelope.getHeight() * scale);

		System.out.println(width + ", " + height);

		/* size of gird = size / gridNum */
		double sizeGrid = (double) size / gridNum;

		/* below use image coordinate system */
		for (int x = 0; x + size < width; x += size / 2) {
			for (int y = 0; y + size < height; y += size / 2) {
				/* start index = (girdY * girdNum + gridX) * 5 */
				double[] boundingBox = new double[gridNum * gridNum * (4 + 4)];

				/* for each sub image */
				for (int i = 0; i < trees.size(); i++) {
					ArrayList<Geometry> geos = trees.get(i);
					for (Geometry g : geos) {
						/* cross check each point */
						Coordinate coord = g.getCoordinate();
						/* relative coordinate of the point in current sub image */
						double coordX = scale * (coord.x - envelope.getMinX()) - x;
						double coordY = scale * (envelope.getMaxY() - coord.y) - y;

						if (coordX >= 0 && coordX < size && coordY >= 0 && coordY < size) {
							// System.out.println(coordX + ", " + coordY + " | " + size + " " + sizeGrid);

							/* point locates in current sub image */
							int gridX = (int) (coordX / sizeGrid);
							int gridY = (int) (coordY / sizeGrid);

							// System.out.println(gridX + " - " + gridY);

							double cornerX = gridX * sizeGrid;
							double cornerY = gridY * sizeGrid;

							/* relative of the point in corresponding gird */
							coordX -= cornerX;
							coordY -= cornerY;

							coordX /= sizeGrid;
							coordY /= sizeGrid;

							int index = (gridY * gridNum + gridX) * (4 + 4);

							boundingBox[index] = coordX; // x
							boundingBox[index + 1] = coordY; // y
							boundingBox[index + 2] = radius[i] / size; // s
							boundingBox[index + 3] = 1; // confidence
							boundingBox[index + 4 + i] = 1; // category
						}
					}
				}

				// this code is for training in mathematica (where the ranks are channel, y, x)
				// by default the ranks are y, x, channel
//				boundingBox = changeRank(boundingBox, gridNum, 8);

				String data = "";
				for (int i = 0; i < boundingBox.length; i++) {
					data += String.valueOf(boundingBox[i]);
					if (i < boundingBox.length - 1)
						data += ",";
					if (boundingBox[i] < 0 || boundingBox[i] > 1)
						System.out.println("Warning");
				}

				String name = (x * 2 / size) + "_" + (y * 2 / size) + ".csv";
				System.out.println(name);
				ZipEntry entry = new ZipEntry(name);
				zip.putNextEntry(entry);
				zip.write(data.getBytes());
				zip.closeEntry();
			}
		}

		zip.close();
	}

	private static double[] changeRank(double[] boundingBox, int size, int channelNum) {
		size = size * size;
		double[] boundingBoxTrans = new double[boundingBox.length];
		for (int i = 0; i < boundingBoxTrans.length; i++) {
			int index = i % size;
			int channel = i / size;
			boundingBoxTrans[i] = boundingBox[index * channelNum + channel];
		}

		return boundingBoxTrans;
	}

	/**
	 * cut source image into size x size sub images with stride of size/2 x size/2
	 * <p>
	 * results are stored in zip file dest
	 * 
	 * @param size
	 * @param source
	 * @param dest
	 * @throws IOException
	 */
	private static void cutImg(int size, String source, String dest) throws IOException {
		ZipOutputStream zip = new ZipOutputStream(new FileOutputStream(dest));

		Mat img = Imgcodecs.imread(source, Imgcodecs.IMREAD_COLOR);
		int cols = img.cols();
		int rows = img.rows();

		int count = 0;

		for (int x = 0; x + size < cols; x += size / 2) {
			for (int y = 0; y + size < rows; y += size / 2) {
				Mat subImg = img.submat(y, y + size, x, x + size);
				MatOfByte imgBytes = new MatOfByte();
				Imgcodecs.imencode(".jpg", subImg, imgBytes);

				String name = (x * 2 / size) + "_" + (y * 2 / size) + ".jpg";
				System.out.println(name);

				ZipEntry entry = new ZipEntry(name);
				zip.putNextEntry(entry);
				zip.write(imgBytes.toArray());
				zip.closeEntry();

				/*
				 * for debug and test in a small subset, remove to get complete data set
				 * (approx. 650MB)
				 */
				count++;
				// if (count >= 400)
				// break;
			}
			// if (count >= 400)
			// break;
		}

		zip.close();
	}
}
