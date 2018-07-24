package exp_yolo;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import processing.core.PApplet;
import processing.core.PImage;
import processing.core.PVector;

/**
 * test the trained model
 */
public class TestTrainedModel_AnyPosition_ForVideo extends PApplet {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	private String file = "data\\validation\\DSC09408_geotag.jpg"; // source image

	private int patchSizePixel = 256; // size of the image patch, currently always 256
	private int gridNum = 5;

	private TrainedModel model = new TrainedModel(gridNum, 8);

	private float gridSize = (float) patchSizePixel / gridNum;
	private int[] colorByType = new int[] { 0xffee0000, 0xff00ee00, 0xff0000ee, 0xffeeee00 };

	private double threshold = 0.5; // threshold of the confidence value

	private PVector smoothPos = new PVector(0, 0);

	private Mat imageFull; // the original satellite image
	private int cols, rows; // size of the satellite image
	private Mat imageDisplay; // the original size image within the display area;
	private PImage buf; // the scaled satellite image, for display

	/** the area for display the satellite image */
	private float[] displayArea = new float[] { 10, 10, 1500, 1060 };
	/** the scale for display the satellite image */
	private float displayScale = 0.1f;
	/**
	 * limit of display scale, to make sure the image always fills the display area
	 */
	private float displayScaleLimit;
	/** position of the displayed image, in pixels */
	private int[] imageTranslatePixel = new int[] { 0, 0 };
	/** mouse position, in pixels */
	private float[] samplingPosDisplay = new float[] { 0, 0 };
	/** the size of the patch, for display */
	private float patchSizeDisplay;

	private boolean recording = false;

	private int recordCount = 0;

	public void setup() {
		/* use openGL for display */
		size(1920, 1080, P3D);

		textAlign(CENTER, CENTER);
		textSize(20);

		try {
			model.loadModel("data\\models\\", "yolo_as_final_2.pb"); // load the trained model
		} catch (IOException e) {
			e.printStackTrace();
		}

		loadSatelliteImg();
		/* apply the display scale */
		changeDisplayScale_center(displayScale);
	}

	/** load satellite image with openCV */
	private void loadSatelliteImg() {
		imageFull = Imgcodecs.imread(file);
		cols = imageFull.cols();
		rows = imageFull.rows();

		float s1 = displayArea[2] / cols;
		float s2 = displayArea[3] / rows;

		/* multiply a small number to prevent floating error */
		displayScaleLimit = max(s1, s2) * 1.001f;
	}

	/**
	 * update the buffer image for display, check the size before calling this
	 * method
	 */
	private void updateBufImpl() {
		int display_w = (int) displayArea[2];
		int display_h = (int) displayArea[3];

		int translate_real_x = imageTranslatePixel[0];
		int translate_real_y = imageTranslatePixel[1];

		int display_real_w = (int) (displayArea[2] / displayScale);
		int display_real_h = (int) (displayArea[3] / displayScale);

		imageDisplay = imageFull.submat(translate_real_y, translate_real_y + display_real_h, translate_real_x, translate_real_x + display_real_w);
		Mat resized = new Mat(display_h, display_w, imageDisplay.type());
		Imgproc.resize(imageDisplay, resized, new Size(display_w, display_h));
		buf = CVRenderUtil.toPImage(this, resized);
	}

	/** update display buffer image */
	private void updateDisplayBuffer(boolean override) {
		// the size of the whole image for display
		int scaled_w = (int) (cols * displayScale);
		int scaled_h = (int) (rows * displayScale);

		patchSizeDisplay = patchSizePixel * displayScale;
		float patchDisplayHalf = patchSizeDisplay / 2;

		/* image translate, in display(screen) scale */
		float txDisplay = imageTranslatePixel[0] * displayScale;
		float tyDisplay = imageTranslatePixel[1] * displayScale;

		/* changed translate value, in display(screen) scale */
		float tx = txDisplay;
		float ty = tyDisplay;

		/* if the sampling area too close to the boundary? */
		float dx = 0;
		if ((dx = samplingPosDisplay[0] - displayArea[0]) < patchDisplayHalf) {
			samplingPosDisplay[0] = displayArea[0] + patchDisplayHalf;
			tx -= (patchDisplayHalf - dx);
		} else if ((dx = displayArea[0] + displayArea[2] - samplingPosDisplay[0]) < patchDisplayHalf) {
			samplingPosDisplay[0] = displayArea[0] + displayArea[2] - patchDisplayHalf;
			tx += (patchDisplayHalf - dx);
		}

		float dy = 0;
		if ((dy = samplingPosDisplay[1] - displayArea[1]) < patchDisplayHalf) {
			samplingPosDisplay[1] = displayArea[1] + patchDisplayHalf;
			ty -= (patchDisplayHalf - dy);
		} else if ((dy = displayArea[1] + displayArea[3] - samplingPosDisplay[1]) < patchDisplayHalf) {
			samplingPosDisplay[1] = displayArea[1] + displayArea[3] - patchDisplayHalf;
			ty += (patchDisplayHalf - dy);
		}

		/* if yes, check if the background image can be moved */
		if (tx < 0)
			tx = 0;
		if (tx + displayArea[2] > scaled_w) {
			tx = scaled_w - displayArea[2];
		}

		if (ty < 0)
			ty = 0;
		if (ty + displayArea[3] > scaled_h) {
			ty = scaled_h - displayArea[3];
		}

		/* if yes, update the background image */
		if (tx != txDisplay || ty != tyDisplay || override) {
			/* apply the new translation in real(pixel) scale */
			imageTranslatePixel[0] = (int) (tx / displayScale);
			imageTranslatePixel[1] = (int) (ty / displayScale);
			updateBufImpl();
		}
	}

	/**
	 * update sampling position
	 * 
	 * @param x
	 *            x coordinate, pixel
	 * @param y
	 *            y coordinate, pixel
	 */
	private void updateSamplingPosition(float x, float y) {
		/* padding area that respond the mouse event around the display area */
		float padding = 100;
		if (x >= displayArea[0] - padding && x <= displayArea[0] + displayArea[2] + padding && y >= displayArea[1] - padding && y <= displayArea[1] + displayArea[3] + padding) {
			samplingPosDisplay[0] = x;
			samplingPosDisplay[1] = y;
		}
	}

	private void changeDisplayScale_center(float newScale) {
		/* check if it exceed the limitation */
		if (newScale < displayScaleLimit)
			newScale = displayScaleLimit;

		/* adjust translate to make it more natural */
		int old_w = (int) (displayArea[2] / displayScale);
		int old_h = (int) (displayArea[3] / displayScale);

		int new_w = (int) (displayArea[2] / newScale);
		int new_h = (int) (displayArea[3] / newScale);

		/* use the display center as the zooming center */
		int cx = old_w / 2 + imageTranslatePixel[0];
		int cy = old_h / 2 + imageTranslatePixel[1];

		int new_tx = cx - new_w / 2;
		int new_ty = cy - new_h / 2;

		if (new_tx < 0)
			new_tx = 0;
		if (new_ty < 0)
			new_ty = 0;

		if (new_tx + new_w > cols)
			new_tx = cols - new_w;
		if (new_ty + new_h > rows)
			new_ty = rows - new_h;

		imageTranslatePixel[0] = new_tx;
		imageTranslatePixel[1] = new_ty;

		/* apply new display scale */
		displayScale = newScale;
		updateDisplayBuffer(true);
	}

	/** change display scale */
	private void changeDisplayScale_sampling(float newScale) {
		/* check if it exceed the limitation */
		if (newScale < displayScaleLimit)
			newScale = displayScaleLimit;

		float u = (samplingPosDisplay[0] - displayArea[0]) / displayArea[2];
		float v = (samplingPosDisplay[1] - displayArea[1]) / displayArea[3];

		/* adjust translate to make it more natural */
		int old_w = (int) (displayArea[2] / displayScale);
		int old_h = (int) (displayArea[3] / displayScale);

		int new_w = (int) (displayArea[2] / newScale);
		int new_h = (int) (displayArea[3] / newScale);

		/* use sampling point as the zooming center */
		int cx = floor(old_w * u) + imageTranslatePixel[0];
		int cy = floor(old_h * v) + imageTranslatePixel[1];

		int new_tx = cx - floor(new_w * u);
		int new_ty = cy - floor(new_h * v);

		if (new_tx < 0)
			new_tx = 0;
		if (new_ty < 0)
			new_ty = 0;

		if (new_tx + new_w > cols)
			new_tx = cols - new_w;
		if (new_ty + new_h > rows)
			new_ty = rows - new_h;

		imageTranslatePixel[0] = new_tx;
		imageTranslatePixel[1] = new_ty;

		/* apply new display scale */
		displayScale = newScale;
		updateDisplayBuffer(true);
	}

	private void drawSamplingArea() {
		pushStyle();
		fill(255,100);
		stroke(255);
		strokeWeight(4);
		rect(samplingPosDisplay[0] - patchSizeDisplay / 2, samplingPosDisplay[1] - patchSizeDisplay / 2, patchSizeDisplay, patchSizeDisplay);
		popStyle();
	}

	private PImage getSamplingData() {
		int colStart = (int) ((samplingPosDisplay[0] - displayArea[0]) / displayScale);
		int rowStart = (int) ((samplingPosDisplay[1] - displayArea[1]) / displayScale);

		colStart -= ceil(patchSizePixel / 2f);
		rowStart -= ceil(patchSizePixel / 2f);

		if (colStart < 0)
			colStart = 0;
		if (rowStart < 0)
			rowStart = 0;

		Mat subImg = imageDisplay.submat(rowStart, rowStart + patchSizePixel, colStart, colStart + patchSizePixel);
		return CVRenderUtil.toPImage(this, subImg);
	}

	private void showPrediction(PImage img, float display_x, float display_y, float display_size) {
		pushStyle();
		strokeWeight(4);

		image(img, display_x, display_y, display_size, display_size);
		float scale = display_size / patchSizePixel;

		float[][][] vector = model.predict(new float[][][][] { toVector(img, false, true) }, 1)[0];

		for (int i = 0; i < gridNum; i++) {
			for (int j = 0; j < gridNum; j++) {
				float confidence = vector[j][i][3];// vector[indexGrid + 3];
				if (confidence > threshold) {
					/* confidence value is large */
					float x = vector[j][i][0];// (float) vector[indexGrid];
					float y = vector[j][i][1];// (float) vector[indexGrid + 1];
					float s = vector[j][i][2];// (float) vector[indexGrid + 2];

					x *= gridSize;
					y *= gridSize;

					x += i * gridSize;
					y += j * gridSize;

					s *= patchSizePixel;

					/* scale up if the display size not equals to the patch size */
					x *= scale;
					y *= scale;
					s *= scale;

					/* category of tree */
					int type = 0;
					for (int k = 1; k < 4; k++) {
						if (vector[j][i][4 + k] > vector[j][i][4 + type]) {
							type = k;
						}
					}

					confidence = (int) (confidence * 1000) / 1000f;
					// draw boxes
					stroke(colorByType[type]);
					pushMatrix();
					translate(display_x, display_y);
					noFill();
					rect(x - s / 2, y - s / 2, s, s);
					fill(255, 0, 0);
					text(String.valueOf(confidence), x, y);
					popMatrix();
				}
			}
		}
		popStyle();
	}

	public void draw() {
		background(255);

		if (mousePressed && mouseButton == LEFT) {
			smoothPos.set(mouseX, mouseY);
		} else {
			PVector toMouse = new PVector(mouseX, mouseY);
			toMouse.sub(smoothPos);

			float mag = toMouse.mag();
			if (mag > 1) {
				toMouse.div(mag);
				mag = sqrt(mag);
				toMouse.mult(mag);
			}

			smoothPos.add(toMouse);
		}

		updateSamplingPosition(smoothPos.x, smoothPos.y);
		updateDisplayBuffer(false);

		PImage img = getSamplingData();
		float leftWidth = width - displayArea[2] - displayArea[0] * 3;
		showPrediction(img, displayArea[0] * 2 + displayArea[2], displayArea[1], leftWidth);

		image(buf, displayArea[0], displayArea[1]);
		drawSamplingArea();

		// screen recording, for video making
		if (recording) {
			save("data\\" + recordCount + ".png");
			recordCount++;
		}
	}

	public void keyPressed() {
		if (keyCode == '=') {
			changeDisplayScale_sampling(displayScale * 1.05f);
		} else if (keyCode == '-') {
			changeDisplayScale_sampling(displayScale / 1.05f);
		}
	}

	public void mousePressed() {
		if (mouseButton == RIGHT)
			recording = !recording;
		if (recording)
			System.out.println("recording");
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

					img[i][j][0] /= 128f;
					img[i][j][1] /= 128f;
					img[i][j][2] /= 128f;
				}
			}
		}

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
