package exp_yolo;

import org.opencv.core.Mat;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;

public class CVRenderUtil {
	public static PImage toPImage(PApplet app, Mat img) {
		int width = img.cols();
		int height = img.rows();

		PImage pimg = app.createImage(width, height, PConstants.RGB);
		pimg.loadPixels();

		int channel = img.channels();
		switch (channel) {
		case 1:
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					int index = j * width + i;
					double[] data = img.get(j, i);
					int gray = (int) data[0];
					int color = 0xff000000 | gray;
					gray <<= 8;
					color |= gray;
					gray <<= 8;
					color |= gray;
					pimg.pixels[index] = color;
				}
			}
			break;
		case 3:
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					int index = j * width + i;
					double[] data = img.get(j, i);
					int b = (int) data[0];
					int g = (int) data[1];
					int r = (int) data[2];
					r <<= 16;
					g <<= 8;
					r |= 0xff000000;
					r |= g;
					r |= b;

					pimg.pixels[index] = r;
				}
			}
			break;
		default:
			throw new RuntimeException();
		}

		pimg.updatePixels();
		return pimg;
	}
}
