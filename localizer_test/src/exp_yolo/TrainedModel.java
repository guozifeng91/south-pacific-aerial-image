package exp_yolo;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

/**
 * this class provide access to the tensorflow model trained with python.
 * <p>
 * 
 * @author guozifeng
 *
 */
public class TrainedModel {
	private byte[] graphDef;
	public int size;
	public int dim;

	public TrainedModel(int size_, int dim_) {
		size = size_;
		dim = dim_;
	}

	public void loadModel(String path, String filename) throws IOException {
		graphDef = Files.readAllBytes((Paths.get(path, filename)));
	}

	public double[][][][] predict(double[][][][] x, int batch_size) {
		return FloatDoubleConvert.toDoubleArray(predict(FloatDoubleConvert.toFloatArray(x), batch_size));
	}

	public float[][][][] predict(float[][][][] x, int batch_size) {
		Tensor x_ = Tensor.create(x);
		float[][][][] predict = new float[batch_size][size][size][dim];

		try (Graph g = new Graph()) {
			g.importGraphDef(graphDef);
			try (Session s = new Session(g)) {
				List<Tensor> result = s.runner().feed("x", x_).fetch("predict:0").run();
				result.get(0).copyTo(predict);
			}
		}

		return predict;
	}

	public double[][] predictFlatten(double[][][][] x, int batch_size) {
		return FloatDoubleConvert.toDoubleArray(predictFlatten(FloatDoubleConvert.toFloatArray(x), batch_size));
	}

	public float[][] predictFlatten(float[][][][] x, int batch_size) {
		float[][][][] predict = predict(x, batch_size);
		float[][] flatten = new float[batch_size][size * size * dim];
		for (int i = 0; i < batch_size; i++) {
			for (int a = 0; a < size; a++) // row (y)
				for (int b = 0; b < size; b++) // col (x)
					for (int c = 0; c < dim; c++) // dim
						flatten[i][(a * size + b) * dim + c] = predict[i][a][b][c];
		}
		return flatten;
	}
}
