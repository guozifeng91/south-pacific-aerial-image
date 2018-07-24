package exp_yolo;

public class FloatDoubleConvert {
	public static float[][][][] toFloatArray(double[][][][] array) {
		float[][][][] arrayf = new float[array.length][][][];
		for (int i = 0; i < array.length; i++) {
			arrayf[i] = toFloatArray(array[i]);
		}
		return arrayf;
	}

	public static float[][][] toFloatArray(double[][][] array) {
		float[][][] arrayf = new float[array.length][][];
		for (int i = 0; i < array.length; i++) {
			arrayf[i] = toFloatArray(array[i]);
		}
		return arrayf;
	}

	public static float[][] toFloatArray(double[][] array) {
		float[][] arrayf = new float[array.length][];
		for (int i = 0; i < array.length; i++) {
			arrayf[i] = toFloatArray(array[i]);
		}
		return arrayf;
	}

	public static float[] toFloatArray(double[] array) {
		float[] arrayf = new float[array.length];
		for (int i = 0; i < array.length; i++) {
			arrayf[i] = (float) array[i];
		}
		return arrayf;
	}

	public static double[][][][] toDoubleArray(float[][][][] array) {
		double[][][][] arrayd = new double[array.length][][][];
		for (int i = 0; i < array.length; i++) {
			arrayd[i] = toDoubleArray(array[i]);
		}
		return arrayd;
	}

	public static double[][][] toDoubleArray(float[][][] array) {
		double[][][] arrayd = new double[array.length][][];
		for (int i = 0; i < array.length; i++) {
			arrayd[i] = toDoubleArray(array[i]);
		}
		return arrayd;
	}

	public static double[][] toDoubleArray(float[][] array) {
		double[][] arrayd = new double[array.length][];
		for (int i = 0; i < array.length; i++) {
			arrayd[i] = toDoubleArray(array[i]);
		}
		return arrayd;
	}

	public static double[] toDoubleArray(float[] array) {
		double[] arrayd = new double[array.length];
		for (int i = 0; i < array.length; i++) {
			arrayd[i] = array[i];
		}
		return arrayd;
	}
}
