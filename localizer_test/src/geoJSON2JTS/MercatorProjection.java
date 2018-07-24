package geoJSON2JTS;

/**
 * The Mercator Projection
 * 
 * @author guoguo
 *
 */
public class MercatorProjection {
	/**
	 * radius of the earth
	 */
	public static final double R = 6378137;

	/**
	 * get the x coordinate of a point
	 * 
	 * @param longitude
	 *            -180 (west) to 180 (east)
	 * @return
	 */
	public static double getX(double longitude) {
		/*
		 * Assume the longitude is given within the range of [-180, 180], R is
		 * the earth radius of 6378137, then
		 * 
		 * x = Rad(longitude) * R = longitude / 180 * PI * R = longitude * a;
		 * 
		 * where a = PI * R / 180 = 111319.49079327357264771338267056
		 */
		if (longitude < -180 || longitude > 180)
			throw new IllegalArgumentException("Longitude value exceeds the range of [-180, 180]: " + longitude);

		return longitude * 111319.49079327357264771338267056;
	}

	/**
	 * get the y coordinate of a point
	 * 
	 * @param latitude
	 *            -85.05113(south) to 85.05113 (north)
	 * @return
	 */
	public static double getY(double latitude) {
		/*
		 * Assume the latitude is given within the range of [-85.05113,
		 * 85.05113], R is the earth radius of 6378137, then
		 * 
		 * y = R * ln(tan(PI / 4 + Rad(latitude) / 2))
		 */
		if (latitude < -85.05113 || latitude > 85.05113)
			throw new IllegalArgumentException("Latitude value exceeds the range of [-85.05113, 85.05113]: " + latitude);

		return R * Math.log(Math.tan(0.78539816339744830961566084581988 + 0.00872664625997164788461845384244 * latitude));
	}
}
