package geoJSON2JTS;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;

import com.vividsolutions.jts.geom.Coordinate;
import com.vividsolutions.jts.geom.Geometry;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.LinearRing;

import processing.data.JSONArray;
import processing.data.JSONObject;

public class GeoJSON {
	static final GeometryFactory gf = new GeometryFactory();
	public static double SCALE = 1;
	public static boolean USE_WGS84 = true;

	public static Geometry toGeometry(JSONObject geo) {
		String type = geo.getString("type");
		JSONArray coord = geo.getJSONArray("coordinates");
		if (type.equals("Point")) {
			return toPoint(coord);
		} else if (type.equals("LineString")) {
			return toLineString(coord);
		} else if (type.equals("Polygon")) {
			return toPolygon(coord);
		} else if (type.equals("MultiPolygon")) {
			Geometry multiPolygon = null;
			for (int j = 0; j < coord.size(); j++) {
				if (multiPolygon == null)
					multiPolygon = toPolygon(coord.getJSONArray(j));
				else
					multiPolygon = multiPolygon.symDifference(toPolygon(coord.getJSONArray(j)));
			}
			return multiPolygon;
		}
		return null;
	}

	public static Geometry toGeometry(String filename) throws FileNotFoundException {
		JSONObject json = new JSONObject(new BufferedReader(new FileReader(filename)));
		JSONArray features = null;
		try {
			features = json.getJSONArray("features");
		} catch (RuntimeException e) {
		}

		if (features != null) {
			return toGeometry(features.getJSONObject(0).getJSONObject("geometry"));
		} else {
			return toGeometry(json);
		}
	}

	public static ArrayList<Geometry> toGeometries(String filename) throws FileNotFoundException {
		ArrayList<Geometry> geometries = new ArrayList<Geometry>();
		JSONObject json = new JSONObject(new BufferedReader(new FileReader(filename)));
		JSONArray features = json.getJSONArray("features");

		for (int i = 0; i < features.size(); i++) {
			JSONObject obj = features.getJSONObject(i);
			JSONObject geo = obj.getJSONObject("geometry");
			String type = geo.getString("type");
			JSONArray coord = geo.getJSONArray("coordinates");
			if (type.equals("Point")) {
				geometries.add(toPoint(coord));
			} else if (type.equals("LineString")) {
				geometries.add(toLineString(coord));
			} else if (type.equals("Polygon")) {
				geometries.add(toPolygon(coord));
			} else if (type.equals("MultiPolygon")) {
				for (int j = 0; j < coord.size(); j++) {
					geometries.add(toPolygon(coord.getJSONArray(j)));
				}
			}

			// System.out.println(geometries.get(geometries.size() - 1));
		}

		return geometries;
	}

	public static Geometry toPoint(JSONArray coord) {
		return gf.createPoint(toCoordinate(coord));
	}

	public static Geometry toLineString(JSONArray coord) {
		return gf.createLineString(toCoordArray(coord));
	}

	public static Geometry toPolygon(JSONArray coord) {
		Coordinate[][] coords = toCoordArray2(coord);
		if (coords.length == 1) {
			return gf.createPolygon(coords[0]);
		}

		LinearRing outer = gf.createLinearRing(coords[0]);
		LinearRing[] inners = new LinearRing[coords.length - 1];
		for (int i = 0; i < inners.length; i++) {
			inners[i] = gf.createLinearRing(coords[i + 1]);
		}
		return gf.createPolygon(outer, inners);
	}

	public static Coordinate toCoordinate(JSONArray coord) {
		if (USE_WGS84)
			return new Coordinate(MercatorProjection.getX(coord.getDouble(0)) * SCALE, MercatorProjection.getY(coord.getDouble(1)) * SCALE);
		else
			return new Coordinate(coord.getDouble(0) * SCALE, coord.getDouble(1) * SCALE);
	}

	public static Coordinate[] toCoordArray(JSONArray coord) {
		int len = coord.size();
		Coordinate[] coords = new Coordinate[len];
		for (int i = 0; i < len; i++) {
			JSONArray arr = coord.getJSONArray(i);
			coords[i] = toCoordinate(arr);
		}
		return coords;
	}

	public static Coordinate[][] toCoordArray2(JSONArray coord) {
		int len = coord.size();
		Coordinate[][] coords = new Coordinate[len][];
		for (int i = 0; i < len; i++) {
			JSONArray arr = coord.getJSONArray(i);
			coords[i] = toCoordArray(arr);
		}
		return coords;
	}
}
