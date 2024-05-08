package ml2.worde4mde;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;

public class VectorUtils {

	public static float dotProduct(float[] v1, float[] v2) {
		return VectorSimilarityFunction.DOT_PRODUCT.compare(v1, v2);
	}

	public static float cosine(float[] v1, float[] v2) {		
		return VectorSimilarityFunction.COSINE.compare(v1, v2);
	}
	
}
