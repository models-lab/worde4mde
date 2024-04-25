package ml2.semantic.emfcompare;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.eclipse.emf.compare.Comparison;
import org.eclipse.emf.compare.match.DefaultComparisonFactory;
import org.eclipse.emf.compare.match.DefaultEqualityHelperFactory;
import org.eclipse.emf.compare.match.DefaultMatchEngine;
import org.eclipse.emf.compare.match.IComparisonFactory;
import org.eclipse.emf.compare.match.IMatchEngine;
import org.eclipse.emf.compare.match.eobject.CachingDistance;
import org.eclipse.emf.compare.match.eobject.EditionDistance;
import org.eclipse.emf.compare.match.eobject.IEObjectMatcher;
import org.eclipse.emf.compare.match.eobject.ProximityEObjectMatcher;
import org.eclipse.emf.compare.match.eobject.ProximityEObjectMatcher.DistanceFunction;
import org.eclipse.emf.compare.match.impl.MatchEngineFactoryImpl;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EStructuralFeature;

import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import mar.indexer.embeddings.Embeddable;
import mar.indexer.embeddings.EmbeddingStrategy;

public class MatchEngineFactory extends MatchEngineFactoryImpl {
	
	public MatchEngineFactory() {
		System.out.println("Create match engine factory");
	}
	
	public IMatchEngine getMatchEngine() {
		if (matchEngine == null) {
			final IComparisonFactory comparisonFactory = new DefaultComparisonFactory(
					new DefaultEqualityHelperFactory());
			final DistanceFunction df = new EmbeddingDistanceFunction();
			final IEObjectMatcher matcher = new ProximityEObjectMatcher(new CachingDistance(df));
			//DefaultMatchEngine.createDefaultEObjectMatcher(
			//		shouldUseIdentifiers, weightProviderRegistry, equalityHelperExtensionProviderRegistry);
			matchEngine = new DefaultMatchEngine(matcher, comparisonFactory);
		

			System.out.println("Create Match Engine");
		}
		return matchEngine;
	}
	
	public static class EmbeddingDistanceFunction extends EditionDistance {
		private EmbeddingStrategy strategy;

		public EmbeddingDistanceFunction() {
			try {
				this.strategy = new EmbeddingStrategy.GloveWordE(new File("/home/jesus/projects/mde-ml/word2vec-mde/vectors/glove_modelling/vectors.txt"));
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		@Override
		public double distance(Comparison inProgress, EObject a, EObject b) {
			float[] v1 = getVector(a);
			float[] v2 = getVector(b);
			return 1 - VectorSimilarityFunction.DOT_PRODUCT.compare(v1, v2);
		}

		@Override
		public boolean areIdentic(Comparison inProgress, EObject a, EObject b) {
			return super.areIdentic(inProgress, a, b);
		}
		
		
		@SuppressWarnings("unchecked")
		public float[] getVector(EObject o) {
			return this.strategy.toNormalizedVector(new EmbeddedEObject(o));
		}
	}
	
	private static class EmbeddedEObject implements Embeddable {

		private List<String> strings;

		public EmbeddedEObject(EObject o) {
			this.strings = new ArrayList<>();			
			for (EStructuralFeature f : o.eClass().getEAllStructuralFeatures()) {
				Object v = o.eGet(f);
				if (v != null && v instanceof Collection) {
					for (Object obj : (Collection<Object>) v) {
						if (obj instanceof String) {
							strings.add((String) obj);
						}
					}
				} else if (v != null) {
					if (v instanceof String) {
						strings.add((String) v);
					}
				} else {
					// We do nothing
				}
			}
		
		}
		
		
		@Override
		public int getSeqId() {
			throw new UnsupportedOperationException();
		}

		@Override
		public List<? extends String> getWords() {
			return strings;
		}
		
	}
	
}
