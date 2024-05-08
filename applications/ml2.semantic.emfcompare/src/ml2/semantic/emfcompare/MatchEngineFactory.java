package ml2.semantic.emfcompare;

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

import ml2.worde4mde.BagOfWords;
import ml2.worde4mde.Embedding;
import ml2.worde4mde.EmbeddingLoader;
import ml2.worde4mde.EmbeddingLoader.Corpus;
import ml2.worde4mde.EmbeddingLoader.EmbeddingModel;
import ml2.worde4mde.VectorUtils;

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
		private Embedding strategy;

		public EmbeddingDistanceFunction() {
			try {
				this.strategy = EmbeddingLoader.INSTANCE.load(EmbeddingModel.GLOVE, Corpus.MDE, 300);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		@Override
		public double distance(Comparison inProgress, EObject a, EObject b) {
			System.out.println("Comparing " + a + " " + b);
			float[] v1 = getVector(a);			
			float[] v2 = getVector(b);
			if (v1 == null || v2 == null) {
				System.out.println("v1: " + v1 + " - v2: " + v2);
				System.out.println(v1);
				System.out.println(v2);
				return Double.MAX_VALUE;
			}
				// return super.distance(inProgress, a, b);
			
			float v = 1 - VectorUtils.cosine(v1, v2);
			
			//this.uriDistance.setComparison(inProgress);
//			double maxDist = Math.max(getThresholdAmount(a), getThresholdAmount(b));
//			double measuredDist = new CountingDiffEngine(maxDist, this.fakeComparison)
//					.measureDifferences(inProgress, a, b);

			System.out.println("Distance: " + v);
			if (v > 0.5) {
				System.out.println("Max!");
				return Double.MAX_VALUE;
			}
			
			return v;

		}

		@Override
		public boolean areIdentic(Comparison inProgress, EObject a, EObject b) {
			return super.areIdentic(inProgress, a, b);
		}
		
		public float[] getVector(EObject o) {
			//return this.strategy.toNormalizedVector(new EmbeddedEObject(o));
			return this.strategy.toVectorOrNull(new EmbeddedEObject(o));
		}
	}

	public static class EmbeddingDistanceFunctionBagOfWords extends EditionDistance {
		private Embedding strategy;

		public EmbeddingDistanceFunctionBagOfWords() {
			try {
				this.strategy = EmbeddingLoader.INSTANCE.load(EmbeddingModel.GLOVE, Corpus.MDE, 300);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}

		@Override
		public double distance(Comparison inProgress, EObject a, EObject b) {
			System.out.println("Comparing " + a + " " + b);
			float[] v1 = getVector(a);			
			float[] v2 = getVector(b);
			if (v1 == null || v2 == null) {
				System.out.println("v1: " + v1 + " - v2: " + v2);
				System.out.println(v1);
				System.out.println(v2);
				return Double.MAX_VALUE;
			}
				// return super.distance(inProgress, a, b);
			
			float v = 1 - VectorUtils.cosine(v1, v2);
			
			//this.uriDistance.setComparison(inProgress);
//			double maxDist = Math.max(getThresholdAmount(a), getThresholdAmount(b));
//			double measuredDist = new CountingDiffEngine(maxDist, this.fakeComparison)
//					.measureDifferences(inProgress, a, b);

			System.out.println("Distance: " + v);
			if (v > 0.5) {
				System.out.println("Max!");
				return Double.MAX_VALUE;
			}
			
			return v;

		}

		@Override
		public boolean areIdentic(Comparison inProgress, EObject a, EObject b) {
			return super.areIdentic(inProgress, a, b);
		}
		
		public float[] getVector(EObject o) {
			//return this.strategy.toNormalizedVector(new EmbeddedEObject(o));
			return this.strategy.toVectorOrNull(new EmbeddedEObject(o));
		}
	}

	private static class EmbeddedEObject implements BagOfWords {

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
		public List<? extends String> getWords() {
			return strings;
		}
		
	}
	
}
