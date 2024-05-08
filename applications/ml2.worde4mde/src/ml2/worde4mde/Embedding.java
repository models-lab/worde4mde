package ml2.worde4mde;

import java.util.List;

import mar.indexer.embeddings.Embeddable;
import mar.indexer.embeddings.EmbeddingStrategy;

public class Embedding {

	private EmbeddingStrategy strategy;

	public Embedding(EmbeddingStrategy strategy) {
		this.strategy = strategy;
	}

	public float[] toNormalizedVector(BagOfWords bag) {		
		return this.strategy.toNormalizedVector(new EmbeddableBag(bag));
	}
	
	public float[] toVectorOrNull(BagOfWords bag) {		
		return this.strategy.toVectorOrNull(new EmbeddableBag(bag));
	}
	
	private static class EmbeddableBag implements Embeddable {

		private BagOfWords bag;

		public EmbeddableBag(BagOfWords bag) {
			this.bag = bag;
		}

		@Override
		public int getSeqId() {
			throw new UnsupportedOperationException();
		}

		@Override
		public List<? extends String> getWords() {
			return bag.getWords();
		}
		
	}

}
