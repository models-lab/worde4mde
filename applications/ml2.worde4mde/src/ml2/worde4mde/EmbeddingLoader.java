package ml2.worde4mde;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import mar.indexer.embeddings.EmbeddingStrategy;

public class EmbeddingLoader {
	
	public static EmbeddingLoader INSTANCE = new EmbeddingLoader();
	
	private Map<String, File> models = new HashMap<>();
	private Map<String, EmbeddingStrategy> alreadyLoaded = new HashMap<>();
	
	public EmbeddingLoader() {
		//String fastText_mde = EmbeddingModel.FASTTEXT.name() + "-" + "mde";	
		//models.put(fastText_mde, getModel)
		
		String glove_mde = getEmbeddingId(EmbeddingModel.GLOVE, Corpus.MDE, 300);	
		models.put(glove_mde, getModelFile(Paths.get("glove_modelling", "vectors.txt")));

	}
	

	private File getModelFile(Path relativePath) {
		// TODO: Use an environment variable / configuraiton element
		Path path = Paths.get(System.getProperty("user.home"), ".worde4mde").resolve(relativePath);		
		return path.toFile();
	}

	private String getEmbeddingId(EmbeddingModel model, Corpus corpus, int size) {
		return model.name() + "-" + corpus.name() + "-" + size;
	}

	public EmbeddingStrategy load(EmbeddingModel type, Corpus corpus, int size) throws IOException {
		String id = getEmbeddingId(type, corpus, size);
		if (alreadyLoaded.containsKey(id))
			return alreadyLoaded.get(id);
		
		File path = models.get(id);
		if (path == null)
			throw new IllegalArgumentException("No model " + id);
		
		EmbeddingStrategy result;
		switch (type) {
		case GLOVE:
			System.out.println("Loading Glove " + path);
			result = new EmbeddingStrategy.GloveWordE(path);
			System.out.println("Loaded!");
			break;
		default:
			throw new IllegalArgumentException("No model supported " + type);
		}
		
		alreadyLoaded.put(id, result);
		return result;
	}
	
	public static enum EmbeddingModel {
		SGRAM,
		GLOVE,
		FASTTEXT
	}
	
	public static enum Corpus {
		MDE
	}

}
