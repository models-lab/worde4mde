package org.modelslab.parser;

import java.io.File;
import java.io.IOException;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;

import mar.modelling.loader.ILoader;

public class Model {
    private final String fileName;
    private final String id;
	private ILoader loader;

    public Model(String fileName, String id, ILoader loader) {
        this.fileName = fileName;
        this.id = id;
        this.loader = loader;
    }

    public String getFileName() {
        return fileName;
    }

    public String getId() {
        return id;
    }

    public Resource getResource() throws IOException{
    	return loader.toEMF(new File(fileName));
        //ResourceSet rs = new ResourceSetImpl();
        //Resource resource = rs.getResource(URI.createFileURI(fileName), true);
        //return resource;
    }
}
