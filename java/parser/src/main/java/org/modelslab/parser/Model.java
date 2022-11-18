package org.modelslab.parser;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;

public class Model {
    private final String fileName;
    private final String id;

    public Model(String fileName, String id) {
        this.fileName = fileName;
        this.id = id;
    }

    public String getFileName() {
        return fileName;
    }

    public String getId() {
        return id;
    }

    public Resource getResource(){
        ResourceSet rs = new ResourceSetImpl();
        Resource resource = rs.getResource(URI.createFileURI(fileName), true);
        return resource;
    }
}
