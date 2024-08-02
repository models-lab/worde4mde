package org.modelslab.parser;

import org.eclipse.emf.common.util.TreeIterator;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.ENamedElement;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EStructuralFeature;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.uml2.uml.Property;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class ParserUML extends Parser{
    @Override
    public List<Item> parse(Model model) throws IOException {
        Resource r = model.getResource();
        String id = model.getId();
        List<Item> items = new ArrayList<>();
        TreeIterator<EObject> it = r.getAllContents();
        while (it.hasNext()) {
            EObject obj = it.next();
            EClass eclassContext = obj.eClass();
            String eClassName = eclassContext.getName();
            if (eClassName.equals("Package") || eClassName.equals("Model")){
                if (obj.eGet(getFeature(eclassContext, "name")) == null)
                    continue;
                String context = obj.eGet(getFeature(eclassContext, "name")).toString();
                List<String> recommendations = new ArrayList<>();
                Collection<EObject> elements = (Collection<EObject>) obj.eGet(getFeature(eclassContext,
                        "packagedElement"));
                for (EObject classifier: elements) {
                	if (classifier instanceof org.eclipse.uml2.uml.Classifier) {
                		String name = ((org.eclipse.uml2.uml.Classifier) classifier).getName();
                		if (name != null)
                			recommendations.add(name);
                	}
                }
                Item item = new Item(context, recommendations, "EPackage", id);
                items.add(item);
            } else if (eClassName.equals("Class")) {
                if (obj.eGet(getFeature(eclassContext, "name")) == null)
                    continue;
                String context = obj.eGet(getFeature(eclassContext, "name")).toString();
                List<String> recommendations = new ArrayList<>();
                Collection<EObject> elements = (Collection<EObject>) obj.eGet(getFeature(eclassContext,
                        "ownedAttribute"));
                for (EObject eStructuralFeature: elements){
                	if (eStructuralFeature instanceof Property) {
                		String name = ((Property) eStructuralFeature).getName();
                		if (name != null)
                			recommendations.add(name);
                	}
                }
                Item item = new Item(context, recommendations, "EClass", id);
                items.add(item);
            } else if (eClassName.equals("Enumeration")) {
                if (obj.eGet(getFeature(eclassContext, "name")) == null)
                    continue;
                String context = obj.eGet(getFeature(eclassContext, "name")).toString();
                List<String> recommendations = new ArrayList<>();
                Collection<EObject> elements = (Collection<EObject>) obj.eGet(getFeature(eclassContext,
                        "ownedLiteral"));
                for (EObject literal: elements){
                    recommendations.add(literal.eGet(getFeature(literal.eClass(),
                            "name")).toString());
                }
                Item item = new Item(context, recommendations, "EEnum", id);
                items.add(item);
            }
        }
        return items;
    }


    private EStructuralFeature getFeature(EClass eclass, String name){
    	
        for (EStructuralFeature feature: eclass.getEAllStructuralFeatures()) {
            if (feature.getName().equals(name))
                return feature;
        }
        
        throw new IllegalStateException("No feature " + name + " for " + eclass);
    }
}
