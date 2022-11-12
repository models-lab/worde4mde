package org.modelslab.parser;

import org.eclipse.emf.common.util.TreeIterator;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EStructuralFeature;
import org.eclipse.emf.ecore.resource.Resource;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class ParserEcore extends Parser{
    @Override
    public List<Item> parse(Resource r) {
        List<Item> items = new ArrayList<>();
        TreeIterator<EObject> it = r.getAllContents();
        while (it.hasNext()) {
            EObject obj = it.next();
            EClass eclassContext = obj.eClass();
            String eClassName = eclassContext.getName();
            if (eClassName.equals("EPackage")){
                if (obj.eGet(getFeature(eclassContext, "name")) == null)
                    continue;
                String context = obj.eGet(getFeature(eclassContext, "name")).toString();
                List<String> recommendations = new ArrayList<>();
                Collection<EObject> elements = (Collection<EObject>) obj.eGet(getFeature(eclassContext,
                        "eClassifiers"));
                for (EObject classifier: elements){
                    recommendations.add(classifier.eGet(getFeature(eclassContext, "name")).toString());
                }
                Item item = new Item(context, recommendations);
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
        return null;
    }
}
