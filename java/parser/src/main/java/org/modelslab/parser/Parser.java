package org.modelslab.parser;

import org.eclipse.emf.ecore.resource.Resource;

import java.util.List;

public abstract class Parser {

    public abstract List<Item> parse(Resource r);

    public class Item {
        private final String context;
        private final List<String> recommendations;

        public Item(String context, List<String> recommendations) {
            this.context = context;
            this.recommendations = recommendations;
        }

        public String getContext() {
            return context;
        }

        public List<String> getRecommendations() {
            return recommendations;
        }
    }

}
