package org.modelslab.parser;

import java.util.List;

public class Item {
    private final String context;
    private final List<String> recommendations;
    private final String contextType;
    private final String id;

    public Item(String context, List<String> recommendations, String contextType, String id) {
        this.context = context;
        this.recommendations = recommendations;
        this.contextType = contextType;
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public String getContext() {
        return context;
    }

    public List<String> getRecommendations() {
        return recommendations;
    }

    public String getContextType() {
        return contextType;
    }
}