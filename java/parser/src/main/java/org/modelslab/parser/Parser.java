package org.modelslab.parser;

import java.io.IOException;
import java.util.List;

public abstract class Parser {

    public abstract List<Item> parse(Model model) throws IOException;

}
