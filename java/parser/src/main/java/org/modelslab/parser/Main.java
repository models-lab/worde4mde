package org.modelslab.parser;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import me.tongfei.progressbar.ProgressBar;
import org.apache.commons.cli.*;
import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;

public class Main {
    private static final Logger logger = LogManager.getLogger(Main.class);
    private static final String MODELSET_HOME = Paths.get(System.getProperty("user.home"),
            ".modelset", "modelset").toString();
    private static final String OUT_DIR_DEFAULT = Paths.get(System.getProperty("user.dir"),
            "out").toString();

    public static void main(String[] args) throws IOException, ParseException {
        // logger
        BasicConfigurator.configure();

        // set up emf
        Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap( ).put("*", new XMIResourceFactoryImpl());

        // Options
        Options options = setUpOptions();
        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);
        String modelSetPath = cmd.getOptionValue("modelsetPath", MODELSET_HOME);
        String outdir = cmd.getOptionValue("outdir", OUT_DIR_DEFAULT);

        // Query database
        List<String> fileNames = getFileNames(modelSetPath);
        Parser ecoreParser = new ParserEcore();
        int counter = 0;
        for (String fileName : ProgressBar.wrap(fileNames, "Parsing procedure")) {
            ResourceSet rs = new ResourceSetImpl();
            Resource resource = rs.getResource(URI.createFileURI(fileName), true);
            List<Parser.Item> items = ecoreParser.parse(resource);

            // logger.debug("Number of items " + items.size());
            File out = Paths.get(outdir, Integer.toString(counter) + ".json").toFile();
            out.getParentFile().mkdirs();
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            Writer writer = new FileWriter(out);
            gson.toJson(items, writer);
            writer.flush();
            writer.close();
            ++counter;
        }

    }

    private static Options setUpOptions(){
        Options options = new Options();

        Option modelSetPath = new Option("p", "modelsetPath", true, "ModelSet path");
        options.addOption(modelSetPath);

        Option outPath = new Option("o", "outdir", true, "Output path");
        options.addOption(outPath);

        return options;
    }

    private static List<String> getFileNames(String modelSetPath){
        ArrayList<String> arrayList = new ArrayList<>();
        Path filePathDB = Paths.get(modelSetPath, "datasets", "dataset.ecore", "data", "ecore.db");
        Path rawData = Paths.get(modelSetPath, "raw-data", "repo-ecore-all");
        try {
            Connection dataset = DriverManager.getConnection("jdbc:sqlite:" + filePathDB.toString());
            PreparedStatement stm = dataset.prepareStatement("select mo.filename " +
                    "from models mo join metadata mm on mo.id = mm.id");
            stm.execute();
            ResultSet rs = stm.getResultSet();
            while (rs.next()) {
                String filename = rs.getString(1);
                arrayList.add(Paths.get(rawData.toString(), filename).toString());
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
        return arrayList;
    }
}