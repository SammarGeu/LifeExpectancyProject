package org.example.model;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;

public class DataLoader {

    private static final String CSV_FILE_PATH = "/Users/sammarsaini/Desktop/LifeExpectency/healthdata.csv";

    public Instances loadData() throws Exception {

        DataSource source = new DataSource(CSV_FILE_PATH);
        Instances data = source.getDataSet();


        int index = data.attribute("Life expectancy ").index();
        if (index >= 0) {
            data.setClassIndex(index);
        } else {
            throw new Exception("Class attribute 'Life expectancy ' not found in data.");
        }

        return data;
    }
}
