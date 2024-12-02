package org.example;

import org.example.model.DataLoader;
import org.example.model.RandomForestModel;
import weka.core.Instance;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        try {


            DataLoader loader = new DataLoader();
            Instances data = loader.loadData();


            Scanner scanner = new Scanner(System.in);
            System.out.print("Enter the country name for prediction: ");
            String countryName = scanner.nextLine().trim();


            Instance countryInstance = findInstanceByCountry(data, countryName);

            if (countryInstance == null) {
                System.out.println("Country not found in the dataset.");
                return;
            }


            RandomForestModel model = new RandomForestModel(data);
            model.trainModel();


            Instance newInstance = new DenseInstance(data.numAttributes());
            newInstance.setDataset(data);


            setValuesFromInstance(newInstance, countryInstance);


            double prediction = model.predict(newInstance);
            System.out.println("Predicted Life Expectancy for " + countryName + ": " + prediction);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private static Instance findInstanceByCountry(Instances data, String countryName) {
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            if (instance.stringValue(data.attribute("Country")).equalsIgnoreCase(countryName)) {
                return instance; // Return the first matching instance
            }
        }
        return null; // Country not found
    }


    private static void setValuesFromInstance(Instance target, Instance source) {
        for (int i = 0; i < source.numAttributes(); i++) {
            target.setValue(i, source.value(i)); // Copy values from source to target instance
        }
    }
}
