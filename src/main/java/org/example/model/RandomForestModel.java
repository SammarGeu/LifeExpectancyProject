package org.example.model;

import weka.core.Instances;
import weka.core.Instance;
import weka.classifiers.trees.RandomForest;

public class RandomForestModel {

    private  final RandomForest rf;
    private  final Instances trainingData;

    
    public RandomForestModel(Instances trainingData) {
        if (trainingData == null || trainingData.numInstances() == 0) {
            throw new IllegalArgumentException("Training data must not be null or empty.");
        }
        this.trainingData = trainingData;
        this.rf = new RandomForest();
    }


    public void trainModel() throws Exception {
        try {
            rf.setNumIterations(100); // Set the number of trees (default is 100)
            rf.buildClassifier(trainingData); // Train the model with the data
            System.out.println("Model training completed successfully.");
        } catch (Exception e) {
            System.err.println("Error during model training: " + e.getMessage());
            throw e; // Propagate exception for handling at higher levels
        }
    }


    public double predict(Instance instance) throws Exception {
        if (instance == null) {
            throw new IllegalArgumentException("Instance for prediction must not be null.");
        }
        if (instance.classIndex() < 0) {
            throw new Exception("Instance class index is not set.");
        }

        try {
            double prediction = rf.classifyInstance(instance); // Perform the prediction
           // System.out.println("Prediction completed: " + prediction);
            return prediction;
        } catch (Exception e) {
            System.err.println("Error during prediction: " + e.getMessage());
            throw e; // Propagate exception for handling at higher levels
        }
    }
}
