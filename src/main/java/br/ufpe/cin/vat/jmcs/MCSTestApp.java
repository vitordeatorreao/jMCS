package br.ufpe.cin.vat.jmcs;

import java.util.Random;

import br.ufpe.cin.vat.jmcs.selection.dynamic.DynamicSelection;
import br.ufpe.cin.vat.jmcs.utils.Labels;
import br.ufpe.cin.vat.jmcs.utils.Statistics;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public abstract class MCSTestApp
{
    public abstract DynamicSelection prepareSelector(
        Instances train, Instances validation, Classifier[] classifiers)
        throws Exception;
    
    public void run(String[] args) throws Exception
    {
        if (args.length < 1) {
            System.out.println("You must provide a first argument, which is " +
                               "the path to the dataset arff file.");
            return;
        }
        DataSource source = new DataSource(args[0]);
        Instances instances = source.getDataSet();
        instances.setClassIndex(instances.numAttributes() - 1);
        Random rand  = new Random(100);
        Random rand2 = new Random(400);
        instances.randomize(rand);
        instances.stratify(10);
        J48 j48 = new J48();
        IBk knn = new IBk();
        NaiveBayes nb = new NaiveBayes();
        knn.setKNN(10);
        j48.setUnpruned(true);
        double[] accuracies = new double[10];
        for (int i = 0; i < 10; i++) {
            // split
            Instances trainVal = instances.trainCV(10, i);
            Instances test  = instances.testCV(10, i);
            // Split 90% train, 10% validation
            int trainSize = (int) Math.round(trainVal.numInstances() * 0.9);
            int valSize   = trainVal.numInstances() - trainSize; 
            trainVal.randomize(rand2);
            Instances train = new Instances(trainVal, 0, trainSize);
            Instances validation = new Instances(trainVal, trainSize, valSize);
            // train the classifiers
            j48.buildClassifier(train);
            knn.buildClassifier(train);
            nb.buildClassifier(train);
            // DynamicSelection
            DynamicSelection selector = this.prepareSelector(train, validation,
                    new Classifier[] { j48, knn, nb});
            // test it
            double agree = 0;
            for (int j = 0; j < test.size(); j++)
            {
                double pred = selector.classifyInstance(test.instance(j));
                double actual = test.get(j).classValue();
                if (Labels.Equals(pred, actual)) agree++;
            }
            accuracies[i] = agree / test.size();
        }
        System.out.println("Mean accuracy = " + Statistics.Mean(accuracies));
    }
}
