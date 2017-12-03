package br.ufpe.cin.vat.jmcs;

import java.util.Random;

import br.ufpe.cin.vat.jmcs.utils.Statistics;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        if (args.length < 1) {
            System.out.println("You must provide a first argument, which is " +
                               "the path to the dataset arff file.");
        }
        try {
            DataSource source = new DataSource(args[0]);
            Instances instances = source.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);
            Random rand = new Random(100);
            instances.randomize(rand);
            instances.stratify(10);
            double[] accuracies = new double[10];
            for (int i = 0; i < 10; i++) {
                // split
                Instances train = instances.trainCV(10, i);
                Instances test  = instances.testCV(10, i);
                // build classifier
                J48 j48 = new J48();
                j48.setUnpruned(true);
                j48.buildClassifier(train);
                // test it
                double agree = 0;
                for (int j = 0; j < test.size(); j++)
                {
                    double pred = j48.classifyInstance(test.instance(j));
                    String actual = test.classAttribute().value((int) test.instance(j).classValue());
                    String predicted = test.classAttribute().value((int) pred);
                    if (actual.equals(predicted)) agree++;
                }
                accuracies[i] = agree / test.size();
            }
            System.out.println("Mean accuracy = " + Statistics.Mean(accuracies));
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
