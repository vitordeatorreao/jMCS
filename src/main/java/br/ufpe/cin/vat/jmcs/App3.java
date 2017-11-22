package br.ufpe.cin.vat.jmcs;

import java.util.Random;

import br.ufpe.cin.vat.jmcs.selection.dynamic.MultiLabelDynamicEnsembleSelection;
import br.ufpe.cin.vat.jmcs.utils.Statistics;
import mulan.classifier.transformation.CalibratedLabelRanking;
import weka.classifiers.trees.J48;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Vote;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class App3
{
    public static void main(String[] args) throws Exception
    {
        try {
            DataSource source = new DataSource("C:\\Users\\vitor\\Documents\\DataSets\\vote.arff");
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
                // Major Vote combiner
                Vote vote = new Vote();
                vote.addPreBuiltClassifier(j48);
                vote.addPreBuiltClassifier(knn);
                vote.addPreBuiltClassifier(nb);
                // MultiLabel Algorithm
                CalibratedLabelRanking clr = new CalibratedLabelRanking();
                // DynamicSelection
                MultiLabelDynamicEnsembleSelection selector = new MultiLabelDynamicEnsembleSelection();
                selector.setThreshold(0.9);
                selector.setClassifiers(new Classifier[] { j48, knn, nb });
                selector.setCombiner(vote);
                selector.setMultiLabelAlgorithm(clr);
                // build the selector
                selector.buildClassifier(validation);
                // test it
                double agree = 0;
                for (int j = 0; j < test.size(); j++)
                {
                    double pred = selector.classifyInstance(test.instance(j));
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
