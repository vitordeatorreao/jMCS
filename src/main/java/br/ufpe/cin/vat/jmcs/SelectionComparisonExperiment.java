package br.ufpe.cin.vat.jmcs;

import java.io.File;
import java.util.Random;

import br.ufpe.cin.vat.jmcs.selection.dynamic.DynamicSelection;
import br.ufpe.cin.vat.jmcs.selection.dynamic.DynamicSelectionDCS;
import br.ufpe.cin.vat.jmcs.selection.dynamic.DynamicVoting;
import br.ufpe.cin.vat.jmcs.selection.dynamic.DynamicVotingSelectionDES;
import br.ufpe.cin.vat.jmcs.selection.dynamic.KNORAEliminateDES;
import br.ufpe.cin.vat.jmcs.selection.dynamic.LocalClassAccuracyDCS;
import br.ufpe.cin.vat.jmcs.selection.dynamic.MCBBasedDCS;
import br.ufpe.cin.vat.jmcs.selection.dynamic.MultiLabelDES;
import br.ufpe.cin.vat.jmcs.selection.dynamic.OverallLocalAccuracyDCS;
import br.ufpe.cin.vat.jmcs.utils.Labels;
import br.ufpe.cin.vat.jmcs.utils.Statistics;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.CalibratedLabelRanking;
import weka.classifiers.Classifier;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.classifiers.meta.Vote;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public final class SelectionComparisonExperiment
{
    private static class MajorVotingNoSelection implements DynamicSelection
    {
        private MultipleClassifiersCombiner combiner;
        private Classifier[] classifiers;
        
        public MajorVotingNoSelection()
        {
            this.combiner = new Vote();
            this.classifiers = new Classifier[0];
        }

        @Override
        public void buildClassifier(Instances data) throws Exception
        {
            return;
        }

        @Override
        public double classifyInstance(Instance instance) throws Exception
        {
            this.combiner.setClassifiers(this.classifiers);
            return this.combiner.classifyInstance(instance);
        }

        @Override
        public double[] distributionForInstance(Instance instance)
                throws Exception
        {
            this.combiner.setClassifiers(this.classifiers);
            return this.combiner.distributionForInstance(instance);
        }

        @Override
        public Capabilities getCapabilities()
        {
            throw new UnsupportedOperationException(
                    "getCapabilities is not implemented for the class " +
                    MajorVotingNoSelection.class.getName());
        }

        @Override
        public void setClassifiers(Classifier[] classifiers)
        {
            this.classifiers = classifiers;
        }

        @Override
        public Classifier[] getClassifiers()
        {
            return this.classifiers;
        }
        
    }

    public enum SelectionAlgorithm
    {
        MLKNN, CLR, OLA, LCA, DV, DS, DVS, KNORAE, MCB, MV
    }

    private static final double MULTILABEL_THRESHOLD = 0.7;

    public static DynamicSelection prepareSelector(Instances train,
            Instances validation, Classifier[] classifiers,
            SelectionAlgorithm alg) throws Exception
    {
        DynamicSelection selector;
        switch (alg) {
        case MLKNN:
            MultiLabelDES mlknn = new MultiLabelDES();
            mlknn.setThreshold(MULTILABEL_THRESHOLD);
            MLkNN knn = new MLkNN(10, 1);
            mlknn.setMultiLabelAlgorithm(knn);
            selector = mlknn;
            break;
        case CLR:
            MultiLabelDES mlclr = new MultiLabelDES();
            mlclr.setThreshold(MULTILABEL_THRESHOLD);
            CalibratedLabelRanking clr = new CalibratedLabelRanking();
            mlclr.setMultiLabelAlgorithm(clr);
            selector = mlclr;
            break;
        case LCA:
            selector = new LocalClassAccuracyDCS();
            break;
        case DS:
            selector = new DynamicSelectionDCS();
            break;
        case MCB:
            selector = new MCBBasedDCS();
            break;
        case DV:
            selector = new DynamicVoting();
            break;
        case DVS:
            selector = new DynamicVotingSelectionDES();
            break;
        case KNORAE:
            selector = new KNORAEliminateDES();
            break;
        case MV:
            selector = new MajorVotingNoSelection();
            break;
        case OLA:
        default:
            selector = new OverallLocalAccuracyDCS();
            break;
        }
        selector.setClassifiers(classifiers);
        selector.buildClassifier(validation);
        return selector;
    }

    public static double[] evaluate(String filePath, SelectionAlgorithm alg,
            Integer classIndex) throws Exception
    {
        DataSource source = new DataSource(filePath);
        Instances instances = source.getDataSet();
        if (classIndex == null) {
            instances.setClassIndex(instances.numAttributes() - 1);
        } else {
            instances.setClassIndex(classIndex);
        }
        Random rand  = new Random(100);
        Random rand2 = new Random(400);
        instances.randomize(rand);
        instances.stratify(10);
        double[] accuracies = new double[10];
        for (int i = 0; i < 10; i++) {
            // split outter cross val
            Instances trainVal = instances.trainCV(10, i);
            Instances test  = instances.testCV(10, i);
            // split inner val
            trainVal.randomize(rand2);
            trainVal.stratify(10);
            Instances train = trainVal.trainCV(10, 0);
            Instances validation = trainVal.testCV(10, 0);
            // generate and train
            Classifier[] initialPool = MultiLabelExperiment
                                             .generateInitialPool(train.size());
            for (Classifier classifier : initialPool) {
                if (classifier == null) System.out.println(
                        "ERROR! Classifier is null");
                classifier.buildClassifier(train);
            }
            DynamicSelection selector = prepareSelector(train, validation,
                                                        initialPool, alg);
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
        return accuracies;
    }

    public static void main(String[] args) throws Exception
    {
        if (args.length < 1) {
            System.out.println("USAGE: arg1 = datasets folder filepath.");
            return;
        }
        File folder = new File(args[0]);
        if (!folder.isDirectory()) {
            System.out.println("The path " + args[0] + " is not a folder!");
            return;
        }
        SelectionAlgorithm[] algorithms = new SelectionAlgorithm[]
                { SelectionAlgorithm.MLKNN, SelectionAlgorithm.CLR,
                  SelectionAlgorithm.OLA, SelectionAlgorithm.LCA,
                  SelectionAlgorithm.DV, SelectionAlgorithm.DS,
                  SelectionAlgorithm.DVS, SelectionAlgorithm.KNORAE,
                  SelectionAlgorithm.MCB, SelectionAlgorithm.MV };
        // CSV header
        System.out.print("file name;");
        for (int i = 0; i < algorithms.length; i++) {
            System.out.print(algorithms[i].toString());
            System.out.print(" (mean)");
            System.out.print(";");
            System.out.print(algorithms[i].toString());
            System.out.print(" (std)");
            if (i + 1 < algorithms.length) {
                System.out.print(";");
            }
        }
        System.out.println();
        // END CSV header
        for (File file : folder.listFiles()) {
            String filepath = file.getAbsolutePath();
            Integer classIndex;
            if (file.getName().equals("wine.arff")) {
                classIndex = 0;
            } else {
                classIndex = null;
            }
            System.out.print(file.getName() + ";");
            int count = 0;
            for (SelectionAlgorithm algorithm : algorithms) {
                double[] accuracies = evaluate(filepath, algorithm, classIndex);
                double mean = Statistics.Mean(accuracies);
                double std = Statistics.StandardDeviation(accuracies, mean);
                System.out.print(mean + ";" + std);
                count++;
                if (count < algorithms.length) {
                    System.out.print(";");
                }
            }
            System.out.println();
        }
    }
}
