package br.ufpe.cin.vat.jmcs;

import java.io.File;
import java.util.Random;

import br.ufpe.cin.vat.jmcs.selection.dynamic.DynamicSelection;
import br.ufpe.cin.vat.jmcs.selection.dynamic.MultiLabelDES;
import br.ufpe.cin.vat.jmcs.utils.Labels;
import br.ufpe.cin.vat.jmcs.utils.Statistics;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.CalibratedLabelRanking;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public final class MultiLabelExperiment
{
    public enum MultiLabelAlgorithm {
        MLKNN, CLR
    }

    // MLP configs
    private static final String[] MLP_HIDDEN_LAYER_NODES = { "1", "2", "4", "8",
                                                             "16" };
    private static final double[] MLP_MOMENTUM_TERMS = { 0.0, 0.2, 0.5, 0.9 };
    private static final double[] MLP_LEARNING_RATES = { 0.3, 0.6 };

    // SVM configs
    private static final String[] SVM_COMPLEXITY_VALUES = {
        "0.00001", "0.0001", "0.001", "0.01", "0.1", "1", "10", "100"
    };
    private static final String[] SVM_KERNELS = {
        "weka.classifiers.functions.supportVector.PolyKernel -E 2.0",
        "weka.classifiers.functions.supportVector.PolyKernel -E 3.0",
        "weka.classifiers.functions.supportVector.RBFKernel -G 0.001",
        "weka.classifiers.functions.supportVector.RBFKernel -G 0.005",
        "weka.classifiers.functions.supportVector.RBFKernel -G 0.01",
        "weka.classifiers.functions.supportVector.RBFKernel -G 0.05",
        "weka.classifiers.functions.supportVector.RBFKernel -G 0.1",
        "weka.classifiers.functions.supportVector.RBFKernel -G 0.5",
        "weka.classifiers.functions.supportVector.RBFKernel -G 1",
        "weka.classifiers.functions.supportVector.RBFKernel -G 2",
    };

    // Decision Tree configs
    private static final String[] DT_CONFIDENCE = {
        "0.1", "0.2", "0.3", "0.4", "0.5"
    };
    private static final String[] DT_LAPLACE_SMOOTHING = {
        "-A", ""
    };
    private static final String[] DT_REDUCED_ERROR_FOLDS = {
        "2", "3", "4", "5"
    };
    private static final String[] DT_MINIMUM_OBJECTS_PER_LEAF = {
        "2", "3"
    };

    // kNN configs
    private static final String[] KNN_WEIGHT = {
        "-I", "-F", ""
    };

    public static Classifier[] generateInitialPool(int trainSetLength)
            throws Exception
    {
        int totalSize = (MLP_HIDDEN_LAYER_NODES.length *
                         MLP_MOMENTUM_TERMS.length *
                         MLP_LEARNING_RATES.length) /* MLP configs */
                         + (SVM_COMPLEXITY_VALUES.length * SVM_KERNELS.length)
                         + (DT_LAPLACE_SMOOTHING.length *
                            (DT_REDUCED_ERROR_FOLDS.length +
                             DT_CONFIDENCE.length) +
                            DT_MINIMUM_OBJECTS_PER_LEAF.length)
                         + (20 * KNN_WEIGHT.length);
        Classifier[] classifiers = new Classifier[totalSize];
        int i = 0;
        int seed = 1000;
        for (String hiddenLayer : MLP_HIDDEN_LAYER_NODES) {
            for (double momentum : MLP_MOMENTUM_TERMS) {
                for (double learningRate : MLP_LEARNING_RATES) {
                    MultilayerPerceptron mlp = new MultilayerPerceptron();
                    mlp.setHiddenLayers(hiddenLayer);
                    mlp.setMomentum(momentum);
                    mlp.setLearningRate(learningRate);
                    mlp.setSeed(seed * i);
                    if (!mlp.getHiddenLayers().equals(hiddenLayer)) {
                        System.out.println("ERROR HIDDEN LAYERS!");
                    }
                    if (mlp.getLearningRate() != learningRate) {
                        System.out.println("ERROR LEARNING RATE!");
                    }
                    if (mlp.getMomentum() != momentum) {
                        System.out.println("ERROR MOMENTUM!");
                    }
                    classifiers[i] = mlp;
                    i++;
                }
            }
        }
        for (String complexity : SVM_COMPLEXITY_VALUES) {
            for (String kernel : SVM_KERNELS) {
                SMO svm = new SMO();
                svm.setOptions(new String[] { "-C", complexity, "-K", kernel,
                                              "-W", "" + (seed * i) });
                classifiers[i] = svm;
                i++;
            }
        }
        for (String laplaceSmoothing : DT_LAPLACE_SMOOTHING) {
            for (String postPruningConfidence : DT_CONFIDENCE) {
                J48 dt = new J48();
                dt.setOptions(new String[] { "-C", postPruningConfidence,
                                             laplaceSmoothing,
                                             "-Q", "" + (seed * i) });
                classifiers[i] = dt;
                i++;
            }
            for (String fold : DT_REDUCED_ERROR_FOLDS) {
                J48 dt = new J48();
                dt.setOptions(new String[] { "-R", "-N", fold,
                                             laplaceSmoothing,
                                             "-Q", "" + (seed * i) });
                classifiers[i] = dt;
                i++;
            }
        }
        for (String objsPerTree : DT_MINIMUM_OBJECTS_PER_LEAF) {
            J48 dt = new J48();
            dt.setOptions(new String[] { "-U", "-M", objsPerTree, "-Q",
                                         "" + (seed * i) });
            classifiers[i] = dt;
            i++;
        }
        int part = trainSetLength / 20;
        for(int k = 0; k < 20; k++) {
            for (String weight : KNN_WEIGHT) {
                IBk knn = new IBk();
                int nn = k * part + 1;
                knn.setOptions(new String[] { "-K", "" + nn, weight });
                classifiers[i] = knn;
                i++;
            }
        }
        return classifiers;
    }

    public static DynamicSelection prepareSelector(Instances train,
            Instances validation, Classifier[] classifiers,
            MultiLabelAlgorithm alg, Double threshold) throws Exception
    {
        // Major Vote combiner
        Vote vote = new Vote();
        // MultiLabel Algorithm
        MultiLabelLearner clr;
        switch (alg) {
        case CLR:
            clr = new CalibratedLabelRanking();
            break;
        case MLKNN:
        default:
            clr = new MLkNN(10, 1);
            break;
        }
        // DynamicSelection
        MultiLabelDES selector = new MultiLabelDES();
        if (threshold != null) {
            selector.setThreshold(threshold);
        }
        selector.setClassifiers(classifiers);
        selector.setCombiner(vote);
        selector.setMultiLabelAlgorithm(clr);
        // build the selector
        selector.buildClassifier(validation);
        return selector;
    }

    public static double[] evaluate(String filePath, MultiLabelAlgorithm alg,
            Double threshold, Integer classIndex) throws Exception
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
            Classifier[] initialPool = generateInitialPool(train.size());
            for (Classifier classifier : initialPool) {
                if (classifier == null) System.out.println(
                        "ERROR! Classifier is null");
                classifier.buildClassifier(train);
            }
            DynamicSelection selector = prepareSelector(train, validation,
                                                        initialPool, alg,
                                                        threshold);
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
        if (args.length < 2) {
            System.out.println(args.length);
            System.out.println("USAGE: arg1 = datasets folder filepath; " +
                               "arg2 = multilabel algorithm to use.");
            return;
        }
        File folder = new File(args[0]);
        if (!folder.isDirectory()) {
            System.out.println("The path " + args[0] + " is not a folder!");
            return;
        }
        MultiLabelAlgorithm alg = MultiLabelAlgorithm.valueOf(args[1]);
        Double[] thresholds = new Double[] { null, 0.5, 0.6, 0.7, 0.8, 0.9 };
        System.out.print("file name;");
        for (int i = 0; i < thresholds.length; i++) {
            System.out.print(thresholds[i] != null ? thresholds[i] : "default");
            System.out.print(" (mean)");
            System.out.print(";");
            System.out.print(thresholds[i] != null ? thresholds[i] : "default");
            System.out.print(" (std)");
            if (i + 1 < thresholds.length) {
                System.out.print(";");
            }
        }
        System.out.println();
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
            for (Double threshold : thresholds) {
                double[] accuracies = evaluate(filepath, alg, threshold,
                                               classIndex);
                double mean = Statistics.Mean(accuracies);
                double std = Statistics.StandardDeviation(accuracies, mean);
                System.out.print(mean + ";" + std);
                count++;
                if (count < thresholds.length) {
                    System.out.print(";");
                }
            }
            System.out.println();
        }
    }
}
