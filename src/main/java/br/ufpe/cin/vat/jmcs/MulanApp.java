package br.ufpe.cin.vat.jmcs;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public final class MulanApp
{
    public static void main(String[] args) throws Exception
    {
        if (args.length < 3) {
            System.out.println("You must provide three arguments, which are: " +
                    "the path to the training dataset arff file, the path to " +
                    "the multilabel definition XML, and the path to the test " +
                    "dataset arff file.");
            return;
        }
        MultiLabelInstances trainDataset = new MultiLabelInstances(args[0], args[1]);
        Instances testDataset = (new DataSource(args[2])).getDataSet();
        MLkNN mlKnn = new MLkNN();
        CalibratedLabelRanking clr = new CalibratedLabelRanking();
        clr.build(trainDataset);
        mlKnn.build(trainDataset);
        for (int i = 0; i < testDataset.size(); i++)
        {
            MultiLabelOutput output = clr.makePrediction(testDataset.instance(i));
            System.out.println(output);
        }
    }
}
