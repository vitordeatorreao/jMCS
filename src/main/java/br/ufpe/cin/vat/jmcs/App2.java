package br.ufpe.cin.vat.jmcs;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public final class App2
{
    public static void main(String[] args) throws Exception
    {
        MultiLabelInstances trainDataset = new MultiLabelInstances("C:\\Users\\vitor\\Documents\\DataSets\\emotions-train.arff",
                "C:\\Users\\vitor\\Documents\\DataSets\\emotions.xml");
        Instances testDataset = (new DataSource("C:\\Users\\vitor\\Documents\\DataSets\\emotions-test.arff")).getDataSet();
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
