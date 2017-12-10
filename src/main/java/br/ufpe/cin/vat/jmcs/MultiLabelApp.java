package br.ufpe.cin.vat.jmcs;

import br.ufpe.cin.vat.jmcs.selection.dynamic.DynamicSelection;
import br.ufpe.cin.vat.jmcs.selection.dynamic.MultiLabelDES;
import mulan.classifier.transformation.CalibratedLabelRanking;
import weka.classifiers.Classifier;
import weka.classifiers.meta.Vote;
import weka.core.Instances;

public class MultiLabelApp extends MCSTestApp
{
    public static void main(String[] args) throws Exception
    {
        MCSTestApp app = new MultiLabelApp();
        app.run(args);
    }

    @Override
    public DynamicSelection prepareSelector(Instances train,
            Instances validation, Classifier[] classifiers) throws Exception
    {
        // Major Vote combiner
        Vote vote = new Vote();
        // MultiLabel Algorithm
        CalibratedLabelRanking clr = new CalibratedLabelRanking();
        // DynamicSelection
        MultiLabelDES selector = new MultiLabelDES();
        selector.setThreshold(0.9);
        selector.setClassifiers(classifiers);
        selector.setCombiner(vote);
        selector.setMultiLabelAlgorithm(clr);
        // build the selector
        selector.buildClassifier(validation);
        return selector;
    }
}
