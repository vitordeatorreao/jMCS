package br.ufpe.cin.vat.jmcs;

import br.ufpe.cin.vat.jmcs.selection.dynamic.*;
import weka.classifiers.Classifier;
import weka.classifiers.meta.Vote;
import weka.core.Instances;

public class DynamicEnsembleSelectionApp extends MCSTestApp
{
    public static void main(String[] args) throws Exception
    {
        MCSTestApp app = new DynamicEnsembleSelectionApp();
        app.run(args);
    }

    @Override
    public DynamicSelection prepareSelector(Instances train,
            Instances validation, Classifier[] classifiers) throws Exception
    {
        // Major Vote combiner
        Vote vote = new Vote();
        for (Classifier classifier : classifiers) {
            vote.addPreBuiltClassifier(classifier);
        }
        // DynamicSelection
        DynamicEnsembleSelection selector = new KNORAEliminateDES();
        selector.setClassifiers(classifiers);
        selector.setCombiner(vote);
        // build the selector
        selector.buildClassifier(validation);
        return selector;
    }
}
