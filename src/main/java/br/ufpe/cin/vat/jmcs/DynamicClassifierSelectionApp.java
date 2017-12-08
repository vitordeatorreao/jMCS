package br.ufpe.cin.vat.jmcs;

import br.ufpe.cin.vat.jmcs.selection.dynamic.*;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class DynamicClassifierSelectionApp extends MCSTestApp
{
    public static void main(String[] args) throws Exception
    {
        MCSTestApp app = new DynamicClassifierSelectionApp();
        app.run(args);
    }

    @Override
    public DynamicSelection prepareSelector(Instances train,
            Instances validation, Classifier[] classifiers) throws Exception
    {
        // DynamicSelection
        DynamicClassifierSelection selector = new LocalClassAccuracyDCS();
        selector.setClassifiers(classifiers);
        // build the selector
        selector.buildClassifier(validation);
        return selector;
    }
}
