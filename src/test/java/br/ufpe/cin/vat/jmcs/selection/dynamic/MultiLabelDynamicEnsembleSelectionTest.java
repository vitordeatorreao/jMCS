package br.ufpe.cin.vat.jmcs.selection.dynamic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mockito;
import org.mockito.junit.MockitoJUnitRunner;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

@RunWith(MockitoJUnitRunner.class)
public class MultiLabelDynamicEnsembleSelectionTest {

    @Test
    public void testGetMultiLabelDataSet() throws Exception {
     // Create Instances programmatically
        // Declare numeric attributes
        Attribute attribute1 = new Attribute("numeric1");
        Attribute attribute2 = new Attribute("numeric2");
        
        // Declare labels
        List<String> label1 = new ArrayList<String>(2);
        label1.add("0");
        label1.add("1");
        Attribute attribute3 = new Attribute("nominal1", label1);
        
        ArrayList<Attribute> attrInfo = new ArrayList<Attribute>(3);
        attrInfo.add(attribute1);
        attrInfo.add(attribute2);
        attrInfo.add(attribute3);
        
        // Create the Data set
        Instances dataset = new Instances("Rel", attrInfo, 2);
        Instance inst1 = new DenseInstance(3);
        inst1.setValue(attribute1, 0.1);
        inst1.setValue(attribute2, 0.2);
        inst1.setValue(attribute3, "1");
        Instance inst2 = new DenseInstance(3);
        inst2.setValue(attribute1, 2.0);
        inst2.setValue(attribute2, 1.0);
        inst2.setValue(attribute3, "0");
        dataset.add(inst1);
        dataset.add(inst2);
        dataset.setClass(attribute3);
        
        // set the classifiers
        Classifier classifier1 = Mockito.mock(Classifier.class);
        Classifier classifier2 = Mockito.mock(Classifier.class);
        Mockito.when(classifier1.classifyInstance(Mockito.any(Instance.class)))
               .thenReturn(dataset.instance(0).classValue());
        Mockito.when(classifier2.classifyInstance(Mockito.any(Instance.class)))
               .thenReturn(dataset.instance(1).classValue());
        Classifier[] classifiers = { classifier1, classifier2 };
        
        // Set selector
        MultiLabelDES selector =
                new MultiLabelDES(classifiers);
        
        Instances multiLabel = selector.getMultiLabelDataSet(dataset);
        assertEquals(2, multiLabel.size());
        assertTrue(multiLabel.classIndex() < 0); // class not set
        assertEquals(4, multiLabel.numAttributes());
        Instance multiLabelinst1 = multiLabel.instance(0);
        assertEquals(0.1, multiLabelinst1.value(attribute1), 0.0001);
        assertEquals(0.2, multiLabelinst1.value(attribute2), 0.0001);
        assertEquals("1", multiLabelinst1.stringValue(2));
        assertEquals("0", multiLabelinst1.stringValue(3));
        Instance multiLabelinst2 = multiLabel.instance(1);
        assertEquals(2.0, multiLabelinst2.value(attribute1), 0.0001);
        assertEquals(1.0, multiLabelinst2.value(attribute2), 0.0001);
        assertEquals("0", multiLabelinst2.stringValue(2));
        assertEquals("1", multiLabelinst2.stringValue(3));
    }

}
