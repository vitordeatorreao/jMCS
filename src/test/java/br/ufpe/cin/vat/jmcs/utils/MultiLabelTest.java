package br.ufpe.cin.vat.jmcs.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import mulan.data.LabelsBuilderException;
import mulan.data.LabelsMetaData;
import weka.core.Attribute;
import weka.core.Instances;

public class MultiLabelTest
{
    @Test
    public void testGetLabelsMetaDataXMLString()
    {
        String[] names = { "amazed-suprised", "happy-pleased", "relaxing-calm",
                "quiet-still", "sad-lonely", "angry-aggresive" };
        String xml = MultiLabel.getLabelsMetaDataXMLString(names);
        String expected = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" +
                "<labels xmlns=\"http://mulan.sourceforge.net/labels\">" +
                "<label name=\"amazed-suprised\"/>" +
                "<label name=\"happy-pleased\"/>" +
                "<label name=\"relaxing-calm\"/>" +
                "<label name=\"quiet-still\"/>" +
                "<label name=\"sad-lonely\"/>" +
                "<label name=\"angry-aggresive\"/>" +
                "</labels>";
        assertEquals(expected, xml);
    }

    @Test
    public void testGetLabelsMetaData() throws UnsupportedEncodingException,
                                               LabelsBuilderException
    {
        // Create Instances programmatically
        // Declare numeric attributes
        Attribute attribute1 = new Attribute("numeric1");
        Attribute attribute2 = new Attribute("numeric2");
        
        // Declare labels
        List<String> label1 = new ArrayList<String>(2);
        label1.add("0");
        label1.add("1");
        Attribute attribute3 = new Attribute("nominal1", label1);
        List<String> label2 = new ArrayList<String>(2);
        label2.add("0");
        label2.add("1");
        Attribute attribute4 = new Attribute("nominal2", label2);

        // Create the list of attributes
        ArrayList<Attribute> attrInfo = new ArrayList<Attribute>(4);
        attrInfo.add(attribute1);
        attrInfo.add(attribute2);
        attrInfo.add(attribute3);
        attrInfo.add(attribute4);

        // Create the Data set
        Instances dataset = new Instances("Rel", attrInfo, 10);
        
        // Finally, try to get the labels meta data
        int[] labelsIndexes = {2, 3};
        LabelsMetaData metaData = MultiLabel.getLabelsMetaData(dataset, labelsIndexes);
        // Check if it is correct
        assertEquals(2, metaData.getNumLabels());
        assertEquals(false, metaData.isHierarchy());
        Set<String> names = metaData.getLabelNames();
        assertTrue(names.contains("nominal1"));
        assertTrue(names.contains("nominal2"));
    }
}
