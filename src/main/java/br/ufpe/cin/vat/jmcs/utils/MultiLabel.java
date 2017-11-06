/* MulanUtility.java
 * Copyright (C) 2017  Vitor de Albuquerque Torreao
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package br.ufpe.cin.vat.jmcs.utils;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.nio.charset.StandardCharsets;

import mulan.data.LabelsBuilder;
import mulan.data.LabelsBuilderException;
import mulan.data.LabelsMetaData;
import weka.core.Instances;

/**
 * Utility class containing static methods for bridging the gap between WEKA
 * and MULAN.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public final class MultiLabel
{
    /**
     * MULAN's XML header for label meta data 
     */
    private static final String LabelMetaDataXMLHeader =
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" +
            "<labels xmlns=\"http://mulan.sourceforge.net/labels\">";

    /**
     * MULAN's XML footer for label meta data
     */
    private static final String LabelMetaDataXMLFooter = "</labels>";

    /**
     * Returns the Labels' meta data in XML format.
     * @param names - The names of the attributes that actually are the labels.
     * @return The labels' meta data in a XML string following MULAN's
     * specification.
     */
    public static String getLabelsMetaDataXMLString(String[] names)
    {
        StringBuilder xml = new StringBuilder();
        xml.append(LabelMetaDataXMLHeader);
        for (String name : names)
        {
            xml.append(String.format("<label name=\"%s\"/>", name));
        }
        xml.append(LabelMetaDataXMLFooter);
        return xml.toString();
    }

    /**
     * Returns the meta data for the Labels in the specified attributes of the
     * given data set. 
     * @param dataset - The data set containing the attribute information. 
     * @param labelsIndexes - The indexes for the columns containing the
     * multiple labels.
     * @return MULAN's LabelsMetaData for the specified columns.
     * @throws UnsupportedEncodingException - In case something went wrong and
     * the code's inner string wasn't really UTF-8 when it should have been.
     * @throws LabelsBuilderException - In case there was a problem parsing the
     * meta data from the specified columns.
     */
    public static LabelsMetaData getLabelsMetaData(
            Instances dataset, int[] labelsIndexes)
            throws UnsupportedEncodingException, LabelsBuilderException
    {
        String[] names = new String[labelsIndexes.length];
        for (int i = 0; i < labelsIndexes.length; i++)
        {
            names[i] = dataset.attribute(labelsIndexes[i]).name();
        }
        String xml = getLabelsMetaDataXMLString(names);
        InputStream input = new ByteArrayInputStream(
                xml.getBytes(StandardCharsets.UTF_8.name()));
        return LabelsBuilder.createLabels(input);
    }
}
