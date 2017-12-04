/* DynamicClassifierSelection.java
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


package br.ufpe.cin.vat.jmcs.selection.dynamic;

import weka.classifiers.Classifier;
import weka.core.Instance;

/**
 * Interface for Multiple Classifier Systems that select a single classifier
 * among the available classifiers in the pool for classifying a given
 * test instance presented to the system.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public interface DynamicClassifierSelection extends DynamicSelection
{
    /**
     * Selects a classifier which should best classify the given test instance.
     * @param testInstance - The test instance to be classified by the Multiple
     * Classifier System.
     * @return The classifier to classify the given test instance.
     * @throws Exception - In case any of the WEKA classes throw some exception.
     * @since 0.1
     */
    Classifier selectClassifier(Instance testInstance) throws Exception;
}
