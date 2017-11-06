/* DynamicSelection.java
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

/**
 * Interface for all Dynamic Selection techniques.
 * @author vitordeatorreao
 * @since 0.1
 */
public interface DynamicSelection
{
    /**
     * Sets the classifiers from the original pool, from which a subset will be
     * selected at each classifying step.
     * @param classifiers - The classifiers that compose the original pool.
     */
    void setClassifiers(Classifier[] classifiers);

    /**
     * Retrieves the classifiers from the original pool, from which a subset
     * will be selected at each classifying step.
     * @return - The classifiers that compose the original pool.
     */
    Classifier[] getClassifiers();
}
