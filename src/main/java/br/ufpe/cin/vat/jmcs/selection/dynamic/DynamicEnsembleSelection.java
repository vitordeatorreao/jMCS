/* DynamicEnsembleSelection.java
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

import weka.classifiers.MultipleClassifiersCombiner;

/**
 * Interface for Multiple Classifier Systems that select a subset of the
 * available classifiers for classifying each test instance presented.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public interface DynamicEnsembleSelection
{
    /**
     * Configures the way the selected subset of classifiers will be combined
     * for providing a final answer to the classification task.
     * @param combiner - An instance implementing the way the selected subset
     * of classifiers will be combined for providing a final answer to the
     * classification task.
     * @since 0.1
     */
    void setCombiner(MultipleClassifiersCombiner combiner);

    /**
     * Retrives the instance implementing the way the selected subset of
     * classifiers will be combined for providing a final answer to the
     * classification task.
     * @return The instance implementing the way the selected subset of
     * classifiers will be combined for providing a final answer to the
     * classification task.
     * @since 0.1
     */
    MultipleClassifiersCombiner getCombiner();
}
