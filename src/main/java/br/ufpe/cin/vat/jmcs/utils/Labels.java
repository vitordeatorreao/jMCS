/* Labels.java
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

/**
 * Utility class providing useful methods for applying to class Labels.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public final class Labels
{
    /**
     * Compares two labels. Return true if they represent the same class, and
     * false otherwise. (In WEKA, classes are always translated to doubles).
     * @param label1 - The first label to be compared.
     * @param label2 - The second label to be compared.
     * @return Whether or not the labels represent the same class.
     */
    public static boolean Equals(double label1, double label2)
    {
        return Math.abs(label1 - label2) < 0.0001;
    }
}
