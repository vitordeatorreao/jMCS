/* Enumerables.java
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
 * Utility class providing many useful methods for arrays.
 * @author vitordeatorreao
 * @since 0.1
 *
 */
public final class Enumerables
{
    /**
     * Retrieves the position of the element with the greatest value in the
     * given array.
     * @param array - The array where the search will be carried.
     * @return The position of the greatest element in the array.
     * @since 0.1
     */
    public static <T extends Object & Comparable<? super T>> int MaxIndex(
            T[] array)
    {
        if (array == null) return -1;
        int i = 0;
        int max = -1;
        T maxValue = null;
        for (T value : array) {
            if (maxValue == null || maxValue.compareTo(value) < 0) {
                maxValue = value;
                max = i;
            }
            i++;
        }
        return max;
    }

    /**
     * Retrieves the position of the element with the least value in the
     * given array.
     * @param array - The array where the search will be carried.
     * @return The position of the least element in the array.
     */
    public static <T extends Object & Comparable<? super T>> int MinIndex(
            T[] array)
    {
        if (array == null) return -1;
        int i = 0;
        int min = -1;
        T minValue = null;
        for (T value : array) {
            if (minValue == null || minValue.compareTo(value) > 0) {
                minValue = value;
                min = i;
            }
            i++;
        }
        return min;
    }
}
