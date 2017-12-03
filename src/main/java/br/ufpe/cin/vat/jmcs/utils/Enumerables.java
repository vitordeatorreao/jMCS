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

public final class Enumerables
{
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
}
