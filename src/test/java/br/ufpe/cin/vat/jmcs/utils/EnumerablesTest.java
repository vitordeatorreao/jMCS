package br.ufpe.cin.vat.jmcs.utils;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class EnumerablesTest
{
    @Test
    public void testMaxIndex()
    {
        int maxIndex = Enumerables.MaxIndex(null);
        assertTrue(maxIndex < 0);

        maxIndex = Enumerables.MaxIndex(new Integer[0]);
        assertTrue(maxIndex < 0);

        Integer[] array0 = { 2 };
        maxIndex = Enumerables.MaxIndex(array0);
        assertEquals(0, maxIndex);

        Integer[] array1 = { 7, 5, 3, 2, 0 };
        maxIndex = Enumerables.MaxIndex(array1);
        assertEquals(0, maxIndex);

        Integer[] array2 = { 7, 5, 3, 2, 9 };
        maxIndex = Enumerables.MaxIndex(array2);
        assertEquals(4, maxIndex);

        Integer[] array3 = { 7, 5, 9, 2, 3 };
        maxIndex = Enumerables.MaxIndex(array3);
        assertEquals(2, maxIndex);
        
        Integer[] array4 = { -2, -3, -1, -10 };
        maxIndex = Enumerables.MaxIndex(array4);
        assertEquals(2, maxIndex);

        Integer[] array5 = { 7, 7, 7, 7, 7 };
        maxIndex = Enumerables.MaxIndex(array5);
        assertEquals(0, maxIndex);
    }

    @Test
    public void testMinIndex()
    {
        int minIndex = Enumerables.MinIndex(null);
        assertTrue(minIndex < 0);

        minIndex = Enumerables.MinIndex(new Integer[0]);
        assertTrue(minIndex < 0);

        Integer[] array0 = { 2 };
        minIndex = Enumerables.MinIndex(array0);
        assertEquals(0, minIndex);

        Integer[] array1 = { -1, 5, 3, 2, 0 };
        minIndex = Enumerables.MinIndex(array1);
        assertEquals(0, minIndex);

        Integer[] array2 = { 7, 5, 3, 2, 9 };
        minIndex = Enumerables.MinIndex(array2);
        assertEquals(3, minIndex);

        Integer[] array3 = { 7, 1, 9, 2, 3 };
        minIndex = Enumerables.MinIndex(array3);
        assertEquals(1, minIndex);

        Integer[] array4 = { -2, -3, -1, -10 };
        minIndex = Enumerables.MinIndex(array4);
        assertEquals(3, minIndex);

        Integer[] array5 = { 7, 7, 7, 7, 7 };
        minIndex = Enumerables.MinIndex(array5);
        assertEquals(0, minIndex);
    }
}
