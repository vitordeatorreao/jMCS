package br.ufpe.cin.vat.jmcs.utils;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class LabelsTest
{
    @Test
    public void EqualsTest()
    {
        assertFalse(Labels.Equals(0.1, 0.2));
        assertFalse(Labels.Equals(-0.1, 0.1));
        assertFalse(Labels.Equals(0.1, -0.1));
        
        assertTrue(Labels.Equals(0.1, 0.1));
        assertTrue(Labels.Equals(1, 1));
        assertTrue(Labels.Equals(0, 0));
    }
}
