package genetic_algorithm;
import java.util.Random;

import org.apache.hadoop.io.RawComparator;

public class IntArraywritableComparator implements RawComparator<IntArrayWritable> {
	private static Random r;

	public IntArraywritableComparator() {
		r = new Random(System.nanoTime());
	}

	public int compare(byte[] arg0, int arg1, int arg2, byte[] arg3, int arg4, int arg5) {
		// Compare two Ints randomly so that the output is shuffled randomly and
		// not according to their values
		if (r.nextBoolean())
			return -1;
		else
			return 1;
	}

	public int compare(IntArrayWritable arg0, IntArrayWritable arg1) {
		// Compare two Ints randomly so that the output is shuffled randomly and
		// not according to their values
		if (r.nextBoolean())
			return -1;
		else
			return 1;
	}

}