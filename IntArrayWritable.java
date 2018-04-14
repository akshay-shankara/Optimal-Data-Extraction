package genetic_algorithm;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;

public class IntArrayWritable implements WritableComparable<IntArrayWritable> {
  private IntWritable[] values;
  private static Random r;

  public IntArrayWritable() {
    r = new Random(System.nanoTime());
  }

  public IntArrayWritable(IntWritable[] iw) {
    r = new Random(System.nanoTime());
    values = iw.clone();
  }

  public IntWritable[] getArray() {
    return values;
  }

  @Override
  public String toString() {
    String str = "";
    for (int i = 0; i < values.length; i++) {
      str += values[i].get() + "|";
    }
    return str;
  }

  public void readFields(DataInput in) throws IOException {
    values = new IntWritable[in.readInt()]; // construct values
    for (int i = 0; i < values.length; i++) {
      IntWritable value = new IntWritable();
      value.readFields(in); // read a value
      values[i] = value; // store it in values
    }
  }

  public void write(DataOutput out) throws IOException {
    out.writeInt(values.length); // write values
    for (int i = 0; i < values.length; i++) {
      values[i].write(out);
    }
  }

  public int compareTo(IntArrayWritable o) {
    // Compare two Ints randomly so that the output is shuffled randomly and not
    // according to their values
    if (r.nextBoolean())
      return -1;
    else
      return 1;
  }
}
