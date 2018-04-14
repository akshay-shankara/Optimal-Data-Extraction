package genetic_algorithm;

import java.io.IOException;

import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Partitioner;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.IdentityReducer;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

@SuppressWarnings("deprecation")
public class MapReduce extends Configured implements Tool {

	public static class InitializerMapper extends MapReduceBase
			implements Mapper<IntArrayWritable, FloatWritable, IntArrayWritable, FloatWritable> {
		Random rng;
		int num_of_features;
		IntWritable[] individual;

		@Override
		public void configure(JobConf jc) {
			num_of_features = Integer.parseInt(jc.get("ga.number_of_features"));
			rng = new Random(System.nanoTime());
			individual = new IntWritable[num_of_features];
		}

		public void map(IntArrayWritable key, FloatWritable value, OutputCollector<IntArrayWritable, FloatWritable> oc,
				Reporter rep) throws IOException {

			// Generate initial individuals
			for (int i = 0; i < value.get(); i++) {
				for (int l = 0; l < num_of_features; l++) {
					int ind = rng.nextBoolean() ? 0 : 1;
					individual[l] = new IntWritable(ind);
				}
				// Send the individual
				oc.collect(new IntArrayWritable(individual), new FloatWritable(0));
			}
		}
	}

	public static class GAMapper extends MapReduceBase
			implements Mapper<IntArrayWritable, FloatWritable, IntArrayWritable, FloatWritable> {
		float max_fitness = -1;
		IntArrayWritable max_individual;
		private String mapTaskId = "";
		float fit = 0;
		JobConf conf;
		int pop = 1;

		@Override
		public void configure(JobConf job) {
			conf = job;
			mapTaskId = job.get("mapred.task.id");
			pop = Integer.parseInt(job.get("ga.populationPerMapper"));
		}

		long fitness(IntWritable[] individual) {
			// Get fitness from mlib algorithm
			// Execute the python script
			Random tempFitness = new Random(System.nanoTime());
			return tempFitness.nextLong();
		}

		int processedInd = 0;

		public void map(IntArrayWritable key, FloatWritable value, OutputCollector<IntArrayWritable, FloatWritable> oc,
				Reporter rep) throws IOException {

			// Compute the fitness for every individual
			IntWritable[] individual = key.getArray();
			fit = fitness(individual);

			// Keep track of the maximum fitness
			if (fit > max_fitness) {
				max_fitness = fit;
				max_individual = new IntArrayWritable(individual);
			}

			// Write the Individual and fitness value
			oc.collect(key, new FloatWritable(fit));

			processedInd++;
			if (processedInd == pop) {
				closeAndWrite();
			}
		}

		public void closeAndWrite() throws IOException {
			// At the end of Map(), write the best found individual to a file
			Path tmpDir = new Path("GA");
			Path outDir = new Path(tmpDir, "global-map");

			// HDFS does not allow multiple mappers to write to the same file, hence create
			// one for each mapper
			Path outFile = new Path(outDir, mapTaskId);
			FileSystem fileSys = FileSystem.get(conf);
			SequenceFile.Writer writer = SequenceFile.createWriter(fileSys, conf, outFile, IntArrayWritable.class,
					FloatWritable.class, CompressionType.NONE);

			writer.append(max_individual, new FloatWritable(max_fitness));
			writer.close();
		}
	}

	// User-defined Partitioner
	@SuppressWarnings("hiding")
	public static class IndividualPartitioner<IntArrayWritable, FloatWritable>
			implements Partitioner<IntArrayWritable, FloatWritable> {

		// Partitions randomly independent of the passed <K, V>
		Random rng;

		public void configure(JobConf arg0) {
			rng = new Random(System.nanoTime());
		}

		public int getPartition(IntArrayWritable arg0, FloatWritable arg1, int numReducers) {
			return (Math.abs(rng.nextInt()) % numReducers);
		}
	}

	void launch(int numMaps, int numReducers, String jt, String dfs, int pop) {
		int num_of_features = 10;
		int it = 0;
		while (true) {
			JobConf jobConf = new JobConf(getConf(), MapReduce.class);

			// Set the Job properties
			jobConf.setSpeculativeExecution(true);
			jobConf.setInputFormat(SequenceFileInputFormat.class);
			jobConf.setOutputKeyClass(IntArrayWritable.class);
			jobConf.setOutputValueClass(FloatWritable.class);
			jobConf.setOutputFormat(SequenceFileOutputFormat.class);
			jobConf.set("ga.number_of_features", num_of_features + "");
			jobConf.setNumMapTasks(numMaps);
			jobConf.setPartitionerClass(IndividualPartitioner.class);
			jobConf.setJobName("ga-mr-" + it);
			if (jt != null) {
				jobConf.set("mapred.job.tracker", jt);
			}
			if (dfs != null) {
				FileSystem.setDefaultUri(jobConf, dfs);
			}

			System.out.println("launching");

			// Declare the directories
			Path tmpDir = new Path("GA");
			Path inDir = new Path(tmpDir, "iter" + it);
			Path outDir = new Path(tmpDir, "iter" + (it + 1));
			FileInputFormat.setInputPaths(jobConf, inDir);
			FileOutputFormat.setOutputPath(jobConf, outDir);

			FileSystem fileSys = null;
			try {
				fileSys = FileSystem.get(jobConf);
			} catch (IOException e1) {
				e1.printStackTrace();
			}
			int populationPerMapper = pop / numMaps;
			jobConf.set("ga.populationPerMapper", populationPerMapper + "");

			if (it == 0) {
				// Initialization
				try {
					fileSys.delete(tmpDir, true);
				} catch (IOException ie) {
					System.out.println("Exception while deleting");
					ie.printStackTrace();
				}
				System.out.println("Deleting dir");

				for (int i = 0; i < numMaps; ++i) {
					Path file = new Path(inDir, "part-" + String.format("%05d", i));
					SequenceFile.Writer writer = null;
					try {
						writer = SequenceFile.createWriter(fileSys, jobConf, file, IntArrayWritable.class,
								FloatWritable.class, CompressionType.NONE);
					} catch (Exception e) {
						System.out.println("Exception while instantiating writer");
						e.printStackTrace();
					}

					// Generate dummy input for all mappers
					IntWritable[] individual = new IntWritable[1];
					individual[0] = new IntWritable(populationPerMapper);
					try {
						writer.append(new IntArrayWritable(individual), new FloatWritable(populationPerMapper));
					} catch (Exception e) {
						System.out.println("Exception while appending to writer");
						e.printStackTrace();
					}

					try {
						writer.close();
					} catch (Exception e) {
						System.out.println("Exception while closing writer");
						e.printStackTrace();
					}
					System.out.println("Writing dummy input for Map #" + i);
				}
				jobConf.setMapperClass(InitializerMapper.class);
				jobConf.setReducerClass(IdentityReducer.class);
				jobConf.setNumReduceTasks(0);
			} // End of if it == 0
			else {
				jobConf.setMapperClass(GAMapper.class);
				try {
					fileSys.delete(outDir, true);
					fileSys.delete(new Path(tmpDir, "global-map"), true);
				} catch (IOException ie) {
					System.out.println("Exception while deleting");
					ie.printStackTrace();
				}
			}

			System.out.println("Starting Job");

			try {
				JobClient.runJob(jobConf);
			} catch (IOException e) {
				System.out.println("Exception while running job");
				e.printStackTrace();
			}
			System.out.println("Job done for: " + jobConf.getMapperClass());
			it++;

			if (it > numMaps) {
				break;
			}
		}
	}

	/**
	 * Launches all the tasks in order.
	 */
	public int run(String[] args) throws Exception {
		if (args.length != 3) {
			System.err.println("Usage: GeneticMR <nMaps> <nReducers> <population>");
			ToolRunner.printGenericCommandUsage(System.err);
			return -1;
		}

		// Set the command-line parameters
		int nMaps = Integer.parseInt(args[0]);
		int nReducers = Integer.parseInt(args[1]);
		int pop = Integer.parseInt(args[2]);

		System.out.println("Number of Maps = " + nMaps);

		launch(nMaps, nReducers, null, null, pop);

		return 0;
	}

	public static void main(String[] argv) throws Exception {
		int res = ToolRunner.run(new Configuration(), new MapReduce(), argv);
		System.exit(res);
	}
}