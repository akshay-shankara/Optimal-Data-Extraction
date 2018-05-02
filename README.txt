Genetic Algorithm in Map Reduce 


Initial Mapper - Generates the initial population specified as the argument

GAMapper - Obtains the fitness values from the python - mllib jobs (Decision Tree and Logistic Regression), for each of the individuals and assigns it to them.

GAReducer - Performs tournament selection on the individuals, selects the best individuals and performs their crossover along with their mutation.

launch() - Iterates over the genetic algorithm until the stopping criteria is achieved. The stopping criteria is achieved when the accuracy of the final individual does not change over 4 iterations.


Command format:
hadoop fs <jar> <main_class> -libjars spark-launcher_2.10-1.5.1.jar <number_of_mappers> <number_of_reducers> <population>

Example:
hadoop fs MapReduce.jar genetic_algorithms.MapReduce -libjars spark-launcher_2.10-1.5.1.jar 100 100 200