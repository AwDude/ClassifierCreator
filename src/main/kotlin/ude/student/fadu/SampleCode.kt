package ude.student.fadu

import weka.classifiers.Evaluation
import weka.classifiers.trees.J48
import weka.core.DenseInstance
import weka.core.Instances
import weka.core.SerializationHelper
import weka.core.converters.ConverterUtils
import java.util.*

class SampleCode {

    // Function for loading a ARFF file and returning the contained dataset
    fun initDataSet(): Instances {

        // Loading the data contained in the file "dataset.arff"
        val dataSource = ConverterUtils.DataSource("data_categorical.arff")

        // Return the contained dataset inside the datasource
        return dataSource.dataSet.apply {

            // Before returning, check if the class index, indicating the target attribute used for classification, is specified
            if (classIndex() == -1) {

                // If not, define the last attribute of each record as classlabel, which is the default of an ARFF file
                setClassIndex(numAttributes() - 1)
            }
        }
    }

    private fun evaluateClassifier() {

        // number of folds (splits) used for cross validation
        val folds = 10

        // get the dataset by using the previously defined function
        val dataSet = initDataSet()

        // instantiate the classifier model. J48 refers to the C4.5 DTC induction algorithm
        val classifier = J48()

        // create an evaluation object on the given dataset used for executing the validation and storing the results
        val evaluation = Evaluation(dataSet)

        // initiate the cross validation
        evaluation.crossValidateModel(classifier, dataSet, folds, Random(1))

        // print the averaged evaluation results for the cross validation over all folds
        println(evaluation.toSummaryString())
    }


    private fun createAndExportModel() {

        // instantiate the classifier model. J48 refers to the C4.5 DTC induction algorithm
        val classifier = J48()

        // get the dataset by using the previously defined function
        val dataSet = initDataSet()

        // train the classifier with the given dataset
        classifier.buildClassifier(dataSet)

        // serialize the model and store in a model file
        SerializationHelper.write("c45.model", classifier)

        // create header from dataset, containing all information about the attributes and the class label, but no records
        val header = Instances(dataSet, 0)

        // serialize the header and store it in a header file
        SerializationHelper.write("dataset.header", header)
    }


    private fun testPrediction() {
        val dataSet = initDataSet()
        val header = Instances(dataSet, 0)

        val classifier = J48()
        classifier.buildClassifier(dataSet)

        val instance = DenseInstance(7).apply {
            setValue(header.attribute(0), "no")
            setValue(header.attribute(1), "w")
            setValue(header.attribute(2), 20.0)
            setValue(header.attribute(3), "undergraduate")
            setValue(header.attribute(4), "full_time")
            setValue(header.attribute(5), "drugs_and_alcohol")
            setDataset(header)
        }

        val prediction = classifier.classifyInstance(instance)
        if (prediction == 0.0) {
            println("will regret")
        } else {
            println("will NOT regret")
        }
    }

}