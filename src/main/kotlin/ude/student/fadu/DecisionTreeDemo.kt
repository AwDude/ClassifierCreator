package ude.student.fadu

import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.trees.J48
import weka.core.DenseInstance
import weka.core.Drawable
import weka.core.Instances
import weka.core.SerializationHelper
import weka.core.converters.ConverterUtils
import weka.gui.treevisualizer.PlaceNode2
import weka.gui.treevisualizer.TreeVisualizer
import java.util.*
import javax.swing.JFrame
import javax.swing.SwingUtilities

fun main(args: Array<String>) {
    DecisionTreeDemo.process()
}

object DecisionTreeDemo {
    private const val CATEGORICAL_DATA_FILENAME = "data_categorical.arff"
    private const val CROSS_VALIDATION_FOLDS = 10

    // Function for loading a ARFF file and returning the contained dataset
    fun initDataSet(): Instances {

        // Loading the data contained in the file "dataset.arff"
        val dataSource = ConverterUtils.DataSource("data_categorical.arff")

        // Return the contained dataset inside the datasource
        return dataSource.dataSet.apply {

            /* Before returning, check if the class index, indicating the target attribute used for classification, is specified */
            if (classIndex() == -1) {

                /* If not, define the last attribute of each record as classlabel, which is the default of an ARFF file */
                setClassIndex(numAttributes() - 1)
            }
        }
    }

    private fun evaluate(classifier: Classifier): Evaluation {
        val dataSet = initDataSet()
        classifier.buildClassifier(dataSet)
        val eval = Evaluation(dataSet)
        eval.crossValidateModel(classifier, dataSet, CROSS_VALIDATION_FOLDS, Random(1))
        println(classifier::class.java.simpleName)
        println(eval.toSummaryString())
        return eval
    }

    private fun <T : Drawable> visualize(classifier: T) = SwingUtilities.invokeLater {
        try {
            val visualizeTree = TreeVisualizer(null, classifier.graph(), PlaceNode2())
            val jFrame = JFrame("Weka J48 Klassisfikator: Entscheidungsbaum")
            jFrame.setSize(600, 500)
            jFrame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            jFrame.contentPane.add(visualizeTree)
            jFrame.isVisible = true
            visualizeTree.fitToScreen()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun createModel() {
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

        //visualize(classifier)

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

    private fun evaluateAllClassifiers() {
        val classifiers = listOf(J48())
    }

    fun process() {

        ClassifierEvaluation().compareAll()

        // testPrediction()


        // createModel()

        //println(dataSet.firstInstance().toString())

        /*val classifier = J48() // = C4.5 algorithm
        evaluate(classifier)*/

        //println(J48().capabilities)
/*
        println(classifier)
        SerializationHelper.write("c45.model", classifier)
        SerializationHelper.write("evaluation.model", evaluation)
*/

        /*
        visualize(classifier)
        evaluate(DecisionStump())
        evaluate(HoeffdingTree())
        evaluate(LMT())
        evaluate(RandomForest())
        evaluate(RandomTree())
        evaluate(REPTree())

        println(classifier)
        println(eval.toMatrixString())
        println(eval.toClassDetailsString())
        */
    }
}
