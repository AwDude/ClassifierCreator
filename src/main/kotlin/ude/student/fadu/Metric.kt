package ude.student.fadu

import weka.classifiers.Evaluation

enum class Metric(val title: String, val description: String) {
    CLASSIFIED_CORRECT("Correct", "Correctly Classified Instances.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.correct()
    },
    CLASSIFIED_INCORRECT("Incorrect", "Incorrectly Classified Instances.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.incorrect()
    },
    UNCLASSIFIED("Unclass.", "Unclassified Instances.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.unclassified()
    },
    TPR("TP Rate", "True Positive Rate (Recall, Sensitivity). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedTruePositiveRate()
    },
    FPR("FP Rate", "False Positive Rate (Type I error). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedFalsePositiveRate()
    },
    TNR("TN Rate", "True Negative Rate (Specificity). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedTrueNegativeRate()
    },
    FNR("FN Rate", "False Negative Rate (Type II error). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedFalseNegativeRate()
    },
    ACCURACY("Accuracy", "Accuracy.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.pctCorrect() / 100.0
    },
    PRECISION("Precision", "Precision (Positive Predictive Value). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedPrecision()
    },
    F1_SCORE("F₁ Score", "F₁ Score. Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedFMeasure()
    },
    KAPPA("Kappa", "Cohen Kappa Coefficient.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.kappa()
    },
    MCC("MCC", "Matthews Correlation Coefficient. Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedMatthewsCorrelation()
    },
    ROC_A("AUC", "Area under ROC (AUC). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedAreaUnderROC()
    },
    PRC_A("AUPRC", "Area under PRC (AUPRC). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedAreaUnderPRC()
    },
    MAE("MA Error", "Mean Absolute Error.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.meanAbsoluteError()
    },
    RMSE("RMS Error", "Root Mean Squared Error.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.rootMeanSquaredError()
    },
    RAE("RA Error", "Relative Absolute Error.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.relativeAbsoluteError() / 100.0
    },
    RRSE("RRS Error", "Root Relative Squared Error.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.rootRelativeSquaredError() / 100.0
    };

    abstract fun retrieveFrom(evaluation: Evaluation): Double
}