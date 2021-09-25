package ude.student.fadu.evaluation

import weka.classifiers.Evaluation

private const val POSITIVE_CLASS = 0

enum class Metrics(val title: String, val description: String, val isBiggerBetter: Boolean = true) {
    CLASSIFIED_CORRECT("Correct", "Correctly Classified Instances.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.correct()
    },
    CLASSIFIED_INCORRECT("Incorrect", "Incorrectly Classified Instances.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.incorrect()
    },
    UNCLASSIFIED("Unclass.", "Unclassified Instances.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.unclassified()
    },
    TP("TP", "Incorrectly Classified Instances.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.numTruePositives(POSITIVE_CLASS)
    },
    FP("FP", "Incorrectly Classified Instances.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.numFalsePositives(POSITIVE_CLASS)
    },
    TN("TN", "Incorrectly Classified Instances.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.numTrueNegatives(POSITIVE_CLASS)
    },
    FN("FN", "Incorrectly Classified Instances.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.numFalseNegatives(POSITIVE_CLASS)
    },
    TPR("TP Rate", "True Positive Rate (Recall, Sensitivity).") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.truePositiveRate(POSITIVE_CLASS)
    },
    FPR("FP Rate", "False Positive Rate (Type I error).", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.falsePositiveRate(POSITIVE_CLASS)
    },
    TNR("TN Rate", "True Negative Rate (Specificity).") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.trueNegativeRate(POSITIVE_CLASS)
    },
    FNR("FN Rate", "False Negative Rate (Type II error).", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.falseNegativeRate(POSITIVE_CLASS)
    },
    W_TPR("w. TP Rate", "True Positive Rate (Recall, Sensitivity).") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedTruePositiveRate()
    },
    W_FPR("w. FP Rate", "False Positive Rate (Type I error). Weighted by class size.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedFalsePositiveRate()
    },
    W_TNR("w. TN Rate", "True Negative Rate (Specificity). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedTrueNegativeRate()
    },
    W_FNR("w. FN Rate", "False Negative Rate (Type II error). Weighted by class size.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedFalseNegativeRate()
    },
    ACCURACY("Accuracy", "Accuracy.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.pctCorrect() / 100.0
    },
    PRECISION("Precision", "Precision (Positive Predictive Value).") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.precision(POSITIVE_CLASS)
    },
    W_PRECISION("w. Precision", "Precision (Positive Predictive Value). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedPrecision()
    },
    F1_SCORE("F₁ Score", "F₁ Score.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.fMeasure(POSITIVE_CLASS)
    },
    W_F1_SCORE("w. F₁ Score", "F₁ Score. Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedFMeasure()
    },
    KAPPA("Kappa", "Cohen Kappa Coefficient.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.kappa()
    },
    MCC("MCC", "Matthews Correlation Coefficient.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.matthewsCorrelationCoefficient(POSITIVE_CLASS)
    },
    ROC_A("AUC", "Area under ROC (AUC).") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.areaUnderROC(POSITIVE_CLASS)
    },
    PRC_A("AUPRC", "Area under PRC (AUPRC).") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.areaUnderPRC(POSITIVE_CLASS)
    },
    W_MCC("w. MCC", "Matthews Correlation Coefficient. Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedMatthewsCorrelation()
    },
    W_ROC_A("w. AUC", "Area under ROC (AUC). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedAreaUnderROC()
    },
    W_PRC_A("w. AUPRC", "Area under PRC (AUPRC). Weighted by class size.") {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.weightedAreaUnderPRC()
    },
    MAE("MA Error", "Mean Absolute Error.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.meanAbsoluteError()
    },
    RMSE("RMS Error", "Root Mean Squared Error.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.rootMeanSquaredError()
    },
    RAE("RA Error", "Relative Absolute Error.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.relativeAbsoluteError() / 100.0
    },
    RRSE("RRS Error", "Root Relative Squared Error.", false) {
        override fun retrieveFrom(evaluation: Evaluation) = evaluation.rootRelativeSquaredError() / 100.0
    };

    abstract fun retrieveFrom(evaluation: Evaluation): Double
}