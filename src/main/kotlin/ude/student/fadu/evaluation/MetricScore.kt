package ude.student.fadu.evaluation

class MetricScore(val name: String, val score: Double) : Comparable<MetricScore> {

    override fun compareTo(other: MetricScore) = score.compareTo(other.score)

}